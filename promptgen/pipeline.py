from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from heterogeneous.core import digest, normalized_hash, read_jsonl, text_hash, write_json, write_jsonl
from heterogeneous.providers import ProviderError
from .provider import complete
from .templates import Templates

VERSION = "1.1"


def seed_context(row: dict) -> list[dict]:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("seed conversations need a nonempty string id")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("seed conversations need messages ending in a completed assistant reply")
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("seed messages need nonempty text content")
        if index == 0 and message.get("role") == "system":
            continue
        if message.get("role") != expected:
            raise ValueError("seed conversations must alternate user and assistant, optionally after a system message")
        expected = "assistant" if expected == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError("seed conversation must end with an assistant reply")
    return [{"role": m["role"], "content": m["content"],
             "author": m.get("author", {"type": "unknown", "source": "seed"})} for m in messages]


def plain(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def make_plan(config: dict, templates: Templates, seeds: list[dict]) -> dict:
    contexts = [seed_context(row) for row in seeds]
    if len({row["id"] for row in seeds}) != len(seeds):
        raise ValueError("seed conversation ids must be unique")
    samples = []
    for i in range(config["generation"]["count"]):
        index = i % len(contexts) if contexts else None
        messages = [dict(m) for m in contexts[index]] if index is not None else []
        system = config["generation"]["system_prompt"]
        if system:
            if messages and messages[0]["role"] == "system":
                raise ValueError("use either the seed's system message or system_prompt, not both")
            messages.insert(0, {"role": "system", "content": system, "author": {"type": "configuration"}})
        samples.append({"id": f"sample-{i:08d}", "seed_id": seeds[index]["id"] if index is not None else None,
                        "seed_reference": seeds[index].get("reference") if index is not None else None,
                        "messages": messages})
    plan = {"schema_version": VERSION, "config": config, "tokenizer": templates.identity, "samples": samples}
    plan["id"] = digest(plan)
    # Verify the first prefix during planning, including tokenizer-specific constraints.
    preview = templates.user_prompt(plain(samples[0]["messages"]))
    if not templates.token_ids(preview):
        raise ValueError("chat template produced an empty prefix")
    return plan


def save_plan(plan: dict, out: Path) -> None:
    path = out / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("output directory contains a different plan; choose a new directory")
    write_json(path, plan)


def load_plan(out: Path) -> dict:
    plan = json.loads((out / "plan.json").read_text())
    if plan.get("schema_version") != VERSION or plan.get("id") != digest({k: v for k, v in plan.items() if k != "id"}):
        raise ValueError("plan version or hash mismatch")
    return plan


def quality_error(result: dict, templates: Templates) -> str | None:
    if result["provenance"]["finish_reason"] not in ("stop", "eos"):
        return "truncated_or_unfinished"
    if not result["text"].strip():
        return "empty"
    if any(token in result["text"] for token in templates.special_tokens):
        return "control_token_in_text"
    if re.match(r"\s*(user|assistant|system)\s*:", result["text"], re.I):
        return "role_label_in_text"
    return None


class Runner:
    def __init__(self, out: Path, plan: dict, templates: Templates):
        if plan["tokenizer"] != templates.identity:
            raise ValueError("tokenizer changed since planning; use the same pinned tokenizer")
        self.out, self.plan, self.templates = out, plan, templates

    def sample(self, sample: dict) -> dict:
        path = self.out / "cache" / f"{sample['id']}.json"
        state = {"plan_id": self.plan["id"], "sample_id": sample["id"], "steps": []}
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("plan_id") != self.plan["id"] or state.get("sample_id") != sample["id"]:
                raise ValueError("cache plan/sample mismatch")
        messages = [dict(m) for m in sample["messages"]]
        generated = []
        g, model = self.plan["config"]["generation"], self.plan["config"]["model"]
        roles = []
        for turn in range(g["user_turns"]):
            roles.append("user")
            if turn < g["user_turns"] - 1 or g["emit_final_response"]:
                roles.append("assistant")
        for step_index, role in enumerate(roles):
            prompt = (self.templates.user_prompt(plain(messages))
                      if role == "user" else self.templates.render(plain(messages), user=False))
            tokens = self.templates.token_ids(prompt)
            if step_index < len(state["steps"]):
                step = state["steps"][step_index]
                if step["role"] != role or step["prompt_sha256"] != text_hash(prompt):
                    raise ValueError("cache prompt/role mismatch")
            else:
                step = {"role": role, "prompt_sha256": text_hash(prompt), "attempts": [], "accepted": None}
                state["steps"].append(step)
            result = step["accepted"]
            while result is None and len(step["attempts"]) < g["attempts"]:
                sampling = {"max_tokens": g["user_max_tokens"] if role == "user" else g["response_max_tokens"],
                            "temperature": g["temperature"] if role == "user" else g["response_temperature"],
                            "top_p": g["top_p"],
                            "seed": int(digest([self.plan["id"], sample["id"], step_index, len(step["attempts"])])[:15], 16),
                            "stop_token_ids": self.templates.user_stops if role == "user" else self.templates.response_stops,
                            "stop": model["stop_strings"] if role == "user" else model["response_stop_strings"]}
                try:
                    candidate = complete(model, prompt, tokens, sampling)
                except ProviderError as error:
                    step["attempts"].append({"error": str(error), "retryable": error.retryable})
                    write_json(path, state)
                    if not error.retryable:
                        raise
                    continue
                reason = quality_error(candidate, self.templates)
                step["attempts"].append({"completion": candidate, "rejection_reason": reason})
                if reason is None:
                    result = candidate
                    step["accepted"] = result
                write_json(path, state)
            if result is None:
                return {"schema_version": VERSION, "id": digest([self.plan["id"], sample["id"]])[:24],
                        "sample_id": sample["id"], "status": "rejected", "reason": "attempts_exhausted",
                        "failed_role": role, "plan_id": self.plan["id"], "messages": messages, "steps": state["steps"]}
            if (result["prompt"] != prompt or result["provenance"]["prompt_token_ids"] != tokens
                    or result["provenance"]["prompt_sha256"] != text_hash(prompt)
                    or result["text_sha256"] != text_hash(result["text"])
                    or quality_error(result, self.templates)):
                raise ValueError("cached completion failed integrity/quality checks")
            message_index = len(messages)
            messages.append({"role": role, "content": result["text"].strip(), "author": result["provenance"]})
            generated.append({"message_index": message_index, "completion": result})
        return {"schema_version": VERSION, "id": digest([self.plan["id"], sample["id"]])[:24],
                "status": "accepted", "plan_id": self.plan["id"], "method": g["method"],
                "conditioned": bool(g["system_prompt"] or (sample["messages"] and sample["messages"][0]["role"] == "system")),
                "seed_id": sample["seed_id"], "seed_reference": sample["seed_reference"],
                "messages": messages, "generated": generated,
                "tokenizer": self.templates.identity, "sample_id": sample["id"]}

    def run(self, jobs: int) -> dict:
        accepted, rejected, seen = [], [], {}
        prompts = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            # Stable plan order makes deduplication independent of response timing.
            for row in pool.map(self.sample, self.plan["samples"]):
                if row["status"] == "rejected":
                    rejected.append(row)
                    continue
                keys = []
                for item in row["generated"]:
                    i = item["message_index"]
                    message = row["messages"][i]
                    if message["role"] == "user":
                        key = digest([plain(row["messages"][:i]), normalized_hash(message["content"])])
                        keys.append(key)
                duplicate = next((seen[key] for key in keys if key in seen), None)
                if duplicate:
                    rejected.append({**row, "status": "duplicate", "duplicate_of": duplicate})
                    continue
                for key in keys:
                    seen[key] = row["id"]
                accepted.append(row)
                for item in row["generated"]:
                    i = item["message_index"]
                    message = row["messages"][i]
                    if message["role"] == "user":
                        prompts.append({"schema_version": VERSION, "id": f"{row['id']}:{i}",
                            "conversation_id": row["id"], "seed_id": row["seed_id"], "plan_id": row["plan_id"],
                            "method": row["method"],
                            "context": row["messages"][:i], "prompt": message["content"], "author": message["author"]})
                print(f"{row['sample_id']}: accepted", flush=True)
        write_jsonl(self.out / "conversations.jsonl", accepted)
        write_jsonl(self.out / "prompts.jsonl", prompts)
        write_jsonl(self.out / "rejected.jsonl", rejected)
        report = {"plan_id": self.plan["id"], "candidates": len(self.plan["samples"]),
                  "accepted_conversations": len(accepted), "user_prompts": len(prompts),
                  "rejected_or_duplicate": len(rejected), "out": str(self.out.resolve())}
        write_json(self.out / "manifest.json", report)
        return report
