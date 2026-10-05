"""Run a sequential model sweep entirely on the GPU host.

Launch this process detached from SSH. It owns only the server it starts,
records failures separately from generated text, and checkpoints every condition.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from heterogeneous.core import digest, read_jsonl, write_json
from .config import normalize_config
from .pipeline import Runner, make_plan, plain, save_plan
from .templates import Templates, load_tokenizer
from .provider import complete


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=30)
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    # The server parent can exit while GPU workers remain alive in its session.
    # Always finish cleaning its own group before starting the next model.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


def ready(process: subprocess.Popen, url: str, timeout: int = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited {process.returncode}; inspect server.log")
        try:
            with urlopen(url + "/models", timeout=5) as handle:
                json.load(handle)
                return
        except (URLError, TimeoutError, ConnectionError, ValueError):
            time.sleep(3)
    raise RuntimeError("vLLM did not become ready within 20 minutes")


def assistant_controls(directory: Path, config: dict, templates: Templates) -> dict:
    """Ordinary assistant inference, kept separate from direct user sampling."""
    cases = [("ready", "Reply with exactly the word READY.", "READY"),
             ("arithmetic", "What is 17 times 19? Give the final integer.", "323"),
             ("explanation", "Explain in two sentences why rain falls from clouds.", None)]
    records = []
    for name, question, expected in cases:
        path = directory / "assistant-controls" / f"{name}.json"
        prompt = templates.render([{"role": "user", "content": question}], user=False)
        ids = templates.token_ids(prompt)
        if path.exists():
            record = json.loads(path.read_text())
            if record["completion"]["provenance"]["prompt_token_ids"] != ids:
                raise ValueError("assistant control encoding changed")
        else:
            result = complete(config["model"], prompt, ids,
                              {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 2048,
                               "stop_token_ids": templates.response_stops,
                               "stop": config["model"]["response_stop_strings"]})
            record = {"id": name, "question": question, "expected_substring": expected,
                      "expected_substring_present": expected in result["text"] if expected else None,
                      "completion": result}
            write_json(path, record)
        records.append(record)
    summary = {"cases": len(records), "nonempty": sum(bool(r["completion"]["text"].strip()) for r in records),
               "substring_checks": {r["id"]: r["expected_substring_present"] for r in records if r["expected_substring"]}}
    write_json(directory / "assistant-controls" / "summary.json", summary)
    return summary


def run(manifest: dict, out: Path, seeds_path: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another sweep already owns this output directory") from error
        _run_locked(manifest, out, seeds_path)


def _run_locked(manifest: dict, out: Path, seeds_path: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    specification = {"manifest": manifest, "seeds_sha256": digest(read_jsonl(seeds_path))}
    spec_path = out / "specification.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
        raise RuntimeError("sweep specification changed; use a new output directory")
    write_json(spec_path, specification)
    status_path = out / "status.json"
    status = {"state": "running", "started_at": now(), "pid": os.getpid(), "models": {}}
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        status["models"] = previous.get("models", {})
    write_json(status_path, status)
    api = HfApi()
    seeds = read_jsonl(seeds_path)
    base_url = f"http://127.0.0.1:{manifest['port']}/v1"
    process = None

    def interrupted(signum, frame):
        stop_server(process)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for entry in manifest["models"]:
        name = entry["name"]
        if status["models"].get(name, {}).get("state") == "complete":
            continue
        directory = out / name
        directory.mkdir(exist_ok=True)
        item = {"state": "preflight", "model": entry["model"], "lab": entry["lab"], "updated_at": now()}
        status["models"][name] = item

        def update(state: str, **fields):
            item.update(state=state, updated_at=now(), **fields)
            status["current_model"] = name
            write_json(status_path, status)
            print(f"{now()} {name}: {state}", flush=True)

        process = None
        try:
            update("preflight")
            revision_path = directory / "revision.json"
            if revision_path.exists():
                revision = json.loads(revision_path.read_text())["revision"]
            else:
                revision = api.model_info(entry["model"]).sha
                write_json(revision_path, {"model": entry["model"], "revision": revision})
            raw = {"model": {"model": entry["model"], "tokenizer": entry["model"],
                             "revision": revision, "tokenizer_revision": revision,
                             "base_url": base_url, "timeout_seconds": 600},
                   "generation": {"count": manifest["samples_per_condition"], "user_turns": 1,
                                  "attempts": 1, "user_max_tokens": manifest["user_max_tokens"],
                                  "top_p": manifest["top_p"], "seed": 42}}
            raw["model"].update(entry.get("model_options", {}))
            config = normalize_config(raw)
            templates = Templates(load_tokenizer(config["model"]), config["model"])
            prefixes = {"first_user": templates.user_prompt([]),
                        "followup_user": templates.user_prompt(plain(seeds[0]["messages"]))}
            write_json(directory / "prefixes.json", {"tokenizer": templates.identity, "prefixes": prefixes})
            # Download gated files before starting vLLM so access failures are
            # reported independently of sampled user-text quality.
            update("downloading", revision=revision)
            snapshot_download(entry["model"], revision=revision,
                              ignore_patterns=["original/*", "*.gguf", "*.bin", "*.pt", "*.pth", "*.md", "*.png", "*.jpg",
                                               *entry.get("ignore_patterns", [])])
            update("starting_server")
            argv = [str(Path(sys.executable).parent / "vllm"), "serve", entry["model"],
                    "--revision", revision, "--tokenizer-revision", revision,
                    "--tensor-parallel-size", "4", "--max-model-len", str(manifest.get("max_model_len", 8192)),
                    "--gpu-memory-utilization", str(manifest.get("gpu_memory_utilization", 0.85)), "--generation-config", "vllm",
                    "--host", "127.0.0.1", "--port", str(manifest["port"]),
                    *entry.get("server_args", [])]
            write_json(directory / "server-command.json", argv)
            server_env = os.environ.copy()
            if entry.get("server_pythonpath"):
                server_env["PYTHONPATH"] = entry["server_pythonpath"] + os.pathsep + server_env.get("PYTHONPATH", "")
                write_json(directory / "server-runtime-override.json", {"pythonpath": entry["server_pythonpath"]})
            with (directory / "server.log").open("a") as server_log:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=server_log,
                                           stderr=subprocess.STDOUT, start_new_session=True, env=server_env)
                update("starting_server", server_pid=process.pid)
                ready(process, base_url)
                if manifest.get("assistant_controls", False):
                    update("assistant_controls")
                    item["assistant_controls"] = assistant_controls(directory, config, templates)
                for regime, context in (("first", []), ("followup", seeds)):
                    for temperature in manifest["temperatures"]:
                        condition = f"{regime}-t{temperature:g}"
                        condition_out = directory / condition
                        raw["generation"]["temperature"] = temperature
                        cfg = normalize_config(raw)
                        plan = make_plan(cfg, templates, context)
                        save_plan(plan, condition_out)
                        update("sampling", condition=condition)
                        report = Runner(condition_out, plan, templates).run(manifest["jobs"])
                        item.setdefault("conditions", {})[condition] = report
                        write_json(status_path, status)
                update("complete")
        except Exception as error:
            # The full trace stays in a private host log; the public status only
            # identifies the failure class, never credentials or HTTP bodies.
            import traceback
            traceback.print_exc()
            update("setup_or_runtime_error", error_type=type(error).__name__)
        finally:
            stop_server(process)
            process = None
            time.sleep(3)
    failed = sum(x["state"] != "complete" for x in status["models"].values())
    status.update(state="complete_with_errors" if failed else "complete", completed_at=now(), current_model=None)
    write_json(status_path, status)
    print(f"{now()} sweep complete; {failed} model setup/runtime errors", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run a detached, sequential MAGPIE model sweep on its GPU host")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.manifest.read_text()), args.out, args.seeds)


if __name__ == "__main__":
    main()
