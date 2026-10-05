from __future__ import annotations

import copy
import math
import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_DATASET = {
    "seed": 42, "variants_per_source": 2, "include_human_controls": True,
    "unit": "paragraph", "mode": "random", "block_sizes": [3, 4, 6],
    "min_blocks": 1, "max_blocks": 3,
    "max_replaced_fraction": 0.6, "summary_min_sentences": 2, "summary_max_sentences": 3,
    "context_chars": 0,
    "copy_ngram": 8, "max_copy_fraction": 0.15,
    "train_fraction": 0.8, "validation_fraction": 0.1, "test_fraction": 0.1,
    "quality_attempts": 3,
    "allow_human_candidates": False,
    "require_prose_paragraphs": False,
}
DEFAULT_BACKEND = {"concurrency": 1, "timeout_seconds": 300, "retries": 2}
BACKEND_KEYS = {
    "name", "kind", "model", "family", "revision", "concurrency", "timeout_seconds", "retries",
    "base_url", "api_key_env", "temperature", "max_tokens", "extra_body", "cli_args",
}


def validate_cli_args(kind: str, args: list[str]) -> None:
    """Extra flags may adjust inference, never override model/output/auth isolation."""
    cursor = 0
    while cursor < len(args):
        flag = args[cursor]
        if cursor + 1 >= len(args):
            raise ValueError("cli_args flags must have explicit values")
        value = args[cursor + 1]
        if kind == "codex":
            if flag not in ("-c", "--config") or "=" not in value or value.split("=", 1)[0] not in {"model_reasoning_effort", "model_reasoning_summary", "model_verbosity"}:
                raise ValueError("Codex cli_args allow only -c/--config for reasoning effort, summary and verbosity")
        elif flag != "--effort" or value not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Claude cli_args allow only --effort with a supported effort value")
        cursor += 2


def positive_int(value, name: str, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")


def number(value, name: str, low: float, high: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def backend_config(raw: dict) -> dict:
    if set(raw) - BACKEND_KEYS:
        raise ValueError(f"unknown backend settings: {sorted(set(raw) - BACKEND_KEYS)}")
    result = {**DEFAULT_BACKEND, **copy.deepcopy(raw)}
    for key in ("name", "model", "kind"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise ValueError(f"backend {key} is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result["name"]):
        raise ValueError("backend name may contain only letters, digits, _ and -")
    if result["kind"] not in ("codex", "claude", "claude-web", "openrouter", "openai-compatible"):
        raise ValueError(f"unsupported backend kind: {result['kind']}")
    for field in ("family", "revision"):
        if field in result and (not isinstance(result[field], str) or not result[field]):
            raise ValueError(f"backend {field} must be a nonempty string")
    positive_int(result["concurrency"], "concurrency")
    positive_int(result["timeout_seconds"], "timeout_seconds")
    positive_int(result["retries"], "retries", zero=True)
    if "temperature" in result:
        number(result["temperature"], "temperature", 0, 2)
    if "max_tokens" in result:
        positive_int(result["max_tokens"], "max_tokens")
    if result["kind"] == "claude-web":
        if set(result) & {"base_url", "api_key_env", "temperature", "max_tokens", "extra_body", "cli_args"}:
            raise ValueError("Browser backends record UI output; HTTP/CLI sampling parameters are unavailable")
    elif result["kind"] in ("codex", "claude"):
        if set(result) & {"base_url", "api_key_env", "temperature", "max_tokens", "extra_body"}:
            raise ValueError("CLI backends don't expose HTTP sampling parameters; use cli_args")
        if not isinstance(result.get("cli_args", []), list) or not all(isinstance(x, str) for x in result.get("cli_args", [])):
            raise ValueError("cli_args must be a list of strings")
        validate_cli_args(result["kind"], result.get("cli_args", []))
    else:
        if "cli_args" in result:
            raise ValueError("cli_args only applies to CLI backends")
        result.setdefault("base_url", "https://openrouter.ai/api/v1" if result["kind"] == "openrouter" else "http://127.0.0.1:8000/v1")
        result.setdefault("api_key_env", "OPENROUTER_API_KEY" if result["kind"] == "openrouter" else "VLLM_API_KEY")
        result.setdefault("temperature", 0.8)
        result.setdefault("max_tokens", 4096)
        if not isinstance(result["api_key_env"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result["api_key_env"]):
            raise ValueError("api_key_env must be an environment variable name")
        if not isinstance(result["base_url"], str):
            raise ValueError("base_url must be a string")
        url = urlsplit(result["base_url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
        if not isinstance(result.get("extra_body", {}), dict):
            raise ValueError("extra_body must be a table")
        if set(result.get("extra_body", {})) & {"model", "models", "messages", "stream", "n", "temperature", "max_tokens"}:
            raise ValueError("extra_body cannot override model/messages/sampling or enable model fallbacks")
    return result


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if set(raw) - {"dataset", "summarizer", "generators"}:
        raise ValueError("config must contain only dataset, summarizer and generators")
    settings = raw.get("dataset", {})
    if set(settings) - set(DEFAULT_DATASET):
        raise ValueError(f"unknown dataset settings: {sorted(set(settings) - set(DEFAULT_DATASET))}")
    dataset = {**copy.deepcopy(DEFAULT_DATASET), **settings}
    for key in ("variants_per_source", "min_blocks", "max_blocks", "summary_min_sentences", "summary_max_sentences", "copy_ngram", "quality_attempts"):
        positive_int(dataset[key], key)
    positive_int(dataset["seed"], "seed", zero=True)
    positive_int(dataset["context_chars"], "context_chars", zero=True)
    if type(dataset["include_human_controls"]) is not bool:
        raise ValueError("include_human_controls must be boolean")
    if type(dataset["allow_human_candidates"]) is not bool:
        raise ValueError("allow_human_candidates must be boolean")
    if type(dataset["require_prose_paragraphs"]) is not bool:
        raise ValueError("require_prose_paragraphs must be boolean")
    if dataset["unit"] not in ("sentence", "paragraph") or dataset["mode"] not in ("random", "alternating"):
        raise ValueError("unit must be sentence/paragraph and mode random/alternating")
    sizes = dataset["block_sizes"]
    if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError("block_sizes must be a nonempty list of positive integers")
    if len(set(sizes)) != len(sizes):
        raise ValueError("block_sizes must not contain duplicates")
    if dataset["min_blocks"] > dataset["max_blocks"] or dataset["summary_min_sentences"] > dataset["summary_max_sentences"]:
        raise ValueError("minimum cannot exceed maximum")
    number(dataset["max_replaced_fraction"], "max_replaced_fraction", 0.001, 0.999)
    number(dataset["max_copy_fraction"], "max_copy_fraction", 0, 1)
    fractions = [dataset[f"{split}_fraction"] for split in ("train", "validation", "test")]
    for fraction in fractions:
        number(fraction, "split fraction", 0, 1)
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("split fractions must sum to one")
    summarizer = backend_config(raw["summarizer"])
    generators = [backend_config(g) for g in raw["generators"]]
    names = [g["name"] for g in generators]
    if not names or len(names) != len(set(names)):
        raise ValueError("generators must have unique names and cannot be empty")
    return {"dataset": dataset, "summarizer": summarizer, "generators": generators}
