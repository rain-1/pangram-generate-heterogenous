from __future__ import annotations

import math
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

DEFAULTS = {
    "model": {"model": "", "tokenizer": "", "revision": "", "tokenizer_revision": "",
              "base_url": "http://127.0.0.1:8000/v1", "api_key_env": "VLLM_API_KEY",
              "timeout_seconds": 600, "stop_token_ids": [], "stop_strings": [],
              "response_stop_strings": [], "encoder": "jinja", "trust_remote_code": False},
    "generation": {"method": "magpie", "count": 100, "user_turns": 1,
                   "emit_final_response": False, "seed": 42, "attempts": 3,
                   "system_prompt": "", "user_max_tokens": 1024,
                   "response_max_tokens": 2048, "temperature": 1.0,
                   "response_temperature": 0.8, "top_p": 0.95},
}


def normalize_config(raw: dict) -> dict:
    if set(raw) - set(DEFAULTS):
        raise ValueError(f"unknown config tables: {sorted(set(raw) - set(DEFAULTS))}")
    result = {}
    for section, defaults in DEFAULTS.items():
        given = raw.get(section, {})
        if not isinstance(given, dict):
            raise ValueError(f"{section} must be a table")
        if set(given) - set(defaults):
            raise ValueError(f"unknown {section} fields: {sorted(set(given) - set(defaults))}")
        result[section] = {**defaults, **given}
    model, generation = result["model"], result["generation"]
    for key in ("model", "tokenizer", "revision", "tokenizer_revision", "base_url", "api_key_env"):
        if not isinstance(model[key], str):
            raise ValueError(f"model.{key} must be a string")
    if not model["model"]:
        raise ValueError("model.model is required")
    model["tokenizer"] = model["tokenizer"] or model["model"]
    if model["encoder"] not in ("jinja", "deepseek_v4", "inkling"):
        raise ValueError("encoder must be jinja, deepseek_v4 or inkling")
    if type(model["trust_remote_code"]) is not bool:
        raise ValueError("trust_remote_code must be boolean")
    url = urlsplit(model["base_url"])
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
    for section, fields in ((model, ("timeout_seconds",)), (generation, ("count", "user_turns", "attempts", "user_max_tokens", "response_max_tokens"))):
        for key in fields:
            if type(section[key]) is not int or section[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    if type(generation["seed"]) is not int or not 0 <= generation["seed"] < 2**63:
        raise ValueError("seed must be a nonnegative signed 64-bit integer")
    if type(generation["emit_final_response"]) is not bool:
        raise ValueError("emit_final_response must be boolean")
    if generation["method"] != "magpie":
        raise ValueError("method must be magpie: inference continues inside the user message")
    if not isinstance(generation["system_prompt"], str):
        raise ValueError("system_prompt must be a string")
    for key in ("temperature", "response_temperature", "top_p"):
        val = generation[key]
        if type(val) not in (int, float) or not math.isfinite(val) or val < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < generation["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not isinstance(model["stop_token_ids"], list) or any(type(i) is not int or i < 0 for i in model["stop_token_ids"]):
        raise ValueError("stop_token_ids must be a list of nonnegative integers")
    for key in ("stop_strings", "response_stop_strings"):
        if not isinstance(model[key], list) or any(not isinstance(s, str) or not s for s in model[key]):
            raise ValueError(f"{key} must be a list of nonempty strings")
    return result


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        return normalize_config(tomllib.load(handle))
