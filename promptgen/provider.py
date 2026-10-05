from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from heterogeneous.core import digest, text_hash
from heterogeneous.providers import ProviderError


def complete(model: dict, prompt: str, token_ids: list[int], sampling: dict) -> dict:
    body = {"model": model["model"], "prompt": token_ids, "stream": False,
            "echo": False, "add_special_tokens": False, "skip_special_tokens": False,
            "stop": model["stop_strings"], **sampling}
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(model["api_key_env"], "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(model["base_url"].rstrip("/") + "/completions",
                      data=json.dumps(body).encode(), headers=headers)
    try:
        with urlopen(request, timeout=model["timeout_seconds"]) as handle:
            response = json.load(handle)
    except HTTPError as error:
        raise ProviderError(f"completion endpoint HTTP {error.code}", retryable=error.code == 429 or error.code >= 500) from error
    except (URLError, TimeoutError, ConnectionError) as error:
        raise ProviderError("completion endpoint connection failed or timed out", retryable=True) from error
    except ValueError as error:
        raise ProviderError("completion endpoint returned invalid JSON") from error
    try:
        choice = response["choices"][0]
        text = choice["text"]
        if not isinstance(text, str):
            raise TypeError()
    except (KeyError, IndexError, TypeError) as error:
        raise ProviderError("completion endpoint returned no text completion") from error
    return {"text": text, "text_sha256": text_hash(text), "prompt": prompt, "provenance": {
        "type": "ai", "kind": "vllm-completion", "requested_model": model["model"],
        "reported_model": response.get("model"), "revision": model["revision"] or None,
        "request_id": response.get("id"), "usage": response.get("usage"),
        "finish_reason": choice.get("finish_reason"), "stop_reason": choice.get("stop_reason"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": text_hash(prompt), "prompt_token_ids": token_ids,
        "prompt_token_ids_sha256": digest(token_ids),
        "sampling": {k: v for k, v in body.items() if k not in ("prompt", "model")},
        "system_fingerprint": response.get("system_fingerprint")}}
