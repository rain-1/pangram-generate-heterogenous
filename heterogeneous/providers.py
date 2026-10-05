from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core import text_hash


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def run_cli(argv: list[str], prompt: str, cwd: str, timeout: int, env: dict) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(argv, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=timeout, env=env)
    except FileNotFoundError as error:
        raise ProviderError(f"{argv[0]} is not installed or not on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise ProviderError(f"{argv[0]} timed out; the request may have consumed quota", retryable=True) from error
    if result.returncode:
        # CLI error streams may contain private text; don't persist them in shared data.
        raise ProviderError(f"{argv[0]} exited {result.returncode}; check login, model access and quota")
    return result


def complete(config: dict, prompt: str) -> dict:
    kind = config["kind"]
    usage, actual_model, request_id, provider, finish_reason = None, None, None, None, None
    metadata = {}
    if kind == "claude-web":
        raise ProviderError("claude-web requires an explicit browser completion import; no HTTP or CLI fallback is allowed")
    if kind == "codex":
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            env.pop(key, None)
        status = run_cli(["codex", "login", "status"], "", os.getcwd(), 30, env)
        if "chatgpt" not in (status.stdout + status.stderr).lower():
            raise ProviderError("Codex subscription backend requires ChatGPT login; run codex login")
        with tempfile.TemporaryDirectory(prefix="heterogeneous-codex-") as directory:
            output = Path(directory) / "response.txt"
            argv = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--model", config["model"], "--output-last-message", str(output)]
            argv.extend(config.get("cli_args", []))
            argv.append("-")
            result = run_cli(argv, prompt, directory, config["timeout_seconds"], env)
            if not output.exists():
                raise ProviderError("Codex did not write a final response")
            text = output.read_text(encoding="utf-8")
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") in ("error", "turn.failed"):
                    raise ProviderError("Codex reported a failed turn")
                if event.get("type") == "thread.started":
                    request_id = event.get("thread_id")
                    actual_model = event.get("model")
                if event.get("type") == "turn.completed":
                    usage = event.get("usage")
                    actual_model = event.get("model", actual_model)
            metadata["model_identity_status"] = "reported" if actual_model else "requested_only"
            finish_reason = "completed"
    elif kind == "claude":
        # --bare ignores subscription OAuth. Use print mode with tools/MCP disabled instead.
        env = os.environ.copy()
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
            env.pop(key, None)
        status = run_cli(["claude", "auth", "status", "--json"], "", os.getcwd(), 30, env)
        try:
            auth = json.loads(status.stdout)
        except ValueError as error:
            raise ProviderError("Claude authentication status returned invalid JSON") from error
        if not auth.get("loggedIn") or auth.get("authMethod") != "claude.ai":
            raise ProviderError("Claude subscription backend requires claude.ai login; run claude auth login")
        with tempfile.TemporaryDirectory(prefix="heterogeneous-claude-") as directory:
            argv = ["claude", "--print", "--output-format", "json", "--model", config["model"], "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "", "--no-session-persistence", "--disable-slash-commands", "--system-prompt", "You produce prose for a dataset. Follow the user's requested output format. Do not use tools."]
            argv.extend(config.get("cli_args", []))
            result = run_cli(argv, prompt, directory, config["timeout_seconds"], env)
            try:
                response = json.loads(result.stdout)
            except ValueError as error:
                raise ProviderError("Claude returned invalid JSON") from error
            if response.get("is_error") or response.get("subtype", "success") != "success":
                raise ProviderError("Claude reported an unsuccessful result; check quota/model access")
            text = response.get("result")
            model_usage = response.get("modelUsage", {})
            actual_model = next(iter(model_usage)) if len(model_usage) == 1 else None
            usage, request_id = response.get("usage"), response.get("session_id")
            metadata = {"model_usage": model_usage, "model_identity_status": "reported" if actual_model else "ambiguous_or_unreported"}
            finish_reason = "completed"
    else:
        key = os.environ.get(config["api_key_env"], "")
        if kind == "openrouter" and not key:
            raise ProviderError(f"set {config['api_key_env']} to your OpenRouter API key")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if kind == "openrouter":
            headers["X-OpenRouter-Title"] = "generate-heterogeneous"
            headers["X-OpenRouter-Metadata"] = "enabled"
            headers["X-OpenRouter-Cache"] = "false"
        body = {
            "model": config["model"], "messages": [{"role": "user", "content": prompt}],
            "temperature": config["temperature"], "max_tokens": config["max_tokens"], "stream": False,
            **config.get("extra_body", {}),
        }
        request = Request(config["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
        try:
            with urlopen(request, timeout=config["timeout_seconds"]) as response_handle:
                response = json.load(response_handle)
        except HTTPError as error:
            raise ProviderError(f"{kind} HTTP {error.code}", retryable=error.code == 429 or error.code >= 500) from error
        except (URLError, TimeoutError, ConnectionError) as error:
            raise ProviderError(f"{kind} connection failed or timed out", retryable=True) from error
        except ValueError as error:
            raise ProviderError(f"{kind} returned invalid JSON") from error
        try:
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"{kind} returned no completion") from error
        if finish_reason not in ("stop", "eos"):
            raise ProviderError(f"{kind} completion was truncated or blocked: {finish_reason}")
        actual_model, usage, request_id = response.get("model"), response.get("usage"), response.get("id")
        metadata = {k: response[k] for k in ("system_fingerprint", "openrouter_metadata", "service_tier") if k in response}
        provider = response.get("provider")
    if not isinstance(text, str) or not text.strip():
        raise ProviderError(f"{kind} returned empty/non-text output")
    return {
        "text": text.strip(),
        "provenance": {
            "backend": config["name"], "kind": kind, "requested_model": config["model"],
            "reported_model": actual_model, "family": config.get("family"), "revision": config.get("revision"),
            "provider": provider, "request_id": request_id,
            "sampling": {k: config[k] for k in ("temperature", "max_tokens", "extra_body", "cli_args") if k in config},
            "generated_at": datetime.now(timezone.utc).isoformat(), "finish_reason": finish_reason,
            "prompt_sha256": text_hash(prompt), "usage": usage, "metadata": metadata,
        },
    }
