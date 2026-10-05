import json
import os
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from heterogeneous.config import backend_config
from heterogeneous.providers import ProviderError, complete


class ProviderTests(unittest.TestCase):
    def test_codex_final_message_and_unreported_model(self):
        calls = []
        def fake(argv, prompt, cwd, timeout, env):
            calls.append(argv)
            self.assertNotIn("OPENAI_API_KEY", env)
            if argv[1:3] == ["login", "status"]:
                return subprocess.CompletedProcess(argv, 0, "", "Logged in using ChatGPT")
            path = Path(argv[argv.index("--output-last-message") + 1])
            path.write_text("Original generated prose.")
            return subprocess.CompletedProcess(argv, 0, '\n'.join([
                '{"type":"thread.started","thread_id":"thread-123"}',
                '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":4}}']), "")
        with patch("heterogeneous.providers.run_cli", side_effect=fake), patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
            result = complete(backend_config({"name": "codex", "kind": "codex", "model": "test-model"}), "prompt")
        self.assertEqual(result["text"], "Original generated prose.")
        self.assertIsNone(result["provenance"]["reported_model"])
        self.assertEqual(result["provenance"]["requested_model"], "test-model")
        self.assertEqual(result["provenance"]["request_id"], "thread-123")
        self.assertIn("--ignore-user-config", calls[1])
        self.assertIn("read-only", calls[1])

    def test_codex_api_login_is_not_subscription(self):
        status = subprocess.CompletedProcess([], 0, "Logged in using an API key", "")
        with patch("heterogeneous.providers.run_cli", return_value=status):
            with self.assertRaisesRegex(ProviderError, "requires ChatGPT"):
                complete(backend_config({"name": "codex", "kind": "codex", "model": "test"}), "prompt")

    def test_claude_single_and_ambiguous_model_usage(self):
        for model_usage, expected in (({"actual-claude": {"inputTokens": 10}}, "actual-claude"), ({"model-a": {}, "model-b": {}}, None)):
            def fake(argv, prompt, cwd, timeout, env):
                self.assertNotIn("ANTHROPIC_API_KEY", env)
                self.assertNotIn("--bare", argv)
                if argv[1:3] == ["auth", "status"]:
                    return subprocess.CompletedProcess(argv, 0, json.dumps({"loggedIn": True, "authMethod": "claude.ai"}), "")
                self.assertEqual(argv[argv.index("--tools") + 1], "")
                return subprocess.CompletedProcess(argv, 0, json.dumps({"result": "New prose.", "is_error": False, "subtype": "success", "modelUsage": model_usage, "usage": {"input_tokens": 10}}), "")
            with patch("heterogeneous.providers.run_cli", side_effect=fake), patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy"}):
                result = complete(backend_config({"name": "claude", "kind": "claude", "model": "alias"}), "prompt")
            self.assertEqual(result["provenance"]["reported_model"], expected)

    def test_claude_auth_error_rejected(self):
        status = subprocess.CompletedProcess([], 0, json.dumps({"loggedIn": True, "authMethod": "api_key"}), "")
        with patch("heterogeneous.providers.run_cli", return_value=status):
            with self.assertRaisesRegex(ProviderError, "requires claude.ai"):
                complete(backend_config({"name": "claude", "kind": "claude", "model": "test"}), "prompt")

    def test_openai_compatible_and_openrouter_http(self):
        captured = []
        response = {"id": "http-123", "model": "served-actual-model", "choices": [{"finish_reason": "stop", "message": {"content": " New generated prose. "}}], "usage": {"prompt_tokens": 10}, "openrouter_metadata": {"provider": "test-provider"}}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))), dict(self.headers)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for kind in ("openai-compatible", "openrouter"):
                backend = backend_config({"name": "http", "kind": kind, "model": "requested-model", "api_key_env": "TEST_HETEROGENEOUS_KEY", "base_url": f"http://127.0.0.1:{server.server_port}/v1"})
                with patch.dict(os.environ, {"TEST_HETEROGENEOUS_KEY": "dummy-not-a-real-key"}):
                    result = complete(backend, "Generate prose")
                self.assertEqual(result["text"], "New generated prose.")
                self.assertEqual(result["provenance"]["reported_model"], "served-actual-model")
                path, body, headers = captured[-1]
                self.assertEqual(path, "/v1/chat/completions")
                self.assertEqual(body["model"], "requested-model")
                self.assertEqual(body["messages"][0]["content"], "Generate prose")
                self.assertEqual(headers["Authorization"], "Bearer dummy-not-a-real-key")
            response["choices"][0]["finish_reason"] = "length"
            with patch.dict(os.environ, {"TEST_HETEROGENEOUS_KEY": "dummy"}):
                with self.assertRaisesRegex(ProviderError, "truncated"):
                    complete(backend, "Generate prose")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_openrouter_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderError, "API key"):
                complete(backend_config({"name": "router", "kind": "openrouter", "model": "a/b"}), "prompt")


if __name__ == "__main__":
    unittest.main()
