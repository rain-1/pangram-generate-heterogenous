import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from heterogeneous.core import digest, read_jsonl, text_hash
from heterogeneous.providers import ProviderError
from promptgen.config import normalize_config
from promptgen.pipeline import Runner, make_plan, save_plan, seed_context
from promptgen.provider import complete
from promptgen.templates import Templates, TokenPrompt
from promptgen.pipeline import quality_error


class FakeTokenizer:
    all_special_tokens = ["<start>", "<end>"]
    all_special_ids = [1, 2]
    eos_token_id = 2
    init_kwargs = {}

    def get_chat_template(self):
        return "test-template-v1"

    def get_vocab(self):
        return {"<start>": 1, "<end>": 2}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, continue_final_message):
        assert tokenize is False
        assert not (add_generation_prompt and continue_final_message)
        value = ""
        for i, message in enumerate(messages):
            value += f"<start>{message['role']}\n{message['content']}"
            if not (continue_final_message and i == len(messages) - 1):
                value += "<end>\n"
        if add_generation_prompt:
            value += "<start>assistant\n"
        return value

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


def config(**generation):
    return normalize_config({"model": {"model": "test-model"}, "generation": {"count": 1, **generation}})


def completion(text, prompt, tokens, sampling, reason="stop"):
    return {"text": text, "text_sha256": text_hash(text), "prompt": prompt,
            "provenance": {"type": "ai", "requested_model": "test-model", "reported_model": "actual-model",
                           "finish_reason": reason, "prompt_sha256": text_hash(prompt),
                           "prompt_token_ids": tokens, "sampling": sampling}}


class PromptGenerationTests(unittest.TestCase):
    def test_native_delimiters_missing_from_special_metadata_are_stopped(self):
        class IncompleteTokenizer(FakeTokenizer):
            def get_vocab(self):
                return {**super().get_vocab(), "<|im_start|>": 9, "<think>": 10}
            def get_added_vocab(self):
                return {"<|im_start|>": 9, "<think>": 10}
        cfg = normalize_config({"model": {"model": "test-model", "stop_strings": ["\nAssistant:"]}})
        t = Templates(IncompleteTokenizer(), cfg["model"])
        self.assertIn(9, t.user_stops)
        self.assertIn(10, t.user_stops)
        self.assertNotIn(10, t.response_stops)
        for text in ("Question<|im_start|>assistant", "Question\nAssistant: Answer"):
            self.assertEqual(quality_error(completion(text, "", [], {}), t), "control_token_in_text")

    def test_native_id_prompt_never_roundtrips_through_diagnostic_text(self):
        t = Templates(FakeTokenizer(), config()["model"])
        p = TokenPrompt("readable description", [400, 500])
        self.assertEqual(t.token_ids(p), [400, 500])
        self.assertNotEqual(t.token_ids(str(p)), [400, 500])

    def test_user_prefix_has_no_closing_token_or_assistant_header(self):
        t = Templates(FakeTokenizer(), config()["model"])
        self.assertEqual(t.render([], user=True), "<start>user\n")
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        self.assertEqual(t.render(history, user=True), "<start>user\nhello<end>\n<start>assistant\nhi<end>\n<start>user\n")
        self.assertTrue(t.render(history[:1], user=False).endswith("<start>assistant\n"))

    def test_empty_final_message_trimming_does_not_remove_header_newline(self):
        class TrimmingTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                rendered = super().apply_chat_template(messages, **kwargs)
                return rendered.rstrip() if kwargs['continue_final_message'] else rendered

        t = Templates(TrimmingTokenizer(), config()["model"])
        self.assertEqual(t.user_prompt([]), "<start>user\n")
        self.assertNotIn("MAGPIE_OPEN_USER_SENTINEL", t.user_prompt([]))

    def test_seed_must_be_a_completed_turn(self):
        for messages in ([{"role": "user", "content": "hi"}], [{"role": "assistant", "content": "hi"}],
                         [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]):
            with self.assertRaises(ValueError):
                seed_context({"id": "bad", "messages": messages})

    def test_config_rejects_instructed_generation_and_bad_fields(self):
        for raw in ({"model": {"model": "x"}, "axes": {"topic": ["flowers"]}},
                    {"model": {"model": "x"}, "generation": {"count": True}},
                    {"model": {"model": "x"}, "generation": {"temperature": float("nan")}},
                    {"model": {"model": "x"}, "generation": {"method": "instructed"}},
                    {"model": {"model": "x", "unexpected": 2}}):
            with self.assertRaises(ValueError):
                normalize_config(raw)

    def test_multiturn_raw_prefix_and_resume(self):
        cfg = config(user_turns=2)
        t = Templates(FakeTokenizer(), cfg["model"])
        plan = make_plan(cfg, t, [])
        responses = ["How should I start a balcony garden?", "Start with herbs and check sunlight.", "Which herbs tolerate afternoon shade?"]
        calls = []
        def fake(model, prompt, tokens, sampling):
            calls.append(prompt)
            return completion(responses[len(calls) - 1], prompt, tokens, sampling)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            save_plan(plan, out)
            with patch("promptgen.pipeline.complete", side_effect=fake):
                report = Runner(out, plan, t).run(2)
            self.assertEqual(report["user_prompts"], 2)
            self.assertTrue(calls[0].endswith("<start>user\n"))
            self.assertTrue(calls[1].endswith("<start>assistant\n"))
            self.assertIn(responses[1], calls[2])
            self.assertTrue(calls[2].endswith("<start>user\n"))
            row = read_jsonl(out / "conversations.jsonl")[0]
            self.assertEqual([m["role"] for m in row["messages"]], ["user", "assistant", "user"])
            self.assertTrue(all(m["author"]["type"] == "ai" for m in row["messages"]))
            with patch("promptgen.pipeline.complete", side_effect=AssertionError("must reuse cached steps")):
                self.assertEqual(Runner(out, plan, t).run(1), report)

    def test_followup_seed_authorship_is_preserved(self):
        cfg = config()
        t = Templates(FakeTokenizer(), cfg["model"])
        seeds = [{"id": "seed", "messages": [{"role": "user", "content": "Question", "author": {"type": "non_ai"}},
                                                 {"role": "assistant", "content": "Answer"}]}]
        plan = make_plan(cfg, t, seeds)
        with tempfile.TemporaryDirectory() as d:
            with patch("promptgen.pipeline.complete", side_effect=lambda m, p, ids, s: completion("Follow-up", p, ids, s)):
                Runner(Path(d), plan, t).run(1)
            row = read_jsonl(Path(d) / "conversations.jsonl")[0]
            self.assertEqual(row["seed_id"], "seed")
            self.assertEqual(row["messages"][0]["author"]["type"], "non_ai")
            self.assertEqual(row["messages"][1]["author"]["type"], "unknown")
            self.assertEqual(row["messages"][2]["author"]["type"], "ai")

    def test_interrupted_conversation_resumes_without_repeating_its_user_prompt(self):
        cfg = config(user_turns=2)
        t = Templates(FakeTokenizer(), cfg["model"])
        plan = make_plan(cfg, t, [])
        calls = []
        def interrupted(m, p, ids, s):
            calls.append(p)
            if len(calls) == 2:
                raise ProviderError("simulated server failure")
            return completion("First question?", p, ids, s)
        with tempfile.TemporaryDirectory() as d:
            with patch("promptgen.pipeline.complete", side_effect=interrupted):
                with self.assertRaises(ProviderError):
                    Runner(Path(d), plan, t).run(1)
            resumed_calls = []
            def resumed(m, p, ids, s):
                resumed_calls.append(p)
                return completion("Assistant reply." if len(resumed_calls) == 1 else "Next question?", p, ids, s)
            with patch("promptgen.pipeline.complete", side_effect=resumed):
                report = Runner(Path(d), plan, t).run(1)
            self.assertEqual(report["user_prompts"], 2)
            self.assertEqual(len(resumed_calls), 2)
            self.assertTrue(resumed_calls[0].endswith("<start>assistant\n"))

    def test_rejects_truncation_and_role_bleed_then_retries(self):
        cfg = config(attempts=3)
        t = Templates(FakeTokenizer(), cfg["model"])
        plan = make_plan(cfg, t, [])
        outputs = iter([("cut off", "length"), ("question<start>assistant\nanswer", "stop"), ("A clean prompt?", "stop")])
        def fake(m, p, ids, s):
            text, reason = next(outputs)
            return completion(text, p, ids, s, reason)
        with tempfile.TemporaryDirectory() as d:
            with patch("promptgen.pipeline.complete", side_effect=fake):
                report = Runner(Path(d), plan, t).run(1)
            self.assertEqual(report["accepted_conversations"], 1)
            state = json.loads((Path(d) / "cache/sample-00000000.json").read_text())
            self.assertEqual([a["rejection_reason"] for a in state["steps"][0]["attempts"]],
                             ["truncated_or_unfinished", "control_token_in_text", None])

    def test_exhaustion_is_recorded_and_not_retried_on_resume(self):
        cfg = config(attempts=1)
        t = Templates(FakeTokenizer(), cfg["model"])
        plan = make_plan(cfg, t, [])
        with tempfile.TemporaryDirectory() as d:
            with patch("promptgen.pipeline.complete", side_effect=lambda m, p, ids, s: completion("", p, ids, s)):
                Runner(Path(d), plan, t).run(1)
            with patch("promptgen.pipeline.complete", side_effect=AssertionError("no repeated exhausted attempts")):
                report = Runner(Path(d), plan, t).run(1)
            self.assertEqual(report["accepted_conversations"], 0)
            self.assertEqual(read_jsonl(Path(d) / "rejected.jsonl")[0]["reason"], "attempts_exhausted")

    def test_duplicate_first_prompts_removed_in_stable_order(self):
        cfg = config(count=2)
        t = Templates(FakeTokenizer(), cfg["model"])
        plan = make_plan(cfg, t, [])
        with tempfile.TemporaryDirectory() as d:
            with patch("promptgen.pipeline.complete", side_effect=lambda m, p, ids, s: completion("Same prompt", p, ids, s)):
                report = Runner(Path(d), plan, t).run(2)
            self.assertEqual(report["accepted_conversations"], 1)
            self.assertEqual(read_jsonl(Path(d) / "rejected.jsonl")[0]["status"], "duplicate")

    def test_user_synthesis_prefix_contains_only_the_conversation_and_open_user_header(self):
        cfg = config()
        t = Templates(FakeTokenizer(), cfg["model"])
        plan = make_plan(cfg, t, [])
        p = t.user_prompt(plan["samples"][0]["messages"])
        self.assertEqual(p, "<start>user\n")
        self.assertNotIn("Write one plausible user message", p)

    def test_cache_content_tampering_is_rejected(self):
        cfg = config()
        t = Templates(FakeTokenizer(), cfg["model"])
        plan = make_plan(cfg, t, [])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache/sample-00000000.json"
            with patch("promptgen.pipeline.complete", side_effect=lambda m, p, ids, s: completion("Original prompt", p, ids, s)):
                Runner(Path(d), plan, t).run(1)
            state = json.loads(path.read_text())
            state["steps"][0]["accepted"]["text"] = "Changed prompt"
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "integrity"):
                Runner(Path(d), plan, t).run(1)

    def test_raw_completion_http_uses_token_ids_and_preserves_special_tokens(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"id": "request", "model": "actual", "choices": [{"text": "Generated question?", "finish_reason": "stop", "stop_reason": 2}]}).encode())
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = config()["model"]
            model["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
            result = complete(model, "prefix", [1, 5], {"stop_token_ids": [2], "max_tokens": 10})
            self.assertEqual(captured[0][0], "/v1/completions")
            body = captured[0][1]
            self.assertEqual(body["prompt"], [1, 5])
            self.assertFalse(body["add_special_tokens"])
            self.assertFalse(body["skip_special_tokens"])
            self.assertNotIn("messages", body)
            self.assertEqual(result["provenance"]["reported_model"], "actual")
            self.assertEqual(result["provenance"]["prompt_token_ids_sha256"], digest([1, 5]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
