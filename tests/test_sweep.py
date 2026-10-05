import json
import signal
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from promptgen.sweep import run, stop_server, assistant_controls
from test_promptgen import FakeTokenizer
from test_promptgen import config, completion
from promptgen.templates import Templates


class SweepTests(unittest.TestCase):
    def test_assistant_controls_use_assistant_headers_and_do_not_stop_at_user_controls(self):
        cfg = config()
        cfg["model"]["stop_strings"] = ["\nAssistant:"]
        templates = Templates(FakeTokenizer(), cfg["model"])
        calls = []
        def fake(model, prompt, ids, sampling):
            calls.append((prompt, sampling))
            return completion("READY 323 Rain falls.", prompt, ids, sampling)
        with tempfile.TemporaryDirectory() as directory, patch("promptgen.sweep.complete", side_effect=fake):
            summary = assistant_controls(Path(directory), cfg, templates)
            self.assertEqual(summary["nonempty"], 3)
            self.assertTrue(all(p.endswith("<start>assistant\n") and s["stop"] == [] for p, s in calls))
            assistant_controls(Path(directory), cfg, templates)
            self.assertEqual(len(calls), 3)

    def test_server_cleanup_only_targets_its_own_process_group(self):
        process = MagicMock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("server", 60), 0]
        with patch("promptgen.sweep.os.killpg") as kill:
            stop_server(process)
        self.assertEqual(kill.call_args_list[0].args, (1234, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (1234, signal.SIGKILL))
        process.poll.return_value = 0
        process.wait.side_effect = None
        with patch("promptgen.sweep.os.killpg") as kill:
            stop_server(process)
            self.assertEqual(kill.call_args_list[-1].args, (1234, signal.SIGKILL))

        with patch("promptgen.sweep.os.killpg", side_effect=ProcessLookupError):
            stop_server(process)

    def test_access_failure_does_not_abort_other_models_and_successes_resume(self):
        hub = types.ModuleType("huggingface_hub")
        api = MagicMock()
        def info(model):
            if model == "gated":
                raise PermissionError("access unavailable")
            return types.SimpleNamespace(sha="frozen-revision")
        api.model_info.side_effect = info
        hub.HfApi = lambda: api
        hub.snapshot_download = MagicMock()
        manifest = {"samples_per_condition": 1, "temperatures": [0.7, 1.0], "top_p": 0.95,
                    "user_max_tokens": 20, "jobs": 1, "port": 18080,
                    "max_model_len": 4096, "gpu_memory_utilization": 0.9,
                    "models": [{"name": "blocked", "model": "gated", "lab": "one"},
                               {"name": "working", "model": "available", "lab": "two",
                                "server_pythonpath": "/task/compat", "ignore_patterns": ["consolidated*"]}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = root / "seeds.jsonl"
            seeds.write_text(json.dumps({"id": "seed", "messages": [
                {"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}) + "\n")
            runner = MagicMock()
            runner.run.return_value = {"accepted_conversations": 1, "user_prompts": 1}
            with patch.dict("sys.modules", {"huggingface_hub": hub}), \
                 patch("promptgen.sweep.load_tokenizer", return_value=FakeTokenizer()), \
                 patch("promptgen.sweep.signal.signal"), patch("promptgen.sweep.time.sleep"), \
                 patch("promptgen.sweep.subprocess.Popen", return_value=MagicMock(pid=999)) as launch, \
                 patch("promptgen.sweep.stop_server"), patch("promptgen.sweep.ready"), \
                 patch("promptgen.sweep.Runner", return_value=runner) as factory, \
                 patch("traceback.print_exc"):
                run(manifest, root / "out", seeds)
                status = json.loads((root / "out/status.json").read_text())
                self.assertEqual(status["state"], "complete_with_errors")
                self.assertEqual(status["models"]["blocked"]["state"], "setup_or_runtime_error")
                self.assertEqual(status["models"]["working"]["state"], "complete")
                self.assertEqual(launch.call_count, 1)
                argv = launch.call_args.args[0]
                self.assertEqual(argv[argv.index("--max-model-len") + 1], "4096")
                self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.9")
                self.assertTrue(launch.call_args.kwargs["env"]["PYTHONPATH"].startswith("/task/compat:"))
                self.assertIn("consolidated*", hub.snapshot_download.call_args.kwargs["ignore_patterns"])
                self.assertEqual(runner.run.call_count, 4)
                self.assertTrue(all(call.args[1]["config"]["generation"]["attempts"] == 1 for call in factory.call_args_list))
                launch.reset_mock()
                runner.run.reset_mock()
                run(manifest, root / "out", seeds)
                launch.assert_not_called()
                runner.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
