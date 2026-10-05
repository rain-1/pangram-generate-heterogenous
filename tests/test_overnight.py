"""Check bounded quota failures and persistent between-batch scheduling."""

import json
import threading

import pytest

from scripts.run_span_expansion import bounded_work
from scripts.run_overnight import remaining_pause, publish_status, check_gate


def test_provider_halt_drains_inflight_without_submitting_entire_campaign():
    release = threading.Event()
    started = []

    def work(index):
        started.append(index)
        if index:
            assert release.wait(3)
        return index

    finished = []

    def completed(index, future):
        finished.append(future.result())
        release.set()
        return index == 0

    assert bounded_work(range(1000), work, 3, completed) is True
    assert set(started) == set(finished) == {0, 1, 2}


def test_successful_bounded_work_runs_all_variants():
    finished = []

    def completed(index, future):
        finished.append(future.result())
        return False

    assert bounded_work(range(23), lambda index: index, 3, completed) is False
    assert sorted(finished) == list(range(23))


def test_restart_only_waits_remainder_of_pause():
    assert remaining_pause(1000, 1800, 2000) == 800
    assert remaining_pause(1000, 1800, 3000) == 0


def test_status_is_safe_to_open_and_retains_checkpoint(tmp_path):
    state = {
        "phase": "sleeping",
        "resume_at": 2800,
        "message": "<script>unsafe</script>",
        "batches": [
            {
                "name": "example",
                "mixed": 15,
                "phase": "retry_cooldown",
                "progress": {"completed_tasks": 28, "total": 1000},
            }
        ],
    }
    publish_status(tmp_path, state)
    assert json.loads((tmp_path / "status.json").read_text())["resume_at"] == 2800
    html = (tmp_path / "status.html").read_text()
    assert "<script>unsafe</script>" not in html and "&lt;script&gt;" in html
    assert "28/1000" in html


def test_probe_gate_refuses_approval_for_different_plan(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.run_overnight.load_plan", lambda run: {"id": "frozen"})
    (tmp_path / "model-probe-audit.json").write_text(
        json.dumps({"plan_id": "different", "decision": "proceed"})
    )
    with pytest.raises(ValueError, match="matching approval"):
        check_gate({"directory": str(tmp_path), "plan_id": "frozen"})


def test_supervisor_restart_preserves_later_retry_deadline(tmp_path, monkeypatch):
    from scripts import run_overnight as module

    clock = [2000.0]
    sleep_states = []
    batches = [
        {"name": f"batch-{n}", "directory": str(tmp_path / str(n)), "plan_id": str(n)}
        for n in range(2)
    ]
    for batch in batches:
        directory = module.Path(batch["directory"])
        directory.mkdir()
        (directory / "detector-quality-audit.json").write_text(
            json.dumps(
                {
                    "mixed_documents": 1000,
                    "models": {"haiku": 1000},
                    "prose_review_flags": [],
                }
            )
        )
        (directory / "parent-copy-audit.json").write_text(
            json.dumps({"flagged_blocks": []})
        )
    (tmp_path / "campaign.json").write_text(
        json.dumps({"batches": batches, "pause_seconds": 1800})
    )
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "phase": "sleeping",
                "resume_at": 3000,
                "batches": [
                    {
                        "name": "batch-0",
                        "plan_id": "0",
                        "mixed": 1000,
                        "phase": "complete",
                        "finished_at": 1000,
                        "attempts": 1,
                    },
                    {
                        "name": "batch-1",
                        "plan_id": "1",
                        "mixed": 15,
                        "phase": "retry_cooldown",
                        "attempts": 1,
                    },
                ],
            }
        )
    )

    def sleep(seconds):
        sleep_states.append(
            json.loads((tmp_path / "status.json").read_text())["resume_at"]
        )
        clock[0] += seconds

    class Child:
        stdout = iter(['{"mixed": 1000}\n'])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    monkeypatch.setattr(module, "check_gate", lambda batch: None)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "combine",
        lambda *args: (
            {"mixed_documents": 2000, "prose_review_flags": []},
            {"mixed_documents": 2000},
        ),
    )
    module.run_campaign(tmp_path)
    assert clock[0] == 3000
    assert set(sleep_states) == {3000}
    final = json.loads((tmp_path / "status.json").read_text())
    assert final["phase"] == "complete"
    assert final["batches"][1]["attempts"] == 2
