"""Resume frozen subscription-backed batches, with persistent between-batch pauses."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from html import escape
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from heterogeneous.core import atomic_text, read_jsonl, write_json, write_jsonl
from heterogeneous.pipeline import load_plan, validate_dataset


ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def remaining_pause(finished_at, pause_seconds, now):
    return max(0, finished_at + pause_seconds - now)


def publish_status(directory, state):
    state["updated_at"] = timestamp()
    write_json(directory / "status.json", state)
    rows = []
    for batch in state["batches"]:
        n = batch.get("mixed", 0)
        progress = batch.get("progress", {})
        calls = (
            str(progress.get("completed_tasks", 0))
            + "/"
            + str(progress.get("total", 1000))
        )
        rows.append(
            f"<tr><td>{escape(batch['name'])}</td><td>{n}/1000</td>"
            f"<td>{escape(calls)}</td><td>{escape(batch.get('phase', 'pending'))}</td></tr>"
        )
    resume = state.get("resume_at")
    resume_text = (
        datetime.fromtimestamp(resume, timezone.utc).isoformat() if resume else "—"
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30">
<title>Overnight span generation</title><style>
body{{font:17px/1.6 system-ui;background:#f6f7f2;color:#20332b;max-width:950px;margin:40px auto;padding:24px}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #dbe3dc}}
a{{color:#17623a}}small{{color:#627168}}</style>
<h1>Overnight span generation</h1><p><strong>{escape(state["phase"])}</strong> ·
{sum(b.get("mixed", 0) for b in state["batches"])}/4000 mixed documents assembled.</p>
<p>800 each: Haiku, Sonnet, current Opus, GPT-6.1 Sol and GPT-6 Luna. 1600 ML / 2400 general.
Matched unchanged controls are stored separately. No new Opus 3.</p>
<table><tr><th>Batch</th><th>Assembled</th><th>Tasks finished this attempt</th><th>State</th></tr>{"".join(rows)}</table>
<p>Next scheduled resume (UTC): {resume_text}. Updated: {escape(state["updated_at"])}.</p>
<p>{escape(state.get("message", ""))}</p><p><a href="status.json">Detailed status</a> ·
<a href="supervisor.log">Supervisor log</a> · <a href="campaign.json">Frozen campaign</a></p>
<p>{" · ".join(f'<a href="batch-{i:02d}/review.html">Batch {i} colored review</a>' for i in range(1, 5))}</p>
<p><small>This runs on this computer and requires it to stay awake. Successful model calls survive restarts.
Extraction candidates retain documented provenance rather than certified human authorship.</small></p></html>"""
    atomic_text(directory / "status.html", page)


def check_gate(batch):
    run = Path(batch["directory"])
    plan = load_plan(run)
    if plan["id"] != batch["plan_id"]:
        raise ValueError("Campaign plan identity changed")
    for filename in ["model-probe-audit.json", "source-visual-review.json"]:
        review = json.loads((run / filename).read_text())
        if review["plan_id"] != plan["id"] or review["decision"] != "proceed":
            raise ValueError(f"Missing matching approval: {run / filename}")


def export_records(out, records, cohorts, flags):
    report = validate_dataset(records)
    mixed = [r for r in records if r["construction"]["kind"] == "mixed"]
    controls = [r for r in records if r["construction"]["kind"] == "human_control"]
    if len(mixed) != len(controls) or len({r["source"]["id"] for r in mixed}) != len(
        mixed
    ):
        raise ValueError("Expected one mixed document and one control per source")
    write_jsonl(out / "dataset.jsonl", records)
    write_jsonl(out / "mixed.jsonl", mixed)
    write_jsonl(out / "human-controls.jsonl", controls)
    for split in ["train", "validation", "test"]:
        write_jsonl(out / f"{split}.jsonl", (r for r in records if r["split"] == split))
    report.update(
        created_at=timestamp(),
        complete=True,
        mixed_documents=len(mixed),
        human_controls=len(controls),
        cohorts=cohorts,
        models=dict(
            Counter(
                r["replacements"][0]["generation"]["provenance"]["backend"]
                for r in mixed
            )
        ),
        prose_review_flags=flags,
        publication_ready=not flags,
        limitations=[
            "Human-origin candidates retain documentary provenance, not certified authorship.",
            "Some PDF formula symbols are flattened during extraction.",
            "New general text is predominantly historical fiction.",
            "Codex model attribution may be requested-only.",
            "Factual drift and length differences are nonblocking.",
        ],
        files={
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in [
                "dataset.jsonl",
                "mixed.jsonl",
                "human-controls.jsonl",
                "train.jsonl",
                "validation.jsonl",
                "test.jsonl",
            ]
        },
    )
    write_json(out / "manifest.json", report)
    return report


def combine(directory, campaign):
    records, cohorts, flags = [], [], []
    for batch in campaign["batches"]:
        run = Path(batch["directory"])
        report = json.loads((run / "detector-quality-audit.json").read_text())
        parent = json.loads((run / "parent-copy-audit.json").read_text())
        if not (
            report["complete"]
            and report["integrity_checks"] == "pass"
            and report["plan_id"] == parent["plan_id"] == batch["plan_id"]
            and parent["covered_sources"] == parent["total_sources"]
            and not parent["flagged_blocks"]
        ):
            raise ValueError(f"Incomplete or failed audit: {run}")
        data = read_jsonl(run / "dataset.jsonl")
        if any(r["construction"]["plan_id"] != batch["plan_id"] for r in data):
            raise ValueError("Dataset contains another plan")
        records.extend(data)
        flags.extend(report["prose_review_flags"])
        cohorts.append(
            {
                **batch,
                "mixed": report["mixed_documents"],
                "controls": report["human_controls"],
                "models": report["models"],
                "integrity_checks": "pass",
                "full_parent_copy_check": "pass",
                "dataset_sha256": hashlib.sha256(
                    (run / "dataset.jsonl").read_bytes()
                ).hexdigest(),
            }
        )
    new = export_records(directory / "new-data", records, cohorts, flags)
    if (
        new["mixed_documents"] != campaign["target_mixed"]
        or new["models"] != campaign["models"]
    ):
        raise ValueError("Campaign output quota mismatch")
    previous = Path(campaign["previous"])
    manifest = json.loads((previous / "manifest.json").read_text())
    expected = manifest["files"]["dataset.jsonl"]
    if (
        hashlib.sha256((previous / "dataset.jsonl").read_bytes()).hexdigest()
        != expected
    ):
        raise ValueError("Previous dataset hash changed")
    combined = export_records(
        directory / "with-previous",
        read_jsonl(previous / "dataset.jsonl") + records,
        manifest["cohorts"] + cohorts,
        flags,
    )
    write_json(directory / "review-needed.json", flags)
    return new, combined


def run_campaign(directory, jobs=20, writer_jobs=2, brief_jobs=4, max_attempts=8):
    directory = directory.resolve()
    campaign = json.loads((directory / "campaign.json").read_text())
    for batch in campaign["batches"]:
        check_gate(batch)
    with (directory / ".supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        state_path = directory / "status.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "started_at": timestamp(),
                "phase": "starting",
                "batches": [
                    {
                        "name": b["name"],
                        "plan_id": b["plan_id"],
                        "phase": "pending",
                        "mixed": 15,
                        "attempts": 0,
                    }
                    for b in campaign["batches"]
                ],
            }
        )
        if [b["plan_id"] for b in state["batches"]] != [
            b["plan_id"] for b in campaign["batches"]
        ]:
            raise ValueError("Status belongs to another campaign")
        state["pid"] = os.getpid()
        env = dict(os.environ, PYTHONPATH=str(ROOT))

        def pause(until, message):
            state.update(phase="sleeping", resume_at=until, message=message)
            publish_status(directory, state)
            while time.time() < until:
                time.sleep(min(30, until - time.time()))
            state.pop("resume_at", None)

        try:
            for index, batch in enumerate(campaign["batches"]):
                entry = state["batches"][index]
                if entry.get("finished_at"):
                    continue
                until = state.get("resume_at", 0)
                if index:
                    until = max(
                        until,
                        state["batches"][index - 1]["finished_at"]
                        + campaign["pause_seconds"],
                    )
                if remaining_pause(until, 0, time.time()):
                    pause(
                        until,
                        "30-minute pause or existing retry cooldown; successful calls retained.",
                    )
                run = Path(batch["directory"])
                while entry["attempts"] < max_attempts:
                    entry.update(phase="generating", attempts=entry["attempts"] + 1)
                    state.update(
                        phase="generating",
                        current_batch=batch["name"],
                        message="Successful calls are cached; subscriptions only.",
                    )
                    publish_status(directory, state)
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/run_span_expansion.py",
                        "--run",
                        str(run),
                        "--scope",
                        "all",
                        "--jobs",
                        str(jobs),
                        "--writer-jobs",
                        str(writer_jobs),
                        "--brief-jobs",
                        str(brief_jobs),
                        "--provider-error-limit",
                        "8",
                    ]
                    with (run / "overnight.log").open("a") as log:
                        log.write(f"\n{timestamp()} attempt {entry['attempts']}\n")
                        log.flush()
                        with subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        ) as child:
                            for line in child.stdout:
                                log.write(line)
                                log.flush()
                                try:
                                    progress = json.loads(line)
                                except ValueError:
                                    continue
                                if "mixed" in progress:
                                    entry["mixed"] = progress["mixed"]
                                if "completed_tasks" in progress:
                                    entry["progress"] = progress
                                publish_status(directory, state)
                            code = child.wait()
                    if code == 0:
                        break
                    entry["phase"] = "retry_cooldown"
                    entry["last_exit_code"] = code
                    if entry["attempts"] >= max_attempts:
                        raise RuntimeError(
                            f"{batch['name']} exhausted {max_attempts} resumable attempts; see overnight.log"
                        )
                    pause(
                        time.time() + campaign["pause_seconds"],
                        "Generation failed; cached successes retained. Retrying after 30 minutes.",
                    )
                else:
                    raise RuntimeError(
                        "Retry attempts exhausted; increase --max-attempts to resume"
                    )
                entry["phase"] = state["phase"] = "auditing"
                publish_status(directory, state)
                with (run / "overnight.log").open("a") as log:
                    for script, extra in [
                        ("audit_parent_copy.py", []),
                        ("audit_span_pilot.py", ["--require-complete"]),
                    ]:
                        subprocess.run(
                            [
                                sys.executable,
                                "-u",
                                f"scripts/{script}",
                                "--run",
                                str(run),
                                *extra,
                            ],
                            cwd=ROOT,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                report = json.loads((run / "detector-quality-audit.json").read_text())
                parent = json.loads((run / "parent-copy-audit.json").read_text())
                if parent["flagged_blocks"]:
                    raise ValueError(
                        "Full-parent copy audit failed; outputs retained for review"
                    )
                entry.update(
                    phase="complete",
                    mixed=report["mixed_documents"],
                    models=report["models"],
                    review_flags=len(report["prose_review_flags"]),
                    finished_at=time.time(),
                )
                publish_status(directory, state)
            state["phase"] = "combining"
            publish_status(directory, state)
            new, combined = combine(directory, campaign)
            state.update(
                phase="complete_with_review_flags"
                if new["prose_review_flags"]
                else "complete",
                completed_at=timestamp(),
                message="New and cumulative JSONL exports are ready; no automatic HF upload.",
                new_mixed=new["mixed_documents"],
                combined_mixed=combined["mixed_documents"],
            )
            publish_status(directory, state)
        except BaseException as error:
            state.update(phase="stopped", message=str(error))
            publish_status(directory, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--writer-jobs", type=int, default=2)
    parser.add_argument("--brief-jobs", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=8)
    args = parser.parse_args()
    if min(args.jobs, args.writer_jobs, args.brief_jobs, args.max_attempts) < 1:
        parser.error("Concurrency and retry values must be positive")
    run_campaign(
        args.campaign, args.jobs, args.writer_jobs, args.brief_jobs, args.max_attempts
    )
