"""Install an isolated inference environment and execute a detached sweep."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    args.runtime.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_HUB_CACHE=str(args.runtime / "model-cache"), OMP_NUM_THREADS="4",
               VLLM_USE_FLASHINFER_SAMPLER="0")
    if args.token_file and args.token_file.is_file():
        env["HF_TOKEN"] = args.token_file.read_text().strip()
    state = {"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}

    def update(stage, **fields):
        state.update(stage=stage, **fields)
        (args.out / "bootstrap-status.json").write_text(json.dumps(state, indent=2) + "\n")
        print(f"bootstrap: {stage}", flush=True)

    try:
        venv = args.runtime / "venv"
        if not (venv / "bin/python").exists():
            update("creating_environment")
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True, env=env)
        python = str(venv / "bin/python")
        update("installing_inference_dependencies")
        subprocess.run([python, "-m", "pip", "install", "--upgrade", "vllm>=0.19", "mistral-common>=1.9"], check=True, env=env)
        subprocess.run([python, "-m", "pip", "install", "-e", str(args.project) + "[prompts]"], check=True, env=env)
        update("checking_project")
        subprocess.run([python, "-m", "unittest", "discover", "-s", "tests", "-q"],
                       cwd=args.project, env=env, check=True)
        update("running_sweep")
        subprocess.run([python, "-m", "promptgen.sweep", "--manifest", str(args.project / "examples/sweep.json"),
                        "--seeds", str(args.project / "examples/magpie-seeds.jsonl"), "--out", str(args.out)],
                       cwd=args.project, env=env, check=True)
        update("finished")
    except Exception as error:
        update("failed", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
