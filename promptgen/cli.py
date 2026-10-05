from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from heterogeneous.core import atomic_text, read_jsonl
from heterogeneous.providers import ProviderError
from .config import load_config
from .pipeline import Runner, load_plan, make_plan, plain, save_plan
from .templates import Templates, load_tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic user messages by continuing an open user prefix")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--input", type=Path, help="seed conversations JSONL, each ending in an assistant reply")
        command.add_argument("--out", type=Path, required=True)
        if name == "run":
            command.add_argument("--jobs", type=int, default=8)
    command = commands.add_parser("resume")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--jobs", type=int, default=8)
    try:
        args = parser.parse_args(argv)
        if getattr(args, "jobs", 1) < 1:
            raise ValueError("jobs must be positive")
        if args.command == "resume":
            plan = load_plan(args.out)
            config = plan["config"]
        else:
            config = load_config(args.config)
        templates = Templates(load_tokenizer(config["model"]), config["model"])
        if args.command != "resume":
            plan = make_plan(config, templates, read_jsonl(args.input) if args.input else [])
            save_plan(plan, args.out)
        if args.command == "plan":
            first = plan["samples"][0]
            report = {"plan_id": plan["id"], "candidates": len(plan["samples"]),
                      "method": config["generation"]["method"], "tokenizer": templates.identity,
                      "first_prefix": templates.user_prompt(plain(first["messages"]))}
            atomic_text(args.out / "first-prefix.txt", report["first_prefix"])
        else:
            report = Runner(args.out, plan, templates).run(args.jobs)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, ProviderError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
