from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .core import read_jsonl
from .evaluate import evaluate
from .pipeline import Runner, load_plan, make_plan, save_plan, validate_dataset
from .providers import ProviderError


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate span-labelled non-AI/AI mixed documents")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name, help="plan without model calls" if name == "plan" else "plan, summarize, generate and assemble")
        command.add_argument("--input", type=Path, required=True, help="verified non-AI source JSONL")
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "run":
            command.add_argument("--jobs", type=int, default=4)
            command.add_argument("--backend", help="one generator name, otherwise all")
    for name in ("summarize", "generate", "assemble"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True, help="directory containing plan.json")
        if name in ("summarize", "generate"):
            command.add_argument("--jobs", type=int, default=4)
        if name in ("generate", "assemble"):
            command.add_argument("--backend", help="one generator name, otherwise all")
        if name == "assemble":
            command.add_argument("--allow-partial", action="store_true", help="explicitly allow missing variants, recorded in manifest")
    command = commands.add_parser("validate", help="validate offsets, source recovery and split isolation")
    command.add_argument("dataset", type=Path)
    command = commands.add_parser("html", help="export mixed documents with green human and orange AI spans")
    command.add_argument("dataset", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--limit", type=int, default=10, help="maximum mixed documents to display")
    command = commands.add_parser("viewer", help="explore local JSONL datasets with colored spans and original-text comparison")
    command.add_argument("datasets", nargs="*", type=Path, help="JSONL files or release/run directories; defaults to the latest local release")
    command.add_argument("--repo", help="download records/all.jsonl from a Hugging Face dataset (requires huggingface_hub)")
    command.add_argument("--revision", default="main", help="Hub revision when using --repo")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8770)
    command = commands.add_parser("evaluate", help="compare predicted spans with gold spans")
    command.add_argument("--gold", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--boundary-tolerance", type=int, default=20)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "jobs") and args.jobs < 1:
            raise ValueError("--jobs must be positive")
        if args.command == "viewer":
            from .viewer import serve
            serve(args.datasets, args.host, args.port, args.repo, args.revision)
            return
        elif args.command == "validate":
            report = validate_dataset(read_jsonl(args.dataset))
        elif args.command == "html":
            from .html import render_html
            report = render_html(read_jsonl(args.dataset), args.output, args.limit)
        elif args.command == "evaluate":
            if args.boundary_tolerance < 0:
                raise ValueError("boundary tolerance must be nonnegative")
            gold = read_jsonl(args.gold)
            validate_dataset(gold)
            report = evaluate(gold, read_jsonl(args.predictions), args.boundary_tolerance)
        else:
            directory = args.out.resolve()
            if args.command in ("plan", "run"):
                plan = make_plan(args.input, load_config(args.config))
                save_plan(plan, directory)
            else:
                plan = load_plan(directory)
            runner = Runner(directory, plan)
            if args.command == "plan":
                blocks = sum(len(v["blocks"]) for v in plan["variants"])
                report = {"plan_id": plan["id"], "sources": len(plan["sources"]), "variants": len(plan["variants"]),
                          "summary_calls_before_retries": blocks,
                          "generation_calls_before_retries": sum(len(v["blocks"]) * len(runner.variant_backends(v)) for v in plan["variants"]),
                          "skipped_variants": len(plan["skipped"]), "out": str(directory)}
            elif args.command == "run":
                runner.backends(args.backend)
                runner.stage("summarize", args.jobs)
                runner.stage("generate", args.jobs, args.backend)
                report = runner.assemble(args.backend)
            elif args.command == "assemble":
                report = runner.assemble(args.backend, args.allow_partial)
            else:
                runner.stage(args.command, args.jobs, getattr(args, "backend", None))
                report = {"stage": args.command, "status": "complete", "out": str(directory)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, ProviderError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
