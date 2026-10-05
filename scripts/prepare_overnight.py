"""Freeze four balanced batches of unused historical/source-candidate passages.

No model calls. Excerpts from the same paper/author retain the same split, even
across batches. The frozen manifest records the construction-parameter sweep.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

import requests

from heterogeneous.core import (
    digest,
    normalized_hash,
    paragraph_spans,
    read_jsonl,
    select_blocks,
    text_hash,
    write_json,
    write_jsonl,
)
from heterogeneous.pipeline import make_plan, save_plan
from scripts.audit_parent_copy import prepare_policy

MODELS = ["haiku", "sonnet", "opus", "gpt_sol", "gpt_luna"]
PROFILES = [
    dict(
        name="one_block",
        block_sizes=[2, 3, 4],
        max_blocks=1,
        max_replaced_fraction=0.55,
        summary_max_sentences=3,
        context_chars=400,
    ),
    dict(
        name="moderate",
        block_sizes=[2, 3, 5],
        max_blocks=2,
        max_replaced_fraction=0.60,
        summary_max_sentences=3,
        context_chars=600,
    ),
    dict(
        name="more_context",
        block_sizes=[2, 4, 6],
        max_blocks=2,
        max_replaced_fraction=0.65,
        summary_max_sentences=3,
        context_chars=900,
    ),
    dict(
        name="richer_briefs",
        block_sizes=[2, 3, 4, 5],
        max_blocks=2,
        max_replaced_fraction=0.60,
        summary_max_sentences=4,
        context_chars=600,
    ),
]
# Historical works. Actual archive header, retrieved bytes and raw hash are saved.
# Exclude titles already used, including editions from a different archive.
BOOKS = [
    (1342, "Pride and Prejudice", "Jane Austen"),
    (158, "Emma", "Jane Austen"),
    (161, "Sense and Sensibility", "Jane Austen"),
    (105, "Persuasion", "Jane Austen"),
    (141, "Mansfield Park", "Jane Austen"),
    (121, "Northanger Abbey", "Jane Austen"),
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (1400, "Great Expectations", "Charles Dickens"),
    (730, "Oliver Twist", "Charles Dickens"),
    (1023, "Bleak House", "Charles Dickens"),
    (786, "Hard Times", "Charles Dickens"),
    (46, "A Christmas Carol", "Charles Dickens"),
    (2701, "Moby Dick", "Herman Melville"),
    (345, "Dracula", "Bram Stoker"),
    (768, "Wuthering Heights", "Emily Bronte"),
    (219, "Heart of Darkness", "Joseph Conrad"),
    (643, "The Strange Case of Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson"),
    (174, "The Picture of Dorian Gray", "Oscar Wilde"),
    (5230, "The Canterville Ghost", "Oscar Wilde"),
    (1237, "The Mystery of the Yellow Room", "Gaston Leroux"),
    (514, "Little Women", "Louisa May Alcott"),
    (145, "Middlemarch", "George Eliot"),
    (4217, "A Portrait of the Artist as a Young Man", "James Joyce"),
    (4300, "Ulysses", "James Joyce"),
    (2852, "The Hound of the Baskervilles", "Arthur Conan Doyle"),
    (36, "The War of the Worlds", "H. G. Wells"),
    (35, "The Time Machine", "H. G. Wells"),
    (159, "The Invisible Man", "H. G. Wells"),
    (215, "The Call of the Wild", "Jack London"),
    (910, "White Fang", "Jack London"),
    (25344, "The Scarlet Letter", "Nathaniel Hawthorne"),
    (45, "Anne of Green Gables", "Lucy Maud Montgomery"),
    (113, "The Secret Garden", "Frances Hodgson Burnett"),
    (1155, "The Secret Agent", "Joseph Conrad"),
    (209, "The Turn of the Screw", "Henry James"),
    (805, "This Side of Paradise", "F. Scott Fitzgerald"),
    (940, "The Last of the Mohicans", "James Fenimore Cooper"),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas"),
    (135, "Les Miserables", "Victor Hugo"),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        unicodedata.normalize("NFKD", value or "")
        .encode("ascii", "ignore")
        .decode()
        .lower(),
    )


def free_runs(paragraphs, occupied=()):
    """Indices of continuous eligible prose; never bridge a removed paragraph."""
    current = []
    for index, p in enumerate(paragraphs):
        good = p.get("eligible_prose", True) and re.search(
            r"""[.!?]["'”’\)\]]*$""", p["text"]
        )
        if good and not any(a <= index < b for a, b in occupied):
            current.append(index)
        else:
            if current:
                yield current
            current = []
    if current:
        yield current


def chunks(indices, paragraphs, minimum=3, target_chars=5000):
    cursor = 0
    while cursor + minimum <= len(indices):
        chosen = []
        chars = 0
        for index in indices[cursor:]:
            if (
                len(chosen) >= minimum
                and chars + len(paragraphs[index]["text"]) > target_chars
            ):
                break
            chosen.append(index)
            chars += len(paragraphs[index]["text"]) + 2
        yield chosen
        cursor += len(chosen)


def paper_pool(cache, old):
    occupied = defaultdict(list)
    for source in old:
        if source["dataset"] == "jmlr_pre2015":
            indices = source["paragraph_indices"]
            occupied[source["group_id"]].append((indices[0], indices[-1] + 1))
    pool = []
    checked = {}
    for path in sorted(cache.glob("v*.json")):
        doc = json.loads(path.read_text())
        pdf = Path(doc["pdf_path"])
        if doc["year"] != doc["pdf_publication_year"] or doc["year"] >= 2015:
            continue
        # Source cache paths can be symlinks to the original acquisition.
        for f, expected in [
            (pdf, doc["pdf_sha256"]),
            (pdf.with_suffix(".xml"), doc["layout_xml_sha256"]),
            (pdf.parent / f"volume-{doc['volume']}.html", doc["index_sha256"]),
        ]:
            if f not in checked:
                checked[f] = sha(f)
            if checked[f] != expected:
                raise ValueError(f"Source integrity failure: {f}")
        paragraphs = doc["paragraphs"]
        for segment in free_runs(paragraphs, occupied[doc["id"]]):
            # Fixed disjoint three-paragraph units allow genuinely new excerpts
            # from existing papers; publication/source checks remain unchanged.
            for cursor in range(0, len(segment) - 2, 3):
                indices = segment[cursor : cursor + 3]
                text = "\n\n".join(paragraphs[i]["text"] for i in indices)
                if len(text) < 900:
                    continue
                pool.append(
                    {
                        "id": f"{doc['id']}:paragraphs:{indices[0]}-{indices[-1] + 1}",
                        "group_id": doc["id"],
                        "reference": doc["reference"],
                        "text": text,
                        "author_id": doc["authors"],
                        "title": doc["title"],
                        "dataset": "jmlr_pre2015",
                        "genre": "machine_learning_research",
                        "language": "en",
                        "human_verified": False,
                        "authorship": "human_candidate",
                        "binary_target": None,
                        "license": "JMLR author copyright; journal CC-BY policy recorded, historical paper-specific license unconfirmed",
                        "provenance": "Cached original journal PDF/index agree on pre-2015 publication; new continuous, non-overlapping prose excerpt.",
                        "human_origin_basis": {
                            "kind": "historical_journal_paper_pre2015",
                            "publication_year": doc["year"],
                            "pdf_publication_year": doc["pdf_publication_year"],
                            "pdf_sha256": doc["pdf_sha256"],
                        },
                        "upstream_metadata": {
                            k: v for k, v in doc.items() if k != "paragraphs"
                        },
                        "paragraph_indices": indices,
                        "preprocessing": {
                            "method": "Existing deterministic Poppler layout/paragraph extraction; no newly generated wording",
                            "selected_paragraphs": [paragraphs[i] for i in indices],
                            "no_model_used": True,
                        },
                    }
                )
    return pool


def book_pool(root, old):
    raw = root / "sources/gutenberg/raw"
    raw.mkdir(parents=True, exist_ok=True)
    used_titles = {canonical(s.get("title")) for s in old if s["genre"] == "fiction"}
    pool = []
    parents = []
    excluded = []
    for book_id, title, author in BOOKS:
        if canonical(title) in used_titles:
            continue
        path = raw / f"book-{book_id}.txt"
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        if not path.exists():
            response = requests.get(
                url,
                timeout=60,
                headers={
                    "User-Agent": "pangram-at-home-research/1.0 (historical prose)"
                },
            )
            response.raise_for_status()
            if len(response.content) < 20000:
                raise ValueError(f"Unexpectedly short archive text: {book_id}")
            path.write_bytes(response.content)
        text = path.read_bytes().decode("utf-8-sig")
        start = re.search(
            r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\r?\n",
            text,
        )
        end = re.search(
            r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text
        )
        if not start or not end or start.end() >= end.start():
            raise ValueError(f"Archive body markers missing: {book_id}")
        header = text[: start.start()]
        actual_author = re.search(r"(?m)^Author:\s*(.+)", header)
        actual_title = re.search(r"(?m)^Title:\s*(.+)", header)
        if (
            not actual_author
            or canonical(author) not in canonical(actual_author[1])
            or not actual_title
            or canonical(title) not in canonical(actual_title[1])
        ):
            excluded.append(
                {
                    "book_id": book_id,
                    "reason": "Archive title/author did not match the requested historical work",
                    "actual_title": actual_title[1].strip() if actual_title else None,
                    "actual_author": actual_author[1].strip()
                    if actual_author
                    else None,
                }
            )
            continue
        meta = {
            "parent_document_id": f"gutenberg/{book_id}",
            "source_file": str(path.resolve()),
            "source_file_sha256": sha(path),
            "download_url": url,
            "archive_header": header,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
        parents.append({"book_id": book_id, "title": title, "author": author, **meta})
        units = paragraph_spans(text[start.end() : end.start()])
        paragraphs = []
        for a, b in units:
            a += start.end()
            b += start.end()
            paragraphs.append({"text": text[a:b], "start": a, "end": b})
        for segment in free_runs(paragraphs):
            for indices in chunks(segment, paragraphs, minimum=6, target_chars=6500):
                a = paragraphs[indices[0]]["start"]
                b = paragraphs[indices[-1]]["end"]
                excerpt = text[a:b]
                if len(excerpt) < 1200:
                    continue
                pool.append(
                    {
                        "id": f"span/gutenberg/{book_id}/{a}-{b}",
                        "group_id": f"book-author/{author}",
                        "text": excerpt,
                        "reference": f"https://www.gutenberg.org/ebooks/{book_id}",
                        "author_id": author,
                        "title": title,
                        "dataset": "gutenberg_selected",
                        "genre": "fiction",
                        "language": "en",
                        "human_verified": False,
                        "authorship": "human_candidate",
                        "binary_target": None,
                        "license": "Public domain in USA declared by Project Gutenberg; archive/trademark terms and jurisdiction-specific rights retained",
                        "provenance": "Identified historical work in the public archive; raw bytes/header pinned; captured edition not independently archived before LLMs.",
                        "human_origin_basis": {
                            "kind": "identified_historical_human_authored_work",
                            "archive_book_id": book_id,
                            "captured_revision_date_verified": False,
                        },
                        "upstream_metadata": deepcopy(meta),
                        "parent_character_range": [a, b],
                    }
                )
        print(json.dumps({"book": title, "fiction_pool": len(pool)}), flush=True)
    write_jsonl(root / "sources/gutenberg/parents.jsonl", parents)
    write_json(root / "sources/gutenberg/exclusions.json", excluded)
    return pool


def documentary_pool(root, corpus):
    """Use the provenance-screened, named-author diverse release, not the news pool."""
    manifest = json.loads((corpus / "manifest.json").read_text())
    for name in ["train.jsonl.gz", "parent_documents.jsonl.gz"]:
        if sha(corpus / name) != manifest["outputs"][name]:
            raise ValueError("Diverse release checksum mismatch")
    with gzip.open(corpus / "parent_documents.jsonl.gz", "rt") as f:
        parents = {r["id"]: r for r in map(json.loads, f)}
    target = root / "sources/documentary-parents"
    target.mkdir(parents=True, exist_ok=True)
    pool = []
    with gzip.open(corpus / "train.jsonl.gz", "rt") as f:
        for row in map(json.loads, f):
            if row["dataset"] not in ["cc_news", "common_pile_news", "hansard"]:
                continue
            if (
                row.get("authorship") != "human_candidate"
                or row.get("binary_target") is not None
            ):
                continue
            parent = parents[row["parent_document_id"]]
            if (
                text_hash(row["text"]) != row["text_sha256"]
                or text_hash(parent["text"]) != parent["text_sha256"]
            ):
                raise ValueError("Source text checksum mismatch")
            parent_hash = text_hash(parent["text"])
            path = target / f"{parent_hash}.txt"
            if not path.exists():
                path.write_text(parent["text"], encoding="utf-8", newline="")
            text = row["text"]
            units = paragraph_spans(text)
            paragraphs = [{"text": text[a:b], "start": a, "end": b} for a, b in units]
            for segment in free_runs(paragraphs):
                for indices in chunks(
                    segment, paragraphs, minimum=4, target_chars=6500
                ):
                    a = paragraphs[indices[0]]["start"]
                    b = paragraphs[indices[-1]]["end"]
                    excerpt = text[a:b]
                    if len(excerpt) < 1200:
                        continue
                    pool.append(
                        {
                            "id": f"overnight:{row['id']}:{a}-{b}",
                            "group_id": row["split_group"],
                            "split": row["split"],
                            "reference": row.get("url") or row.get("canonical_url"),
                            "text": excerpt,
                            "author_id": row.get("author"),
                            "title": row.get("title")
                            or (excerpt.splitlines()[0][:120]),
                            "genre": row.get("genre", "documentary_prose"),
                            "dataset": row["dataset"],
                            "language": "en",
                            "human_verified": False,
                            "authorship": "human_candidate",
                            "binary_target": None,
                            "license": row["license"],
                            "provenance": "Pinned provenance-screened diverse release; named byline/historical parliamentary evidence retained.",
                            "human_origin_basis": row["human_origin_basis"],
                            "upstream_metadata": {
                                **{
                                    k: v
                                    for k, v in row.items()
                                    if k not in ["text", "source_text"]
                                },
                                "parent_text_path": str(path.resolve()),
                                "parent_text_sha256": parent_hash,
                                "release_manifest_sha256": sha(
                                    corpus / "manifest.json"
                                ),
                            },
                            "parent_character_range": [
                                row["character_range"][0] + a,
                                row["character_range"][0] + b,
                            ],
                        }
                    )
    return pool


def eligible(source, settings, seed):
    source["sha256"] = text_hash(source["text"])
    source["normalized_sha256"] = normalized_hash(source["text"])
    return bool(
        select_blocks(
            source["text"], settings, digest([seed, source["id"], source["sha256"], 0])
        )
    )


def prepare(root, previous, cache, documentary, seed):
    if (root / "campaign.json").exists():
        raise ValueError("Campaign already frozen; use the runner to resume it")
    old_records = read_jsonl(previous / "dataset.jsonl")
    old = {r["source"]["id"]: r["source"] for r in old_records}
    group_splits = {}
    author_groups = {}
    for r in old_records:
        s = r["source"]
        split = r["split"]
        if group_splits.setdefault(s["group_id"], split) != split:
            raise ValueError("Prior group split conflict")
        if s.get("genre") == "fiction" and s.get("author_id"):
            author_groups[canonical(s["author_id"])] = s["group_id"]
    # Preserve the donor corpus's held-out author partitions for newly acquired
    # books too. Existing generated-record assignments take precedence.
    core = Path("../pretraining-datawork/data/corpus-human-diverse-core-v1")
    for split in ["train", "validation", "test"]:
        with gzip.open(core / f"{split}.jsonl.gz", "rt") as f:
            for source in map(json.loads, f):
                if source["dataset"] != "gutenberg_selected":
                    continue
                group_splits.setdefault(source["split_group"], source["split"])
                author_groups.setdefault(
                    canonical(source["author"]), source["split_group"]
                )
    claude = json.loads(
        Path("runs/heterogeneous-expansion-20261005/plan.json").read_text()
    )["config"]
    gpt = json.loads(Path("runs/gpt-general-20261005/plan.json").read_text())["config"]
    config = deepcopy(claude)
    config["generators"] += deepcopy(gpt["generators"])
    for b in config["generators"]:
        b["concurrency"] = 2
    root.mkdir(parents=True, exist_ok=True)
    pools = {
        "ml": paper_pool(cache, list(old.values())),
        "fiction": book_pool(root, list(old.values())),
        "documentary": documentary_pool(root, documentary),
    }
    print(
        "source_pool_counts",
        json.dumps({k: len(v) for k, v in pools.items()}),
        flush=True,
    )
    rng = random.Random(seed)
    for rows in pools.values():
        rng.shuffle(rows)
    used_ids = set(old)
    used_hashes = {normalized_hash(s["text"]) for s in old.values()}
    selected_ranges = defaultdict(list)
    batches = []
    for number, profile in enumerate(PROFILES, 1):
        run = root / f"batch-{number:02d}"
        settings = deepcopy(config["dataset"])
        settings.update({k: v for k, v in profile.items() if k != "name"})
        settings.update(
            seed=seed + number,
            variants_per_source=1,
            min_blocks=1,
            summary_min_sentences=2,
        )
        batch_config = deepcopy(config)
        batch_config["dataset"] = settings
        selected = []
        author_counts = Counter()

        def take(pool_name, name, count):
            rows = pools[pool_name]
            kept = []
            picked = 0
            for source in rows:
                if picked >= count:
                    kept.append(source)
                    continue
                s = deepcopy(source)
                key = s["group_id"]
                h = normalized_hash(s["text"])
                if s["id"] in used_ids or h in used_hashes:
                    continue
                if pool_name == "fiction" and author_counts[key] >= 55:
                    kept.append(source)
                    continue
                position = (
                    (s["paragraph_indices"][0], s["paragraph_indices"][-1] + 1)
                    if pool_name == "ml"
                    else s.get("parent_character_range")
                )
                parent = (
                    key
                    if pool_name == "ml"
                    else s["upstream_metadata"].get(
                        "parent_document_id",
                        s["upstream_metadata"].get("parent_text_sha256", key),
                    )
                )
                if position and any(
                    position[0] < b and a < position[1]
                    for a, b in selected_ranges[parent]
                ):
                    continue
                if not eligible(s, settings, settings["seed"]):
                    kept.append(source)
                    continue
                if pool_name == "fiction":
                    alias = author_groups.get(canonical(s["author_id"]))
                    if alias:
                        s["group_id"] = alias
                        key = alias
                if key in group_splits:
                    s["split"] = group_splits[key]
                s["generator_names"] = [name]
                selected.append(s)
                picked += 1
                used_ids.add(s["id"])
                used_hashes.add(h)
                author_counts[key] += 1
                if position:
                    selected_ranges[parent].append(position)
            pools[pool_name] = kept
            if picked != count:
                raise ValueError(
                    f"Insufficient {pool_name} passages for batch {number}/{name}: {picked}/{count}"
                )

        for name in MODELS:
            if name == "opus":
                take("fiction", name, 200)
            else:
                take("ml", name, 100)
                # Documentary passages supplement fiction while that smaller
                # provenance-screened pool lasts; remaining general texts are books.
                target = min(35, len(pools["documentary"]))
                before = len(selected)
                try:
                    take("documentary", name, target)
                except ValueError:
                    # Successfully selected texts stay selected; top up from fiction.
                    pass
                take("fiction", name, 100 - (len(selected) - before))
        rng.shuffle(selected)
        run.mkdir(parents=True)
        write_jsonl(run / "input.jsonl", selected)
        write_json(run / "config.json", batch_config)
        plan = make_plan(run / "input.jsonl", batch_config)
        if plan["skipped"] or len(plan["variants"]) != 1000:
            raise ValueError("Unexpected skipped sources in frozen batch")
        for source in plan["sources"]:
            split = plan["splits"][source["id"]]
            if group_splits.setdefault(source["group_id"], split) != split:
                raise ValueError("Campaign group split conflict")
        save_plan(plan, run)
        models = Counter(v["generator_names"][0] for v in plan["variants"])
        if models != Counter({name: 200 for name in MODELS}):
            raise ValueError("Unbalanced campaign")
        lookup = {s["id"]: s for s in plan["sources"]}
        probes = []
        for name in MODELS:
            candidates = [v for v in plan["variants"] if v["generator_names"] == [name]]
            chosen = []
            for dataset_type in [
                "jmlr_pre2015",
                "gutenberg_selected",
                "common_pile_news",
                "cc_news",
                "hansard",
            ]:
                match = next(
                    (
                        v
                        for v in candidates
                        if lookup[v["source_id"]]["dataset"] == dataset_type
                    ),
                    None,
                )
                if match and len(chosen) < 3:
                    chosen.append(match)
            for v in candidates:
                if len(chosen) >= 3:
                    break
                if v not in chosen:
                    chosen.append(v)
            probes.extend(v["id"] for v in chosen)
        write_json(
            run / "verification-selection.json",
            {
                "plan_id": plan["id"],
                "variants": probes,
                "models": {name: 3 for name in MODELS},
            },
        )
        write_json(
            run / "quality-gate.json",
            {
                "plan_id": plan["id"],
                "decision": "proceed",
                "scope": "probe",
                "evaluation_policy": "detector_task_fitness",
                "paragraph_count_policy": "soft_target",
            },
        )
        write_json(
            run / "source-audit.json",
            {
                "plan_id": plan["id"],
                "input_sha256": sha(run / "input.jsonl"),
                "sources": 1000,
                "models": dict(models),
                "collections": dict(Counter(s["dataset"] for s in selected)),
                "unique_papers": len(
                    {s["group_id"] for s in selected if s["dataset"] == "jmlr_pre2015"}
                ),
                "profile": profile,
                "previous_dataset_sha256": sha(previous / "dataset.jsonl"),
                "checks": [
                    "Pinned PDF/XML/index integrity; matching pre-2015 journal years",
                    "Continuous eligible paper paragraphs; no overlap with prior selected paper ranges",
                    "Historical-book archive header/raw hashes; previously used book titles excluded",
                    "Documentary release/parent text checksums; original human-candidate status preserved",
                    "Unique source IDs and normalized text across campaign/previous data",
                    "One writer per source; prior and across-batch group splits retained; Opus fiction only",
                ],
                "limitations": [
                    "New paper excerpts can be three paragraphs long; several disjoint excerpts can share a paper.",
                    "Source verification is documentary evidence, not independent certification.",
                ],
            },
        )
        prepare_policy(run, plan)
        policy = json.loads((run / "parent-copy-policy.json").read_text())
        if policy["uncovered_sources"]:
            raise ValueError("Missing full-parent copy references")
        batches.append(
            {
                "name": f"overnight_batch_{number:02d}",
                "directory": str(run.resolve()),
                "plan_id": plan["id"],
                "profile": profile,
                "mixed_target": 1000,
                "models": dict(models),
                "ml": 400,
                "general": 600,
                "probe_documents": len(probes),
            }
        )
        print(
            json.dumps(
                {
                    "batch": number,
                    "plan_id": plan["id"],
                    "sources": 1000,
                    "blocks": sum(len(v["blocks"]) for v in plan["variants"]),
                }
            ),
            flush=True,
        )
    campaign = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "previous": str(previous.resolve()),
        "target_mixed": 4000,
        "target_controls": 4000,
        "models": {name: 800 for name in MODELS},
        "ml": 1600,
        "general": 2400,
        "pause_seconds": 1800,
        "batches": batches,
        "brief_model": config["summarizer"]["model"],
        "quality_policy": "Detector-task prose, exact spans/provenance and 15% excerpt/full-parent copy checks; length/factual differences nonblocking.",
    }
    write_json(root / "campaign.json", campaign)
    print(
        json.dumps(
            {
                "campaign": str(root.resolve()),
                "target": 4000,
                "models": campaign["models"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--previous", type=Path, default=Path("runs/heterogeneous-20261005-with-gpt")
    )
    p.add_argument(
        "--paper-cache",
        type=Path,
        default=Path("runs/ml-papers-20261005/sources/jmlr/raw"),
    )
    p.add_argument(
        "--documentary",
        type=Path,
        default=Path("../pretraining-datawork/data/corpus-human-diverse-v1"),
    )
    p.add_argument("--seed", type=int, default=2026100505)
    a = p.parse_args()
    prepare(a.out, a.previous, a.paper_cache, a.documentary, a.seed)
