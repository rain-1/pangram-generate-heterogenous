"""Read-only local JSONL explorer. Index metadata; load document bodies on demand."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit


MODEL_LABELS = {
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-sonnet-5-5": "Sonnet 5.5",
    "claude-opus-5-5": "Opus 5.5",
    "opus-3": "Opus 3",
    "gpt-6.1-sol": "GPT-6.1 Sol",
    "gpt-6-luna": "GPT-6 Luna",
}
COHORT_LABELS = {
    "original": "Original batch",
    "ml_papers": "Original ML papers",
    "expansion": "Claude expansion",
    "gpt_general": "GPT general documents",
    "gpt_ml_papers": "GPT ML papers",
}
COLLECTION_LABELS = {
    "gutenberg_selected": "Project Gutenberg",
    "standardebooks": "Standard Ebooks",
    "wikitext2_raw": "WikiText-2",
    "beigebook": "Beige Book",
    "jmlr_pre2015": "JMLR · pre-2015",
}


def normalize_record(raw, cohorts):
    rich = "source" in raw
    source = raw["source"] if rich else {}
    text = raw["text"]
    original = source["text"] if rich else raw["source_text"]
    if (
        raw.get("offset_unit", "unicode_codepoint") != "unicode_codepoint"
        or raw.get("end_exclusive", True) is not True
    ):
        raise ValueError("Viewer requires exclusive Unicode-code-point span offsets")
    spans = []
    cursor = source_cursor = 0
    for index, span in enumerate(raw["spans"]):
        author = span.get("author", {})
        normalized = {
            k: span.get(k)
            for k in [
                "start",
                "end",
                "label",
                "source_start",
                "source_end",
                "replacement_id",
            ]
        }
        normalized.update(
            author_id=author.get("id") if rich else span.get("author_id"),
            backend=author.get("backend") if rich else span.get("backend"),
            requested_model=author.get("requested_model")
            if rich
            else span.get("requested_model"),
            reported_model=author.get("reported_model")
            if rich
            else span.get("reported_model"),
            model_identity_status=author.get("metadata", {}).get(
                "model_identity_status"
            )
            if rich
            else span.get("model_identity_status"),
            index=index,
        )
        a, b, sa, sb = (
            normalized[k] for k in ["start", "end", "source_start", "source_end"]
        )
        if (
            any(type(v) is not int for v in [a, b, sa, sb])
            or a != cursor
            or sa != source_cursor
            or not a < b <= len(text)
            or not sa < sb <= len(original)
        ):
            raise ValueError(
                "Spans must partition both the output and original text without gaps or overlaps"
            )
        if normalized["label"] not in ["human", "ai"]:
            raise ValueError("Unknown authorship label")
        if normalized["label"] == "human" and text[a:b] != original[sa:sb]:
            raise ValueError(
                "Retained human span does not equal its original source slice"
            )
        cursor = b
        source_cursor = sb
        spans.append(normalized)
    if cursor != len(text) or source_cursor != len(original):
        raise ValueError("Incomplete span coverage")
    ai = [span for span in spans if span["label"] == "ai"]
    model_keys = list(
        dict.fromkeys(
            s["requested_model"]
            or s["reported_model"]
            or s["backend"]
            or "Unreported model"
            for s in ai
        )
    )
    construction = raw.get("construction", {})
    cohort = (
        raw.get("cohort")
        or cohorts.get(construction.get("plan_id"))
        or (
            "ML papers"
            if source.get("dataset", raw.get("source_dataset")) == "jmlr_pre2015"
            else "General documents"
        )
    )
    result = {
        "id": raw["id"],
        "source_id": source.get("id", raw.get("source_id")),
        "title": source.get("title", raw.get("source_title")) or "Untitled document",
        "source_author": source.get("author_id", raw.get("source_author")),
        "source_reference": source.get("reference", raw.get("source_reference")),
        "source_license": source.get("license", raw.get("source_license")),
        "human_verified": source.get("human_verified", raw.get("human_verified")),
        "source_dataset": source.get("dataset", raw.get("source_dataset"))
        or "Unspecified collection",
        "genre": source.get("genre"),
        "split": raw.get("split") or "Unspecified",
        "cohort": cohort,
        "kind": construction.get("kind", raw.get("kind"))
        or ("mixed" if ai else "human_control"),
        "model_keys": model_keys,
        "model_labels": [MODEL_LABELS.get(m, m) for m in model_keys],
        "ai_fraction": sum(s["end"] - s["start"] for s in ai) / len(text)
        if text
        else 0,
        "ai_spans": len(ai),
        "characters": len(text),
        "source_characters": len(original),
        "text": text,
        "source_text": original,
        "spans": spans,
        "replacements": [
            {
                "id": r["id"],
                "brief": r.get("summary"),
                "prompt": r.get("generation", {}).get("prompt"),
                "brief_prompt": r.get("summarization", {}).get("prompt"),
                "quality": r.get("generation", {}).get("quality"),
                "generated_at": r.get("generation", {})
                .get("provenance", {})
                .get("generated_at"),
            }
            for r in raw.get("replacements", [])
        ],
    }
    return result


def resolve_dataset(path):
    # Hub snapshot files are symlinks into a blob store. Keep the snapshot path
    # so its adjacent manifests remain discoverable.
    path = path.expanduser().absolute()
    if path.is_dir():
        for relative in [
            "records/all.jsonl",
            "dataset.jsonl",
            "all.jsonl",
            "mixed.jsonl",
        ]:
            if (path / relative).is_file():
                return path / relative
        raise ValueError(f"No dataset JSONL in {path}")
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    return path


def latest_local_dataset():
    root = Path.cwd() / "releases"
    candidates = []
    for p in root.glob("heterogeneous-ai-spans-v*"):
        match = re.fullmatch(r"heterogeneous-ai-spans-v(\d+)\.(\d+)\.(\d+)", p.name)
        if match and (p / "records/all.jsonl").is_file():
            candidates.append((tuple(map(int, match.groups())), p))
    if not candidates:
        raise ValueError(
            "Provide a JSONL/release path, or use --repo to load the Hub dataset"
        )
    return max(candidates)[1]


class DatasetIndex:
    def __init__(self, path):
        self.path = resolve_dataset(path)
        self.id = hashlib.sha256(str(self.path).encode()).hexdigest()[:16]
        root = (
            self.path.parent.parent
            if self.path.parent.name == "records"
            else self.path.parent
        )
        cohorts = {}
        release = {}
        for name in ["manifest/upstream-handoff.json", "manifest.json"]:
            f = root / name
            if f.is_file():
                for c in json.loads(f.read_text()).get("cohorts", []):
                    cohorts[c["plan_id"]] = c["name"]
        if (root / "manifest/release.json").is_file():
            release = json.loads((root / "manifest/release.json").read_text())
        self.cohorts = cohorts
        self.name = (
            f"Human / AI Spans · v{release['release_version']}"
            if release
            else root.name + " / " + self.path.name
        )
        self.offsets = {}
        self.rows = []
        self.by_id = {}
        self.source_rows = defaultdict(list)
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    record = normalize_record(raw, cohorts)
                    if (
                        not isinstance(record["id"], str)
                        or record["id"] in self.offsets
                    ):
                        raise ValueError("Missing or duplicate document ID")
                except (ValueError, KeyError, TypeError) as e:
                    raise ValueError(
                        f"{self.path.name}, line {line_number}: {e}"
                    ) from e
                self.offsets[record["id"]] = offset
                meta = {
                    k: v
                    for k, v in record.items()
                    if k not in ["text", "source_text", "spans", "replacements"]
                }
                self.rows.append(meta)
                self.by_id[meta["id"]] = meta
                self.source_rows[meta["source_id"]].append(meta)
        if not self.rows:
            raise ValueError("Dataset is empty")
        models = Counter()
        kinds = Counter()
        cohort_counts = Counter()
        collection_counts = Counter()
        splits = Counter()
        for row in self.rows:
            siblings = self.source_rows[row["source_id"]]
            row["filter_models"] = (
                list(
                    dict.fromkeys(
                        m for sibling in siblings for m in sibling["model_keys"]
                    )
                )
                if row["kind"] == "human_control"
                else row["model_keys"]
            )
            row["_search"] = " ".join(
                str(row.get(k) or "")
                for k in ["id", "source_id", "title", "source_author"]
            ).casefold()
            kinds[row["kind"]] += 1
            cohort_counts[row["cohort"]] += 1
            collection_counts[row["source_dataset"]] += 1
            splits[row["split"]] += 1
            for model in row["model_keys"]:
                models[model] += 1
        self.summary = {
            "id": self.id,
            "name": self.name,
            "records": len(self.rows),
            "mixed": kinds["mixed"],
            "controls": kinds["human_control"],
            "models": [
                {"value": m, "label": MODEL_LABELS.get(m, m), "count": n}
                for m, n in sorted(models.items())
            ],
            "cohorts": [
                {"value": c, "label": COHORT_LABELS.get(c, c), "count": n}
                for c, n in sorted(cohort_counts.items())
            ],
            "collections": [
                {"value": c, "label": COLLECTION_LABELS.get(c, c), "count": n}
                for c, n in sorted(collection_counts.items())
            ],
            "splits": [
                {"value": s, "label": s.title(), "count": n}
                for s, n in sorted(splits.items())
            ],
        }

    def query(self, filters):
        q = filters.get("q", "").strip().casefold()
        rows = []
        for row in self.rows:
            if q and q not in row["_search"]:
                continue
            if filters.get("model") and filters["model"] not in row["filter_models"]:
                continue
            if any(
                filters.get(k) and filters[k] != row[field]
                for k, field in [
                    ("kind", "kind"),
                    ("cohort", "cohort"),
                    ("collection", "source_dataset"),
                    ("split", "split"),
                ]
            ):
                continue
            rows.append(row)
        try:
            offset = max(0, int(filters.get("offset", "0")))
            limit = min(100, max(1, int(filters.get("limit", "50"))))
        except ValueError as e:
            raise ValueError("Invalid pagination") from e
        return {
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "rows": [
                {k: v for k, v in r.items() if k != "_search"}
                for r in rows[offset : offset + limit]
            ],
        }

    @lru_cache(maxsize=24)
    def raw_record(self, record_id):
        offset = self.offsets[record_id]
        with self.path.open("rb") as f:
            f.seek(offset)
            return json.loads(f.readline())

    def record(self, record_id):
        record = normalize_record(self.raw_record(record_id), self.cohorts)
        record["paired_records"] = [
            {"id": r["id"], "kind": r["kind"], "models": r["model_labels"]}
            for r in self.source_rows[record["source_id"]]
            if r["id"] != record_id
        ]
        return record


def make_server(indexes, host="127.0.0.1", port=8770):
    datasets = {index.id: index for index in indexes}
    assets = files("heterogeneous").joinpath("viewer_assets")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(
            self,
            status,
            body,
            content_type="application/json; charset=utf-8",
            download=None,
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            if download:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{download}"'
                )
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def json(self, status, value, download=None):
            self.send(
                status,
                json.dumps(value, ensure_ascii=False).encode(),
                download=download,
            )

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            url = urlsplit(self.path)
            args = {
                k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()
            }
            if url.path in ["/", "/app.js", "/style.css"]:
                name = {
                    "/": "index.html",
                    "/app.js": "app.js",
                    "/style.css": "style.css",
                }[url.path]
                mime = {
                    "index.html": "text/html; charset=utf-8",
                    "app.js": "text/javascript; charset=utf-8",
                    "style.css": "text/css; charset=utf-8",
                }[name]
                self.send(200, assets.joinpath(name).read_bytes(), mime)
                return
            if url.path == "/api/datasets":
                self.json(200, {"datasets": [i.summary for i in indexes]})
                return
            if url.path not in ["/api/records", "/api/record", "/api/export"]:
                self.json(404, {"error": "Not found"})
                return
            index = datasets.get(args.get("dataset", indexes[0].id))
            if not index:
                self.json(404, {"error": "Dataset not found"})
                return
            try:
                if url.path == "/api/records":
                    self.json(200, index.query(args))
                elif url.path == "/api/record":
                    self.json(200, index.record(args.get("id", "")))
                else:
                    record_id = args.get("id", "")
                    name = (
                        "document-"
                        + hashlib.sha256(record_id.encode()).hexdigest()[:12]
                        + ".json"
                    )
                    self.json(200, index.raw_record(record_id), name)
            except KeyError:
                self.json(404, {"error": "Document ID not found in this dataset"})
            except ValueError as e:
                self.json(400, {"error": str(e)})

    return ThreadingHTTPServer((host, port), Handler)


def serve(paths, host="127.0.0.1", port=8770, repo=None, revision="main"):
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    if repo:
        if paths:
            raise ValueError("Choose local datasets or --repo, not both")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ValueError(
                "Install huggingface_hub to use --repo, or provide a local release directory"
            ) from e
        paths = [
            Path(
                hf_hub_download(
                    repo, "records/all.jsonl", repo_type="dataset", revision=revision
                )
            )
        ]
        # Metadata is small; retain cohort mapping and release labels alongside the cached JSONL.
        for name in ["manifest/upstream-handoff.json", "manifest/release.json"]:
            hf_hub_download(repo, name, repo_type="dataset", revision=revision)
    indexes = [DatasetIndex(path) for path in (paths or [latest_local_dataset()])]
    if len({i.id for i in indexes}) != len(indexes):
        raise ValueError("The same dataset was supplied more than once")
    server = make_server(indexes, host, port)
    print(
        f"Viewer: http://{host}:{server.server_port}\nLoaded {sum(len(i.rows) for i in indexes):,} records across {len(indexes)} dataset(s). Press Ctrl+C to stop.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
