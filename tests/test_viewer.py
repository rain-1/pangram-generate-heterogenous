"""Viewer tests use synthetic fixtures, never human-corpus examples."""

import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from heterogeneous.viewer import DatasetIndex, make_server, normalize_record


def fixture(record_id="mixed", control=False):
    original = "🐧 Human.\n\nOriginal words.\n\nEnd."
    start = original.index("Original")
    end = start + len("Original words.")
    generated = "New AI wording 🦊."
    output = original if control else original[:start] + generated + original[end:]

    def span(a, b, sa, sb, label):
        return {
            "start": a,
            "end": b,
            "source_start": sa,
            "source_end": sb,
            "label": label,
            "author": {"id": "Synthetic fixture author"}
            if label == "human"
            else {
                "backend": "sol",
                "requested_model": "gpt-6.1-sol",
                "reported_model": None,
                "metadata": {"model_identity_status": "requested_only"},
            },
            "replacement_id": None if label == "human" else "replacement",
        }

    return {
        "id": record_id,
        "split": "test",
        "offset_unit": "unicode_codepoint",
        "end_exclusive": True,
        "text": output,
        "source": {
            "id": "source-🐧",
            "text": original,
            "title": "Synthetic fixture <script>",
            "author_id": "Synthetic fixture author",
            "dataset": "jmlr_pre2015",
            "human_verified": False,
        },
        "spans": [span(0, len(original), 0, len(original), "human")]
        if control
        else [
            span(0, start, 0, start, "human"),
            span(start, start + len(generated), start, end, "ai"),
            span(start + len(generated), len(output), end, len(original), "human"),
        ],
        "construction": {
            "kind": "human_control" if control else "mixed",
            "plan_id": "plan",
        },
        "replacements": []
        if control
        else [
            {
                "id": "replacement",
                "summary": "A synthetic brief.",
                "generation": {
                    "prompt": "Synthetic prompt </script>",
                    "quality": {"passed": True},
                },
            }
        ],
    }


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "records").mkdir()
        (self.root / "manifest").mkdir()
        self.path = self.root / "records/all.jsonl"
        self.raw = [fixture(), fixture("control", True)]
        self.path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.raw)
        )
        (self.root / "manifest/upstream-handoff.json").write_text(
            json.dumps({"cohorts": [{"plan_id": "plan", "name": "gpt_ml_papers"}]})
        )
        (self.root / "manifest/release.json").write_text(
            json.dumps({"release_version": "1.2.0"})
        )
        self.index = DatasetIndex(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_unicode_and_requested_only_identity_survive(self):
        record = self.index.record("mixed")
        self.assertEqual(record["cohort"], "gpt_ml_papers")
        self.assertEqual(record["model_labels"], ["GPT-6.1 Sol"])
        self.assertEqual(record["spans"][1]["model_identity_status"], "requested_only")
        self.assertIsNone(record["spans"][1]["reported_model"])
        for span in record["spans"]:
            if span["label"] == "human":
                self.assertEqual(
                    record["text"][span["start"] : span["end"]],
                    record["source_text"][span["source_start"] : span["source_end"]],
                )
        self.assertEqual(record["paired_records"][0]["id"], "control")
        self.assertEqual(self.index.raw_record("mixed"), self.raw[0])

    def test_controls_filter_by_paired_model_but_are_human(self):
        result = self.index.query(
            {
                "model": "gpt-6.1-sol",
                "kind": "human_control",
                "collection": "jmlr_pre2015",
                "q": "SOURCE-🐧",
            }
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["rows"][0]["id"], "control")
        self.assertEqual(result["rows"][0]["model_keys"], [])
        self.assertEqual(result["rows"][0]["ai_fraction"], 0)
        self.assertEqual(self.index.summary["models"][0]["count"], 1)
        self.assertEqual(self.index.query({"split": "train"})["total"], 0)

    def test_pagination_and_duplicate_ids(self):
        self.assertEqual(
            self.index.query({"offset": "1", "limit": "1"})["rows"][0]["id"], "control"
        )
        self.assertEqual(self.index.query({"offset": "10"})["rows"], [])
        with self.assertRaises(ValueError):
            self.index.query({"offset": "oops"})
        self.path.write_text(
            json.dumps(self.raw[0]) + "\n" + json.dumps(self.raw[0]) + "\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            DatasetIndex(self.path)

    def test_bad_boundaries_and_changed_human_text_rejected(self):
        for mutate in [
            "gap",
            "source_gap",
            "human_mismatch",
            "coverage",
            "offset_unit",
        ]:
            raw = copy.deepcopy(self.raw[0])
            if mutate == "gap":
                raw["spans"][1]["start"] += 1
            if mutate == "source_gap":
                raw["spans"][1]["source_start"] += 1
            if mutate == "human_mismatch":
                raw["text"] = "X" + raw["text"][1:]
            if mutate == "coverage":
                raw["spans"].pop()
            if mutate == "offset_unit":
                raw["offset_unit"] = "utf16"
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                normalize_record(raw, {})

    def test_flat_export_supported(self):
        raw = self.raw[0]
        flat = {
            "id": raw["id"],
            "split": raw["split"],
            "kind": "mixed",
            "cohort": "gpt_ml_papers",
            "text": raw["text"],
            "source_text": raw["source"]["text"],
            "source_id": raw["source"]["id"],
            "source_title": raw["source"]["title"],
            "source_dataset": "jmlr_pre2015",
            "spans": [],
        }
        for span in raw["spans"]:
            converted = {k: v for k, v in span.items() if k != "author"}
            converted.update(
                {k: v for k, v in span["author"].items() if k != "metadata"}
            )
            converted["author_id"] = span["author"].get("id")
            converted["model_identity_status"] = (
                span["author"].get("metadata", {}).get("model_identity_status")
            )
            flat["spans"].append(converted)
        normalized = normalize_record(flat, {})
        self.assertEqual(normalized["source_text"], raw["source"]["text"])
        self.assertEqual(normalized["model_keys"], ["gpt-6.1-sol"])

    def test_hub_snapshot_symlink_retains_adjacent_metadata(self):
        blob = self.root / "blob"
        self.path.rename(blob)
        self.path.symlink_to(blob)
        index = DatasetIndex(self.path)
        self.assertEqual(index.name, "Human / AI Spans · v1.2.0")
        self.assertEqual(index.record("mixed")["cohort"], "gpt_ml_papers")

    def test_http_routes_export_and_no_filesystem_access(self):
        server = make_server([self.index], port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path in [
                "/",
                "/style.css",
                "/app.js",
                "/api/datasets",
                "/api/records?kind=mixed",
                "/api/record?id=mixed",
            ]:
                with self.subTest(path=path), urlopen(base + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(
                        "frame-ancestors 'none'",
                        response.headers["Content-Security-Policy"],
                    )
            with urlopen(base + "/api/export?id=mixed") as response:
                self.assertEqual(json.load(response), self.raw[0])
                self.assertIn("attachment;", response.headers["Content-Disposition"])
            with urlopen(
                Request(base + "/api/record?id=mixed", method="HEAD")
            ) as response:
                self.assertEqual(response.read(), b"")
            for path, status in [
                ("/../pyproject.toml", 404),
                ("/api/record?id=missing", 404),
                ("/api/records?dataset=unknown", 404),
                ("/api/records?limit=oops", 400),
            ]:
                with self.subTest(path=path), self.assertRaises(HTTPError) as context:
                    urlopen(base + path)
                self.assertEqual(context.exception.code, status)
            with self.assertRaises(HTTPError) as context:
                urlopen(Request(base + "/api/record", data=b"{}", method="POST"))
            self.assertEqual(context.exception.code, 501)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
