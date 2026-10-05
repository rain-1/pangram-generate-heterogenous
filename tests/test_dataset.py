"""All prose here is synthetic software-test data, not a human corpus."""
import copy
import json
import random
import tempfile
import threading
import unittest
from pathlib import Path

from heterogeneous.config import DEFAULT_DATASET, backend_config, load_config
from heterogeneous.core import (assemble, copy_fraction, digest, normalized_hash,
                                paragraph_spans, read_jsonl, select_blocks,
                                sentence_spans, text_hash, token_labels,
                                validate_record, write_jsonl)
from heterogeneous.evaluate import evaluate, matched_boundaries
from heterogeneous.pipeline import (Runner, generate_prompt, generation_quality,
                                    load_plan, load_sources, make_plan, save_plan,
                                    split_sources, summarize_prompt, summary_quality, validate_dataset)
from heterogeneous.providers import ProviderError, complete as provider_complete


def source(text="A non-AI fixture 🐧 for tests only. Another section remains."):
    return {"id": "fixture", "group_id": "fixture", "text": text, "human_verified": True,
            "reference": "test://synthetic-fixture", "provenance": "Synthetic unit-test fixture; never train on this",
            "sha256": text_hash(text), "normalized_sha256": normalized_hash(text)}


def response(text, prompt="test", model="fixture-model", kind="openai-compatible", backend="test"):
    return {"text": text, "prompt": prompt, "quality": {}, "quality_attempt": 1,
            "rejected_attempts": [], "cache_key": "a" * 64,
            "provenance": {"backend": backend, "kind": kind, "requested_model": model,
            "reported_model": model, "family": "fixture", "revision": None, "provider": None,
            "request_id": "fixture-request", "sampling": {}, "generated_at": "2026-10-03T00:00:00+00:00",
            "finish_reason": "stop", "prompt_sha256": text_hash(prompt), "usage": None, "metadata": {}}}


def replacement(start, end, text, name="replacement"):
    return {"id": name, "source_start": start, "source_end": end, "generated_text": text,
            "source_units": 1, "source_paragraphs": 1, "summary": "a brief",
            "generation": response(text), "summarization": response("a brief")}


def settings(**overrides):
    return {**copy.deepcopy(DEFAULT_DATASET), "block_sizes": [2], "variants_per_source": 1,
            "min_blocks": 1, "max_blocks": 1, **overrides}


def config():
    summary = backend_config({"kind": "openai-compatible", "model": "fixture-summary", "name": "brief"})
    generator = backend_config({"kind": "openai-compatible", "model": "fixture-generation", "name": "gen"})
    return {"dataset": settings(), "summarizer": summary, "generators": [generator]}


def long_source():
    paragraphs = [" ".join(
        " ".join(f"source{i}sentence{s}term{j}" for j in range(15)) + "."
        for s in range(2)) for i in range(6)]
    return source("\n\n".join(paragraphs))


class SpliceTests(unittest.TestCase):
    def test_positive_context_keeps_complete_nearest_paragraphs(self):
        original=source('Earlier unrelated paragraph.\n\nThe whole preceding paragraph exceeds the tiny budget.\n\nReplace this selected passage.\n\nThe whole following paragraph exceeds the tiny budget.\n\nLater unrelated paragraph.')
        start=original['text'].index('Replace');end=start+len('Replace this selected passage.')
        block={'start':start,'end':end,'paragraphs':1}
        prompt=generate_prompt(original,[block],block,'A short brief.',settings(context_chars=7))
        data=json.loads(prompt.split('\n')[-1])
        self.assertEqual(data['preceding_context'],'The whole preceding paragraph exceeds the tiny budget.\n\n')
        self.assertEqual(data['following_context'],'\n\nThe whole following paragraph exceeds the tiny budget.')
        self.assertNotIn('Earlier unrelated',prompt);self.assertNotIn('Later unrelated',prompt)
    def test_browser_backend_never_silently_falls_back_to_http_or_cli(self):
        cfg = backend_config({'name': 'opus3', 'kind': 'claude-web', 'model': 'opus-3'})
        self.assertNotIn('base_url', cfg)
        with self.assertRaisesRegex(ProviderError, 'browser completion import'):
            provider_complete(cfg, 'Test prompt')
        with self.assertRaisesRegex(ValueError, 'sampling parameters'):
            backend_config({**cfg, 'temperature': .8})
    def test_per_source_generators_do_not_reuse_a_source_for_other_models(self):
        cfg = config()
        cfg['generators'].append(backend_config({'kind':'openai-compatible','model':'other-model','name':'other'}))
        a={**long_source(),'id':'a','generator_names':['gen']}
        b={**long_source(),'id':'b','generator_names':['other']}
        calls=[]
        def complete(backend,prompt):
            calls.append(backend['name'])
            text=('A concise description. Another key idea.' if backend['name']=='brief' else
                  'Fresh prose develops an idea.\n\nA second paragraph extends the discussion.')
            return response(text,prompt,backend['model'],backend=backend['name'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_jsonl(root/'input.jsonl',[a,b]);plan=make_plan(root/'input.jsonl',cfg)
            runner=Runner(root,plan,complete);runner.stage('summarize',1);runner.stage('generate',1);runner.assemble()
            mixed=[r for r in read_jsonl(root/'dataset.jsonl') if r['construction']['kind']=='mixed']
            self.assertEqual(len(mixed),2)
            self.assertEqual(calls.count('gen'),1);self.assertEqual(calls.count('other'),1)
            for r in mixed:
                expected='gen' if r['source']['id']=='a' else 'other'
                self.assertEqual(r['replacements'][0]['generation']['provenance']['backend'],expected)
    def test_prose_selection_keeps_headings_out_of_ai_blocks(self):
        text = 'HEADING\n\nFirst part. More detail.\n\nSecond part. More detail.\n\nFOOTER'
        blocks = select_blocks(text, settings(require_prose_paragraphs=True, max_replaced_fraction=.9), 'fixture')
        self.assertEqual(len(blocks), 1)
        self.assertEqual(text[blocks[0]['start']:blocks[0]['end']], 'First part. More detail.\n\nSecond part. More detail.')
    def test_candidate_origin_requires_explicit_opt_in(self):
        candidate = {**source(), "human_verified": False, "authorship": "human_candidate",
                     "human_origin_basis": {"kind": "documentary_source_fixture"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.jsonl"
            write_jsonl(path, [candidate])
            with self.assertRaisesRegex(ValueError, "human_verified"):
                load_sources(path)
            loaded = load_sources(path, allow_human_candidates=True)
            self.assertIs(loaded[0]["human_verified"], False)
            del candidate["human_origin_basis"]
            write_jsonl(path, [candidate])
            with self.assertRaisesRegex(ValueError, "human_verified"):
                load_sources(path, allow_human_candidates=True)

    def test_candidate_record_preserves_uncertainty(self):
        candidate = {**source(), "human_verified": False, "authorship": "human_candidate",
                     "binary_target": None, "human_origin_basis": {"kind": "documentary_source_fixture"}}
        with self.assertRaisesRegex(ValueError, "provenance"):
            assemble(candidate, [], "candidate", "train", {})
        record = assemble(candidate, [], "candidate", "train", {"source_origin_policy": "documented_candidates"})
        self.assertIs(record["source"]["human_verified"], False)
        self.assertIsNone(record["source"]["binary_target"])
        self.assertIn("candidate", record["label_definition"]["human"])
        record["label_definition"]["human"] = "verified human"
        with self.assertRaisesRegex(ValueError, "disclose"):
            validate_record(record)

    def test_runner_candidate_controls_keep_source_policy(self):
        candidate = {**long_source(), "human_verified": False, "authorship": "human_candidate",
                     "human_origin_basis": {"kind": "documentary_source_fixture"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "input.jsonl", [candidate])
            cfg = config()
            cfg["dataset"]["allow_human_candidates"] = True
            plan = make_plan(root / "input.jsonl", cfg)
            Runner(root, plan).assemble(allow_partial=True)
            record = read_jsonl(root / "dataset.jsonl")[0]
            self.assertEqual(record["construction"]["source_origin_policy"], "documented_candidates")
            self.assertIs(record["source"]["human_verified"], False)

    def record(self):
        original = source("Préface 🐧.\r\n\r\nReplace this sentence.\n\nEnding remains e\u0301.")
        start = original["text"].index("Replace")
        end = start + len("Replace this sentence.")
        return assemble(original, [replacement(start, end, "New prose with 🚀 and changed length.")], "example", "train", {"kind": "mixed", "plan_id": "a" * 64})

    def test_exact_unicode_and_whitespace(self):
        record = self.record()
        spans = record["spans"]
        self.assertEqual(len(spans), 3)
        self.assertEqual(record["text"][spans[1]["start"]:spans[1]["end"]], "New prose with 🚀 and changed length.")
        self.assertEqual(spans[0]["author"], {"type": "non_ai", "id": None})
        self.assertEqual(spans[1]["author"]["requested_model"], "fixture-model")
        self.assertEqual(record["text"], "Préface 🐧.\r\n\r\nNew prose with 🚀 and changed length.\n\nEnding remains e\u0301.")
        validate_record(record)

    def test_offsets_survive_arbitrary_length_changes(self):
        rng = random.Random(42)
        for _ in range(100):
            text = " ".join(rng.choice(["a", "🐧", "é", "e\u0301", "🚀", "z"]) for _ in range(30))
            original = source(text)
            start, end = sorted(rng.sample(range(len(text) + 1), 2))
            new_text = "replacement " * rng.randint(1, 20)
            record = assemble(original, [replacement(start, end, new_text)], "fuzz", "train", {"kind": "mixed", "plan_id": "a" * 64})
            self.assertEqual(record["text"], text[:start] + new_text + text[end:])

    def test_adjacent_different_ai_authors(self):
        original = source("abcdef")
        one, two = replacement(0, 3, "AAA", "one"), replacement(3, 6, "BBBBB", "two")
        two["generation"]["provenance"]["requested_model"] = "second-model"
        record = assemble(original, [one, two], "id", "test", {})
        self.assertEqual(len(record["spans"]), 2)
        self.assertEqual(record["ai_fraction_chars"], 1)

    def test_reject_overlaps(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            assemble(source("abcdef"), [replacement(0, 4, "AI", "one"), replacement(2, 5, "AI", "two")], "id", "train", {})

    def test_human_control_and_corruption(self):
        record = assemble(source(), [], "id", "train", {})
        self.assertEqual(record["ai_fraction_chars"], 0)
        self.assertEqual(len(record["spans"]), 1)
        for mutate in (
            lambda r: r["spans"][0].update(start=1),
            lambda r: r.update(text=r["text"] + "a"),
            lambda r: r["source"].update(text="changed"),
            lambda r: r.update(ai_fraction_chars=0.1),
        ):
            broken = copy.deepcopy(record)
            mutate(broken)
            with self.assertRaises(ValueError):
                validate_record(broken)

    def test_token_mapping_boundary_ties_and_specials(self):
        spans = [{"start": 0, "end": 3, "label": "human"}, {"start": 3, "end": 8, "label": "ai"}]
        self.assertEqual(token_labels(spans, [(0, 0), (0, 3), (2, 4), (4, 7), (0, 0)]), [-100, 0, -100, 1, -100])


class SelectionTests(unittest.TestCase):
    def test_sentence_offsets_and_abbreviations(self):
        text = 'Dr. Jones left at 2.5. "Hello there!"\n\nLast line'
        spans = sentence_spans(text)
        self.assertEqual([text[a:b] for a, b in spans], ['Dr. Jones left at 2.5.', '"Hello there!"', 'Last line'])

    def test_paragraph_offsets(self):
        text = "  One.\r\n \r\nTwo.\n\n\n  Three.  "
        self.assertEqual([text[a:b] for a, b in paragraph_spans(text)], ["One.", "Two.", "Three."])

    def test_seeded_nonoverlap_and_fraction(self):
        text = long_source()["text"]
        opts = settings(max_blocks=3)
        blocks = select_blocks(text, opts, "same")
        self.assertEqual(blocks, select_blocks(text, opts, "same"))
        self.assertTrue(blocks)
        self.assertLessEqual(sum(b["end"] - b["start"] for b in blocks), len(text) * opts["max_replaced_fraction"])
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(blocks, blocks[1:])))

    def test_alternating(self):
        text = long_source()["text"]
        units = paragraph_spans(text)
        blocks = select_blocks(text, settings(mode="alternating"), "same")
        self.assertEqual([b["start"] for b in blocks], [units[2][0]])
        self.assertEqual(blocks[0]["paragraphs"], 2)
        self.assertEqual(blocks[0]["sentences"], 4)

    def test_sentence_mode_still_uses_sentence_blocks(self):
        text = long_source()["text"]
        blocks = select_blocks(text, settings(unit="sentence", block_sizes=[4], mode="alternating"), "same")
        self.assertEqual(blocks[0]["start"], sentence_spans(text)[4][0])
        self.assertEqual(blocks[0]["sentences"], 4)

    def test_blocks_must_have_more_sentences_than_the_brief(self):
        text = "One sentence.\n\nAnother sentence.\n\nLast sentence."
        self.assertEqual(select_blocks(text, settings(block_sizes=[1]), "same"), [])

    def test_generator_never_receives_replaced_text(self):
        original = source("prefix visible. SECRET ORIGINAL ONE. human neighbour. SECRET ORIGINAL TWO. trailing text.")
        blocks = []
        for secret in ("SECRET ORIGINAL ONE.", "SECRET ORIGINAL TWO."):
            start = original["text"].index(secret)
            blocks.append({"start": start, "end": start + len(secret), "paragraphs": 1})
        for block in blocks:
            prompt = generate_prompt(original, blocks, block, "a brief", settings(context_chars=999))
            self.assertNotIn("SECRET", prompt)
            self.assertIn("human neighbour", prompt)


class SplitTests(unittest.TestCase):
    def test_transitive_group_and_duplicate_isolation(self):
        a = source("Same text.")
        b = {**source("Different text."), "id": "b"}
        c = {**source(" SAME   TEXT. "), "id": "c", "group_id": "other"}
        d = {**source("fourth"), "id": "d", "group_id": "other"}
        assignments = split_sources([a, b, c, d], settings())
        self.assertEqual(len(set(assignments.values())), 1)

    def test_conflicting_explicit_splits_rejected(self):
        a = {**source(), "split": "train"}
        b = {**a, "id": "second", "split": "test"}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            split_sources([a, b], settings())

    def test_dataset_leakage_rejected(self):
        a = assemble(source(), [], "a", "train", {})
        b = assemble(source(), [], "b", "test", {})
        with self.assertRaisesRegex(ValueError, "leaked"):
            validate_dataset([a, b])

    def test_unverified_sources_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            write_jsonl(path, [{**source(), "human_verified": False}])
            with self.assertRaisesRegex(ValueError, "human_verified"):
                load_sources(path)

    def test_invalid_metadata_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            write_jsonl(path, [{**source(), "author_id": ["not", "an", "id"]}])
            with self.assertRaisesRegex(ValueError, "author_id"):
                load_sources(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "input.jsonl"
        write_jsonl(self.input, [long_source()])
        self.config = config()
        self.plan = make_plan(self.input, self.config)
        save_plan(self.plan, self.root)
        self.calls = []
        self.call_lock = threading.Lock()

    def tearDown(self):
        self.temp.cleanup()

    def fake(self, backend, prompt):
        with self.call_lock:
            self.calls.append(backend["name"])
        if backend["name"] == "brief":
            text = "A proposal changes local travel. Residents consider its consequences."
        else:
            count = json.loads(prompt.rsplit("\n", 1)[1])["target_paragraphs"]
            text = "\n\n".join(f"New events develop in section {i}. Further details illuminate their consequences." for i in range(count))
        return response(text, prompt, backend["model"], backend["kind"], backend["name"])

    def test_end_to_end_resume_and_no_calls_in_assembly(self):
        runner = Runner(self.root, self.plan, self.fake)
        runner.stage("summarize", 2)
        runner.stage("generate", 2)
        before = len(self.calls)
        report = runner.assemble()
        self.assertEqual(report["records"], 2)
        self.assertEqual(len(self.calls), before)
        runner.stage("summarize", 2)
        runner.stage("generate", 2)
        self.assertEqual(len(self.calls), before)
        records = read_jsonl(self.root / "dataset.jsonl")
        self.assertEqual({r["construction"]["kind"] for r in records}, {"human_control", "mixed"})
        self.assertEqual(validate_dataset(records)["sources"], 1)

    def test_detector_gate_accepts_variable_layout_with_valid_labels(self):
        gate={"plan_id":self.plan["id"],"decision":"proceed",
              "evaluation_policy":"detector_task_fitness","paragraph_count_policy":"soft_target"}
        (self.root/"quality-gate.json").write_text(json.dumps(gate))
        def varied(backend,prompt):
            if backend["name"]=="brief":return self.fake(backend,prompt)
            return response("A new development reshapes the situation. Everyone considers an unexpected possibility.",prompt,backend["model"],backend["kind"],backend["name"])
        runner=Runner(self.root,self.plan,varied)
        self.assertTrue(any(b["paragraphs"]>1 for v in self.plan["variants"] for b in v["blocks"]))
        runner.stage("summarize",1);runner.stage("generate",1);runner.assemble()
        records=read_jsonl(self.root/"dataset.jsonl")
        self.assertEqual(validate_dataset(records)["records"],2)
        mixed=next(r for r in records if r["construction"]["kind"]=="mixed")
        self.assertTrue(any(s["label"]=="ai" for s in mixed["spans"]))
        self.assertEqual(mixed["source"]["text"],long_source()["text"])
        self.assertTrue(all(b["generation"]["quality"]["paragraphs"]==1 for b in mixed["replacements"]))

    def test_detector_gate_cannot_relax_another_plan(self):
        (self.root/"quality-gate.json").write_text(json.dumps({
            "plan_id":"another-plan","decision":"proceed","evaluation_policy":"detector_task_fitness",
            "paragraph_count_policy":"soft_target"}))
        runner=Runner(self.root,self.plan,self.fake)
        with self.assertRaisesRegex(ValueError,"paragraphs"):
            generation_quality("A newly generated paragraph.",{"paragraphs":2},long_source(),runner.generation_quality_settings)

    def test_missing_variant_not_relabelled(self):
        runner = Runner(self.root, self.plan, self.fake)
        with self.assertRaisesRegex(ValueError, "missing"):
            runner.assemble()
        self.assertFalse((self.root / "dataset.jsonl").exists())
        report = runner.assemble(allow_partial=True)
        self.assertTrue(report["partial"])
        self.assertEqual(report["records"], 1)
        self.assertEqual(self.calls, [])

    def test_generation_requires_briefs_without_calls(self):
        runner = Runner(self.root, self.plan, self.fake)
        variant = self.plan["variants"][0]
        with self.assertRaisesRegex(ValueError, "missing brief"):
            runner.generation(variant, variant["blocks"][0], self.config["generators"][0])
        self.assertEqual(self.calls, [])

    def test_plan_change_refused(self):
        changed = make_plan(self.input, {**self.config, "dataset": settings(seed=99)})
        with self.assertRaisesRegex(ValueError, "different plan"):
            save_plan(changed, self.root)

    def test_plan_integrity(self):
        self.assertEqual(load_plan(self.root), self.plan)
        changed = copy.deepcopy(self.plan)
        changed["config"]["dataset"]["seed"] += 1
        (self.root / "plan.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "modified"):
            load_plan(self.root)

    def test_bounded_quality_retries(self):
        attempts = []
        def bad(backend, prompt):
            attempts.append(prompt)
            return response("too many words " * 10, prompt, backend["model"])
        runner = Runner(self.root, self.plan, bad)
        variant = self.plan["variants"][0]
        with self.assertRaisesRegex(ValueError, "quality failed"):
            runner.summary(variant, variant["blocks"][0])
        self.assertEqual(len(attempts), self.config["dataset"]["quality_attempts"])
        self.assertIn("Previous attempt failed", attempts[1])

    def test_example_config_is_valid(self):
        loaded = load_config(Path(__file__).resolve().parents[1] / "config.example.toml")
        self.assertEqual(len(loaded["generators"]), 4)

    def test_cannot_override_http_model(self):
        with self.assertRaisesRegex(ValueError, "override"):
            backend_config({"name": "bad", "kind": "openrouter", "model": "x", "extra_body": {"model": "y"}})

    def test_cli_args_cannot_override_model_or_auth(self):
        for args in (["--model", "wrong"], ["-c", 'model_provider="wrong"'], ["--dangerously-bypass-approvals-and-sandbox"]):
            with self.assertRaises(ValueError):
                backend_config({"name": "bad", "kind": "codex", "model": "x", "cli_args": args})
        valid = backend_config({"name": "ok", "kind": "codex", "model": "x", "cli_args": ["-c", 'model_reasoning_effort="low"']})
        self.assertIn("cli_args", valid)


class QualityTests(unittest.TestCase):
    def test_copy_coverage(self):
        original = " ".join(f"word{i}" for i in range(20))
        generated = " ".join(f"word{i}" for i in range(10)) + " novel new phrasing"
        self.assertAlmostEqual(copy_fraction(generated, original, 8), 10 / 13)

    def test_excessive_copy_rejected(self):
        original = long_source()
        text = original["text"].split("\n\n")[0]
        with self.assertRaisesRegex(ValueError, "copies"):
            generation_quality(text, {"paragraphs": 1}, original, settings())

    def test_summary_sentence_limits_enforced(self):
        block = {"characters": 1000}
        for text in ("An overly terse description.", "One sentence. Another sentence. A third sentence. A fourth sentence."):
            with self.assertRaisesRegex(ValueError, "description must contain"):
                summary_quality(text, block, long_source(), settings())

    def test_multi_sentence_brief_is_accepted(self):
        text = "The village considers changing its transport system. Residents disagree about how the change would affect them."
        quality = summary_quality(text, {"characters": 1000}, long_source(), settings())
        self.assertEqual(quality["sentences"], 2)
        self.assertNotIn("words", quality)

    def test_fragment_with_closing_quote_is_not_a_complete_sentence(self):
        with self.assertRaisesRegex(ValueError, "punctuation"):
            summary_quality('A complete sentence. An unfinished fragment”', {"characters": 1000}, long_source(), settings())

    def test_old_prompt_plan_cannot_reuse_cached_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {"schema_version": "1.0", "prompt_version": "1"}
            old = {"id": digest(old), **old}
            Path(directory, "plan.json").write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "incompatible prompt version"):
                load_plan(Path(directory))

    def test_summary_must_be_shorter_than_source(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            summary_quality("Two sentences. Several extra details.", {"characters": 10}, long_source(), settings())

    def test_generation_paragraph_count_enforced(self):
        with self.assertRaisesRegex(ValueError, "2 paragraphs"):
            generation_quality("Only one paragraph is generated.", {"paragraphs": 2}, long_source(), settings())

    def test_prompts_use_sentences_and_paragraphs(self):
        summary = summarize_prompt("Source paragraphs.", 2, 3)
        self.assertIn("2 to 3 concise, complete sentences", summary)
        self.assertNotIn(" words", summary)
        block = {"start": 0, "end": 10, "paragraphs": 3}
        prompt = generate_prompt(long_source(), [block], block, "Some description. More detail.", settings())
        self.assertIn("3 developed paragraphs", prompt)
        self.assertIn('"target_paragraphs": 3', prompt)
        self.assertNotIn("target_words", prompt)


class EvaluationTests(unittest.TestCase):
    def test_exact_and_boundary_metrics(self):
        original = source("abcdefghij")
        record = assemble(original, [replacement(3, 7, "WXYZ")], "gold", "test", {})
        predictions = [{"id": record["id"], "text_sha256": record["text_sha256"], "spans": [
            {"start": 0, "end": 4, "label": "human"}, {"start": 4, "end": 8, "label": "ai"}, {"start": 8, "end": 10, "label": "human"}]}]
        report = evaluate([record], predictions, 1)
        self.assertEqual(report["character_metrics"]["tp"], 3)
        self.assertEqual(report["character_metrics"]["fp"], 1)
        self.assertEqual(report["character_metrics"]["fn"], 1)
        self.assertEqual(report["character_metrics"]["accuracy"], 0.8)
        self.assertEqual(report["boundary_metrics"]["f1"], 1)
        self.assertEqual(report["document_ai_fraction_mae"], 0)

    def test_boundary_matches_are_one_to_one(self):
        self.assertEqual(matched_boundaries([10], [9, 11], 2), 1)

    def test_reject_missing_or_wrong_text(self):
        record = assemble(source(), [], "id", "test", {})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate([record], [])
        with self.assertRaisesRegex(ValueError, "hash"):
            evaluate([record], [{"id": "id", "text_sha256": "bad", "spans": record["spans"]}])


if __name__ == "__main__":
    unittest.main()
