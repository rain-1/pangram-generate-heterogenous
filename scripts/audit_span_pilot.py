"""Audit detector labels and provenance without judging factual fidelity."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import median

import jsonschema

from heterogeneous.core import read_jsonl, text_hash, write_json
from heterogeneous.pipeline import Runner, load_plan, validate_dataset


def audit(run: Path, require_complete: bool = False):
    def no_generation(*args):
        raise AssertionError('Auditing must never request model completions')

    plan = load_plan(run)
    runner = Runner(run, plan, completion=no_generation)
    records = read_jsonl(run / 'dataset.jsonl')
    report = validate_dataset(records)
    schema = json.loads(Path('dataset.schema.json').read_text())
    validator = jsonschema.Draft202012Validator(schema)
    mixed = [r for r in records if r['construction']['kind'] == 'mixed']
    controls = [r for r in records if r['construction']['kind'] == 'human_control']
    variants = {v['source_id']: v for v in plan['variants']}
    assert len(variants) == len(plan['variants']) == len(plan['sources'])
    expected_counts = Counter(v['generator_names'][0] for v in plan['variants'])
    assert len({r['source']['id'] for r in mixed}) == len(mixed)
    assert {r['source']['id'] for r in controls} == set(variants)
    counts = Counter()
    ratios, compressions, flags = [], [], []
    # These are review flags, not an automatic claim that dialogue is a refusal.
    metatext = re.compile(r"^(?:I'm sorry[, ]+but|I (?:cannot|can't) (?:assist|help|provide|generate)|(?:Certainly|Sure)[,!]|Here(?:'s| is) (?:the|a) (?:rewritten|generated|expanded)|As an AI)", re.I)
    for r in records:
        validator.validate(r)
        assert r['source']['text'] == runner.sources[r['source']['id']]['text']
        assert r['source']['human_verified'] is False
        assert r['source']['binary_target'] is None
        if r['construction']['kind'] != 'mixed':
            continue
        v = variants[r['source']['id']]
        backend, = runner.variant_backends(v)
        counts[backend['name']] += 1
        assert [b['id'] for b in r['replacements']] == [b['id'] for b in v['blocks']]
        for b, expected in zip(r['replacements'], v['blocks']):
            assert (b['source_start'], b['source_end']) == (expected['start'], expected['end'])
            brief = runner.summary(v, expected)
            generation = runner.generation(v, expected, backend)
            assert brief == b['summarization'] and generation == b['generation']
            assert generation['text'] == b['generated_text']
            provenance = generation['provenance']
            assert text_hash(generation['prompt']) == provenance['prompt_sha256']
            assert provenance['requested_model'] == backend['model']
            if backend['kind'] == 'claude-web':
                assert provenance['reported_model'] == 'Opus 3'
                assert provenance['metadata']['model_identity_status'] == 'ui_label_only'
                assert provenance['metadata']['exact_checkpoint_unreported'] is True
            else:
                assert provenance['reported_model'] == backend['model']
                assert provenance['metadata']['model_identity_status'] == 'reported'
            ratios.append(len(b['generated_text']) / b['source_characters'])
            compressions.append(len(b['summary']) / b['source_characters'])
            if metatext.search(b['generated_text'].lstrip()):
                flags.append({'record_id': r['id'], 'block_id': b['id'], 'model': backend['name'],
                              'reason': 'possible refusal or introductory instruction text',
                              'opening': b['generated_text'][:300]})
    complete = counts == expected_counts
    if require_complete:
        assert complete, dict(counts)
    finished = {r['source']['id'] for r in mixed}
    report.update({
        'plan_id': plan['id'], 'audited_at': datetime.now(timezone.utc).isoformat(),
        'evaluation_policy': 'detector_task_fitness', 'mixed_documents': len(mixed),
        'human_controls': len(controls), 'complete': complete, 'models': dict(counts),
        'expected_models': dict(expected_counts),
        'missing_variants': [v['id'] for v in plan['variants'] if v['source_id'] not in finished],
        'integrity_checks': 'pass', 'prose_review_flags': flags,
        'replacement_length_ratio': {'min':min(ratios), 'median':median(ratios), 'max':max(ratios)} if ratios else None,
        'brief_compression_ratio': {'min':min(compressions), 'median':median(compressions), 'max':max(compressions)} if compressions else None,
        'limitations': [
            'Source spans are documented human-origin candidates, not certified human authorship.',
            'Browser Opus 3 identity is the selected UI label; its exact checkpoint is unreported.',
            'Factual drift and length differences are not rejection criteria for this detector dataset.',
            'Pattern-based prose flags require inspection and are not semantic quality judgments.',
        ],
    })
    write_json(run / 'detector-quality-audit.json', report)
    print(json.dumps({k:report[k] for k in ['mixed_documents','human_controls','complete','models','integrity_checks','prose_review_flags']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    audit(args.run, args.require_complete)
