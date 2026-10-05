"""Generate the historical-paper probe/full run with resumable stage caches."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading

from heterogeneous.core import digest, read_jsonl, write_json, write_jsonl
from heterogeneous.html import render_html
from heterogeneous.pipeline import Runner, load_plan


def assemble(run):
    plan = load_plan(run)
    report = Runner(run, plan).assemble(allow_partial=True)
    records = read_jsonl(run / 'dataset.jsonl')
    mixed = [r for r in records if r['construction']['kind'] == 'mixed']
    controls = [r for r in records if r['construction']['kind'] == 'human_control']
    write_jsonl(run / 'mixed.jsonl', mixed)
    write_jsonl(run / 'human-controls.jsonl', controls)
    if mixed:
        render_html(mixed, run / 'review.html', len(mixed))
    selected = set(json.loads((run / 'verification-selection.json').read_text())['variants'])
    ids = {v['source_id'] for v in plan['variants'] if v['id'] in selected}
    probe = [r for r in mixed if r['source']['id'] in ids]
    if probe:
        render_html(probe, run / 'model-probe-review.html', len(probe))
    return report


def source_check(run):
    plan = load_plan(run)
    papers = set()
    references = set()
    bibliographic_ids = set()
    for source in plan['sources']:
        meta = source['upstream_metadata']
        path = Path(meta['pdf_path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['pdf_sha256']
        assert hashlib.sha256(path.with_suffix('.xml').read_bytes()).hexdigest() == meta['layout_xml_sha256']
        index = path.parent / f"volume-{meta['volume']}.html"
        assert hashlib.sha256(index.read_bytes()).hexdigest() == meta['index_sha256']
        assert meta['year'] == meta['pdf_publication_year'] < 2015
        assert source['group_id'] not in papers
        papers.add(source['group_id'])
        assert source['reference'] not in references
        references.add(source['reference'])
        bibliographic_id = (meta['title'], meta['authors'], meta['year'])
        assert bibliographic_id not in bibliographic_ids
        bibliographic_ids.add(bibliographic_id)
        assert source['text'] == '\n\n'.join(p['text'] for p in source['preprocessing']['selected_paragraphs'])
        assert all(p['eligible_prose'] for p in source['preprocessing']['selected_paragraphs'])
        indices = source['paragraph_indices']
        assert indices == list(range(indices[0], indices[-1]+1))
        assert source['preprocessing']['no_model_used'] is True
    report = {'plan_id': plan['id'], 'sources': len(papers), 'structural_checks': 'pass',
              'checks': ['Raw PDF and layout XML integrity', 'Journal-index integrity',
                         'Matching index/PDF publication years before 2015',
                         'Unique paper, reference and bibliographic identity per source', 'Continuous eligible paragraph indices',
                         'Exact extracted-paragraph/source equality', 'No model used for extraction'],
              'visual_review': json.loads((run / 'source-visual-review.json').read_text()),
              'human_origin': 'Strong historical evidence; candidate status preserved, not certified.'}
    assert report['visual_review']['decision'] == 'proceed'
    assert report['visual_review']['plan_id'] == plan['id']
    write_json(run / 'source-check.json', report)
    return report


def process(run, scope, jobs):
    plan = load_plan(run)
    source_check(run)
    if scope == 'all':
        probe = json.loads((run / 'model-probe-audit.json').read_text())
        assert probe['plan_id'] == plan['id'] and probe['decision'] == 'proceed'
    selected = set(json.loads((run / 'verification-selection.json').read_text())['variants'])
    assert json.loads((run / 'verification-selection.json').read_text())['plan_id'] == plan['id']
    variants = [v for v in plan['variants'] if scope == 'all' or v['id'] in selected]
    runner = Runner(run, plan)
    runner.semaphores[digest(plan['config']['summarizer'])] = threading.Semaphore(8)
    errors = []
    def work(v):
        for attempt in range(3):
            try:
                for block in v['blocks']:
                    runner.summary(v, block)
                for backend in runner.variant_backends(v):
                    for block in v['blocks']:
                        runner.generation(v, block, backend)
                return v['id']
            except Exception:
                if attempt == 2:
                    raise
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(work, v): v for v in variants}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                errors.append({'variant': futures[future]['id'], 'error': str(error)})
            progress = {'scope': scope, 'completed_tasks': i, 'total': len(variants), 'errors': len(errors)}
            write_json(run / f'{scope}-progress.json', progress)
            write_json(run / f'{scope}-errors.json', errors)
            print(json.dumps(progress), flush=True)
    assemble(run)
    if errors:
        raise ValueError(f'{len(errors)} variants need retries; successful stages retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scope', choices=['probe', 'all', 'assemble'], default='probe')
    parser.add_argument('--jobs', type=int, default=20)
    args = parser.parse_args()
    if args.scope == 'assemble':
        assemble(args.run)
    else:
        process(args.run, args.scope, args.jobs)
