"""Resume the pilot and exchange auditable requests with a browser-only model.

Browser interaction is deliberately external: this script never calls private
claude.ai endpoints or pretends a CLI result came from the browser.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from heterogeneous.core import digest, read_jsonl, text_hash, write_json, write_jsonl
from heterogeneous.html import render_html
from heterogeneous.pipeline import Runner, generate_prompt, generation_quality, load_plan


def smoke_variants(plan):
    seen = set(); selected = []
    sources = {s['id']: s for s in plan['sources']}
    for v in plan['variants']:
        key = (v['generator_names'][0], sources[v['source_id']]['dataset'])
        if key not in seen:
            seen.add(key); selected.append(v)
    assert len(selected) == 16, len(selected)
    return selected


def verification_variants(plan):
    selected = [v for v in smoke_variants(plan) if v['generator_names'][0] != 'opus3']
    rng = random.Random(plan['config']['dataset']['seed'] + 15)
    for name in ['haiku', 'sonnet', 'opus']:
        candidates = [v for v in plan['variants'] if v['generator_names'] == [name] and v not in selected]
        selected.append(rng.choice(candidates))
    assert Counter(v['generator_names'][0] for v in selected) == {'haiku':5,'sonnet':5,'opus':5}
    return selected


def process(run, scope, jobs, brief_jobs=None):
    plan = load_plan(run); runner = Runner(run, plan)
    if jobs < 1 or (brief_jobs is not None and brief_jobs < 1):
        raise ValueError('Worker counts must be positive.')
    if brief_jobs is not None:
        # Scheduling does not alter prompts, model parameters or cache identity.
        runner.semaphores[digest(plan['config']['summarizer'])] = threading.Semaphore(brief_jobs)
    if scope == 'all':
        gate = json.loads((run / 'quality-gate.json').read_text())
        if gate.get('plan_id') != plan['id'] or gate.get('decision') != 'proceed':
            raise ValueError('Full generation requires a documented small-batch quality gate for this plan.')
    variants = (smoke_variants(plan) if scope == 'smoke' else verification_variants(plan)
                if scope == 'verify-five' else plan['variants'])
    write_json(run / f'{scope}-selection.json', {'plan_id': plan['id'], 'variants': [v['id'] for v in variants],
               'scheduling': {'workers': jobs, 'brief_concurrency': brief_jobs or plan['config']['summarizer']['concurrency']}})
    errors = []
    def work(v):
        for b in v['blocks']: runner.summary(v, b)
        for backend in runner.variant_backends(v):
            if backend['kind'] != 'claude-web':
                for b in v['blocks']: runner.generation(v, b, backend)
        return v['id']
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(work, v): v['id'] for v in variants}
        for i, f in enumerate(as_completed(futures), 1):
            try: f.result()
            except Exception as e:
                error={'variant': futures[f], 'error': str(e)}
                errors.append(error)
                print(json.dumps({'scope':scope,'failed':error}),flush=True)
                write_json(run / f'{scope}-errors.json', errors)
            print(json.dumps({'scope': scope, 'completed': i, 'total': len(variants), 'errors': len(errors)}), flush=True)
    write_json(run / f'{scope}-errors.json', errors)
    if scope != 'verify-five': export_browser(run, scope)
    assemble(run)
    if errors: raise ValueError(f'{len(errors)} failed variants; successes cached for resuming.')


def export_browser(run, scope):
    plan = load_plan(run); runner = Runner(run, plan)
    retry_file = run / 'browser-extra-attempts.json'
    retry_budget = json.loads(retry_file.read_text()) if retry_file.exists() else {}
    if retry_budget and retry_budget.get('plan_id') != plan['id']:
        raise ValueError('Browser retry budget belongs to a different frozen plan.')
    variants = smoke_variants(plan) if scope == 'smoke' else plan['variants']
    tasks = []; missing = []
    for v in variants:
        for backend in runner.variant_backends(v):
            if backend['kind'] != 'claude-web': continue
            for block in v['blocks']:
                if runner.cache_path('generation', block['id'], backend).exists(): continue
                summary_path = runner.cache_path('summary', block['id'], plan['config']['summarizer'])
                if not summary_path.exists(): missing.append(block['id']); continue
                brief = runner.summary(v, block)
                source = runner.sources[v['source_id']]
                rejected_path = run / 'browser-rejections' / f"{block['id']}.json"
                rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else []
                extra = retry_budget.get('blocks', {}).get(block['id'], 0)
                if not isinstance(extra, int) or isinstance(extra, bool) or not 0 <= extra <= 6:
                    raise ValueError('Browser extra-attempt budget must be an integer from zero to six.')
                if len(rejected) >= plan['config']['dataset']['quality_attempts'] + extra:
                    missing.append({'block_id': block['id'], 'reason': 'browser_quality_attempts_exhausted'}); continue
                feedback = rejected[-1]['error'] if rejected else ''
                if extra and len(rejected) >= plan['config']['dataset']['quality_attempts']:
                    feedback += (' Do not reconstruct a published passage from memory, even if you recognize '
                                 'the people, characters or scene. Compose every sentence anew with different '
                                 'wording and sentence structure, including any dialogue. Expand only the brief; '
                                 'do not quote a known book or copy the neighbouring paragraphs. '
                                 'For a recognisable fictional scene, you may invent different character names '
                                 'and fresh descriptive details to prevent memorized reproduction: this retry '
                                 'prioritizes original wording over fidelity to proper names.')
                if extra and len(rejected) >= 6 and source.get('genre') == 'fiction':
                    feedback += (' Use present-day English and indirect narration rather than quoted dialogue '
                                 'for this retry. Change the names and descriptive details of the fictional '
                                 'scene if necessary. The required outcome is newly composed AI prose, '
                                 'not restoration of the original literary text.')
                prompt = generate_prompt(source, v['blocks'], block, brief['text'], plan['config']['dataset'], feedback)
                tasks.append({'id': block['id'], 'variant_id': v['id'], 'source_id': source['id'],
                              'dataset': source['dataset'], 'backend': backend['name'], 'prompt': prompt})
    batches = []
    # A fresh chat per request; no source block is included in the expansion prompt.
    for task in tasks:
        batch = {'id': digest([plan['id'], task])[:24], 'plan_id': plan['id'], 'tasks': [task], 'prompt': task['prompt']}
        batches.append(batch)
    write_json(run / f'browser-{scope}-queue.json', {'plan_id': plan['id'], 'batches': batches, 'missing_briefs': missing})
    print(json.dumps({'browser_scope': scope, 'pending_calls': len(tasks), 'missing_briefs': len(missing)}), flush=True)
    return batches


def import_browser(run, queue, request_id, response_file, chat_url):
    plan = load_plan(run); runner = Runner(run, plan)
    payload = json.loads(queue.read_text()); assert payload['plan_id'] == plan['id']
    batch = next(b for b in payload['batches'] if b['id'] == request_id)
    raw = response_file.read_text(encoding='utf-8'); text = raw.strip()
    assert len(batch['tasks']) == 1
    task = batch['tasks'][0]; variant = next(v for v in plan['variants'] if v['id'] == task['variant_id'])
    block = next(b for b in variant['blocks'] if b['id'] == task['id'])
    backend = next(b for b in runner.variant_backends(variant) if b['name'] == task['backend'])
    rejected_path = run / 'browser-rejections' / f"{block['id']}.json"
    rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else []
    try:
        quality = runner.check_generation_quality(text, block, runner.sources[variant['source_id']])
    except ValueError as e:
        rejected.append({'request_id':request_id,'prompt':batch['prompt'],'raw_response':raw,'chat_url':chat_url,'error':str(e)})
        write_json(rejected_path,rejected)
        raise
    if not chat_url.startswith('https://claude.ai/chat/'): raise ValueError('Must record the observed Claude chat URL.')
    captured = datetime.now(timezone.utc).isoformat()
    result = {'text': text, 'prompt': batch['prompt'], 'quality': quality, 'quality_attempt': len(rejected)+1, 'rejected_attempts': [r['error'] for r in rejected],
              'cache_key': digest([plan['id'], 'generation', block['id'], backend]),
              'provenance': {'backend': backend['name'], 'kind': 'claude-web', 'requested_model': backend['model'],
                  'reported_model': 'Opus 3', 'family': backend.get('family'), 'revision': None,
                  'provider': 'claude.ai', 'request_id': chat_url, 'sampling': {}, 'generated_at': captured,
                  'finish_reason': 'ui_completed', 'prompt_sha256': text_hash(batch['prompt']), 'usage': None,
                  'metadata': {'model_identity_status': 'ui_label_only', 'selected_ui_model': 'Opus 3',
                               'exact_checkpoint_unreported': True, 'raw_response_sha256': text_hash(raw),
                               'chat_url': chat_url, 'project_context': 'empty', 'project_instructions': 'empty',
                               'fresh_chat': True, 'ui_system_prompt_uncontrolled': True}}}
    path = runner.cache_path('generation', block['id'], backend)
    if path.exists(): raise ValueError('Refusing to overwrite a generated browser result.')
    write_json(run / 'browser-captures' / f'{request_id}.json', {'request': batch, 'raw_response': raw, 'chat_url': chat_url, 'captured_at': captured})
    write_json(path, result)
    print(json.dumps({'imported': block['id'], 'quality': quality}), flush=True)


def assemble(run):
    plan = load_plan(run); manifest = Runner(run, plan).assemble(allow_partial=True)
    records = read_jsonl(run / 'dataset.jsonl')
    mixed = [r for r in records if r['construction']['kind'] == 'mixed']
    controls = [r for r in records if r['construction']['kind'] == 'human_control']
    write_jsonl(run / 'mixed.jsonl', mixed); write_jsonl(run / 'human-controls.jsonl', controls)
    if mixed: render_html(mixed, run / 'review.html', len(mixed))
    print(json.dumps({'mixed': len(mixed), 'controls': len(controls),
                      'models': dict(Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in mixed)),
                      'partial': manifest['partial']}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['smoke', 'verify-five', 'all', 'queue', 'import', 'assemble'])
    p.add_argument('--run', type=Path, required=True); p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--brief-jobs', type=int, help='Override summary scheduling concurrency without changing model parameters.')
    p.add_argument('--scope', choices=['smoke', 'all'], default='smoke')
    p.add_argument('--queue', type=Path); p.add_argument('--request-id'); p.add_argument('--response', type=Path); p.add_argument('--chat-url')
    args = p.parse_args()
    if args.command in ['smoke', 'verify-five', 'all']: process(args.run, args.command, args.jobs, args.brief_jobs)
    elif args.command == 'queue': export_browser(args.run, args.scope)
    elif args.command == 'import': import_browser(args.run, args.queue, args.request_id, args.response, args.chat_url)
    else: assemble(args.run)

if __name__ == '__main__': main()
