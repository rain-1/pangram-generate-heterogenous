"""Pin full parent texts and quarantine substantial memorized replacements."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from heterogeneous.core import read_jsonl, text_hash, write_json
from heterogeneous.parent_copy import parent_fraction
from heterogeneous.pipeline import Runner, load_plan


def prepare_policy(run, plan):
    target = run / 'parent-texts'; target.mkdir(exist_ok=True)
    standard_path = run / 'sources/standardebooks/records.jsonl'
    standard = {r['source_id']: r for r in read_jsonl(standard_path)} if standard_path.exists() else {}
    beige_path = run / 'sources/beigebook/records.jsonl'
    beige = {r['id']: r for r in read_jsonl(beige_path)} if beige_path.exists() else {}
    mapping = {}
    wiki_tables = {}
    for source in plan['sources']:
        meta = source['upstream_metadata']; text = None; evidence = None
        if source['dataset'] == 'gutenberg_selected':
            raw = Path(meta['source_file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == meta['source_file_sha256']
            text = raw.decode('utf-8-sig'); evidence = meta['source_file_sha256']
        elif source['dataset'] == 'standardebooks':
            row = standard[meta['source_id']]
            assert text_hash(row['text']) == row['clean_sha256']
            text = row['text']; evidence = row['clean_sha256']
        elif source['dataset'] == 'beigebook':
            row = beige[meta['parent_report_id']]
            assert text_hash(row['text']) == row['text_sha256']
            text = row['text']; evidence = row['text_sha256']
        elif source['dataset'] == 'jmlr_pre2015':
            pdf = Path(meta['pdf_path'])
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == meta['pdf_sha256']
            document = json.loads(pdf.with_suffix('.json').read_text())
            assert document['pdf_sha256'] == meta['pdf_sha256']
            text = '\n\n'.join(p['text'] for p in document['paragraphs'])
            evidence = meta['pdf_sha256']
        elif source['dataset'] == 'wikitext2_raw':
            import pyarrow.parquet as pq
            path = meta['source_file']
            if path not in wiki_tables:
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == meta['source_file_sha256']
                wiki_tables[path] = pq.read_table(path)['text'].to_pylist()
            begin, end = meta['source_row_range']
            text = ''.join(wiki_tables[path][begin:end])
            assert text_hash(text) == meta['raw_article_text_sha256']
            # Same mechanical punctuation recovery as the pinned source builder.
            text = re.sub(r'\s+@-@\s+', '-', text)
            text = re.sub(r'\s+@,@\s+', ',', text)
            text = re.sub(r'\s+@\.@\s+', '.', text)
            text = re.sub(r'[ \t]+([,.;:!?%)\]])', r'\1', text)
            text = re.sub(r'([(\[])[ \t]+', r'\1', text)
            text = re.sub(r"(?<=\w)[ \t]+('[sdm]|'re|'ve|'ll|n't)\b", r'\1', text)
            evidence = meta['raw_article_text_sha256']
        elif meta.get('parent_text_path') and meta.get('parent_text_sha256'):
            path = Path(meta['parent_text_path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['parent_text_sha256']
            text = path.read_bytes().decode('utf-8')
            evidence = meta['parent_text_sha256']
        if text is None:
            continue
        content_hash = text_hash(text)
        path = target / f'{content_hash}.txt'
        if not path.exists():
            path.write_text(text, encoding='utf-8', newline='')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == content_hash
        mapping[source['id']] = {'path': str(path.resolve()), 'sha256': content_hash,
                                'origin_sha256': evidence, 'scope': 'full original book/report or extracted paper body'}
    policy = {'plan_id': plan['id'], 'sources': mapping,
              'method': 'Eight-word exact coverage after case/Unicode/apostrophe normalization; same 15% bound, broader parent reference; parent prose never sent to writer.',
              'uncovered_sources': [s['id'] for s in plan['sources'] if s['id'] not in mapping]}
    write_json(run / 'parent-copy-policy.json', policy)
    return mapping


def audit(run, quarantine=False):
    plan = load_plan(run); policy = prepare_policy(run, plan); runner = Runner(run, plan)
    checks = []; flagged = []; variants = set(); quarantined = []
    for variant in plan['variants']:
        reference = policy.get(variant['source_id'])
        if not reference:
            continue
        for backend in runner.variant_backends(variant):
            for block in variant['blocks']:
                path = runner.cache_path('generation', block['id'], backend)
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                fraction = parent_fraction(result['text'], reference, plan['config']['dataset']['copy_ngram'])
                item = {'variant': variant['id'], 'block_id': block['id'], 'source_id': variant['source_id'],
                        'model': backend['name'], 'full_parent_copy_fraction': fraction,
                        'parent_sha256': reference['sha256']}
                checks.append(item)
                if fraction <= plan['config']['dataset']['max_copy_fraction']:
                    continue
                flagged.append(item); variants.add(variant['id'])
                if not quarantine:
                    continue
                destination = run / 'quarantine/full-parent-copy' / backend['name'] / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError('Prior quarantined result exists; refusing to overwrite history')
                path.rename(destination)
                error = (f'passage copies too much full-parent wording ({fraction:.1%}; maximum 15.0%); '
                         'compose entirely new prose and do not quote or reconstruct any known chapter or published passage, '
                         'including material outside the selected excerpt')
                if backend['kind'] == 'claude-web':
                    rejection_path = run / 'browser-rejections' / f"{block['id']}.json"
                    history = json.loads(rejection_path.read_text()) if rejection_path.exists() else []
                    history.append({'request_id': result['provenance']['request_id'], 'prompt': result['prompt'],
                                    'raw_response': result['text'], 'chat_url': result['provenance']['request_id'],
                                    'error': error, 'reason': 'full-parent audit found memorized material outside selected excerpt'})
                    write_json(rejection_path, history)
                quarantined.append({**item, 'preserved_cache': str(destination.resolve())})
    report = {'plan_id': plan['id'], 'audited_at': datetime.now(timezone.utc).isoformat(),
              'covered_sources': len(policy), 'total_sources': len(plan['sources']),
              'checked_blocks': len(checks), 'flagged_blocks': flagged, 'flagged_variants': sorted(variants),
              'flagged_models': dict(Counter(item['model'] for item in flagged)),
              'quarantined': quarantined, 'checks': checks,
              'limitations': ['Own-parent comparison does not search every human document on the Internet.',
                              'Uncovered collections retain the excerpt-level copy check.']}
    filename = 'parent-copy-quarantine.json' if quarantine else 'parent-copy-audit.json'
    write_json(run / filename, report)
    if quarantine:
        write_json(run / 'parent-copy-retry-selection.json', [{'variant': v} for v in sorted(variants)])
    print(json.dumps({k:report[k] for k in ['covered_sources','total_sources','checked_blocks','flagged_models','flagged_variants']},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--quarantine',action='store_true'); args=parser.parse_args()
    audit(args.run,args.quarantine)
