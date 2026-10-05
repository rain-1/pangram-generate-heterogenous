"""Build a portable, audited Hub release without changing local run artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from heterogeneous.core import read_jsonl, write_json, write_jsonl
from heterogeneous.pipeline import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / 'runs/heterogeneous-20261005-with-gpt'
CONFIGS = ['mixed', 'human_controls', 'all', 'original_mixed', 'ml_papers_mixed']
SPLITS = ['train', 'validation', 'test']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value, counts):
    """Remove machine identifiers; never alter text, prompts, offsets or hashes."""
    if isinstance(value, dict):
        return {k: clean(v, counts) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v, counts) for v in value]
    if isinstance(value, str):
        if value.startswith('https://claude.ai/chat/'):
            counts['private_chat_references_hashed'] += 1
            return 'redacted-chat-sha256:' + hashlib.sha256(value.encode()).hexdigest()
        if value.startswith(('/home/', '/root/', '/tmp/', '/mnt/')):
            counts['machine_paths_replaced'] += 1
            if '/pangram-at-home/' in value:
                return 'upstream-workspace/' + value.split('/pangram-at-home/', 1)[1]
            return 'upstream-local-artifact/' + Path(value).name
    return value


def portable_record(original, counts):
    record = clean(deepcopy(original), counts)
    source = record['source']
    if original['source']['reference'].startswith('/'):
        source['upstream_metadata']['original_reference_hint'] = source['reference']
        source['reference'] = source['upstream_metadata']['url']
    assert record['text'] == original['text']
    assert source['text'] == original['source']['text']
    for before, after in zip(original['replacements'], record['replacements']):
        for step in ['generation', 'summarization']:
            assert before[step]['prompt'] == after[step]['prompt']
            assert before[step]['text'] == after[step]['text']
    return record


SPAN = pa.struct([
    pa.field('start', pa.int64()), pa.field('end', pa.int64()),
    pa.field('label', pa.string()), pa.field('source_start', pa.int64()),
    pa.field('source_end', pa.int64()), pa.field('replacement_id', pa.string()),
    pa.field('author_type', pa.string()), pa.field('author_id', pa.string()),
    pa.field('backend', pa.string()), pa.field('requested_model', pa.string()),
    pa.field('reported_model', pa.string()), pa.field('model_identity_status', pa.string()),
])
VIEW_SCHEMA = pa.schema([
    *[pa.field(k, pa.string()) for k in [
        'id', 'split', 'kind', 'cohort', 'generator_backend', 'requested_model',
        'reported_model', 'model_identity_status', 'source_id', 'group_id',
        'source_dataset', 'source_author', 'source_title', 'source_reference',
        'source_license', 'source_text', 'text',
        'text_sha256', 'offset_unit',
    ]],
    pa.field('human_verified', pa.bool_()), pa.field('end_exclusive', pa.bool_()),
    pa.field('ai_fraction_chars', pa.float64()), pa.field('spans', pa.list_(SPAN)),
])


def identity(author):
    if author['type'] == 'non_ai':
        return 'source_candidate'
    return author.get('metadata', {}).get('model_identity_status', 'unreported')


def flat_record(record, cohort_names):
    source = record['source']
    author = next((s['author'] for s in record['spans'] if s['label'] == 'ai'), {})
    spans = []
    for s in record['spans']:
        a = s['author']
        spans.append({
            **{k: s.get(k) for k in ['start', 'end', 'label', 'source_start', 'source_end', 'replacement_id']},
            'author_type': a['type'], 'author_id': a.get('id'),
            **{k: a.get(k) for k in ['backend', 'requested_model', 'reported_model']},
            'model_identity_status': identity(a),
        })
    return {
        **{k: record[k] for k in ['id', 'split', 'text', 'text_sha256', 'offset_unit', 'end_exclusive', 'ai_fraction_chars']},
        'kind': record['construction']['kind'],
        'cohort': cohort_names[record['construction']['plan_id']],
        'generator_backend': author.get('backend'), 'requested_model': author.get('requested_model'),
        'reported_model': author.get('reported_model'),
        'model_identity_status': identity(author) if author else None,
        'source_id': source['id'], 'group_id': source['group_id'],
        'source_dataset': source['dataset'], 'source_author': source.get('author_id'),
        'source_title': source.get('title'), 'source_reference': source['reference'],
        'source_license': source['license'], 'human_verified': source['human_verified'],
        'source_text': source['text'], 'spans': spans,
    }


def matches(record, config):
    mixed = record['kind'] == 'mixed'
    return (config == 'all' or config == 'mixed' and mixed
            or config == 'human_controls' and not mixed
            or config == 'original_mixed' and mixed and record['cohort'] == 'original'
            or config == 'ml_papers_mixed' and mixed and record['source_dataset'] == 'jmlr_pre2015'
            or config == 'expansion_mixed' and mixed and record['cohort'] == 'expansion'
            or config == 'gpt_general_mixed' and mixed and record['cohort'] == 'gpt_general'
            or config == 'gpt_ml_papers_mixed' and mixed and record['cohort'] == 'gpt_ml_papers')


def build(out, repo_id, combined=COMBINED, version='1.2.0', previous_release=None):
    if out.exists():
        raise ValueError(f'Release directory already exists: {out}; choose a new directory.')
    out.mkdir(parents=True)
    source_manifest = json.loads((combined / 'manifest.json').read_text())
    assert source_manifest['complete']
    for name, expected in source_manifest['files'].items():
        assert sha(combined / name) == expected, name
    raw = read_jsonl(combined / 'dataset.jsonl')
    runs = {}; cohort_names = {}
    for cohort in source_manifest['cohorts']:
        name=cohort.get('name') or ('original' if 'opus3' in cohort['models'] else 'ml_papers')
        assert name not in runs
        runs[name]=Path(cohort['directory']);cohort_names[cohort['plan_id']]=name
    configs=CONFIGS+(['expansion_mixed'] if 'expansion' in runs else [])
    configs += [f'{name}_mixed' for name in ['gpt_general','gpt_ml_papers'] if name in runs]
    counts = Counter()
    records = [portable_record(r, counts) for r in raw]
    preserved_previous=0
    if previous_release is not None:
        previous=read_jsonl(previous_release/'records/all.jsonl')
        by_id={r['id']:r for r in records}
        for record in previous:
            assert by_id[record['id']]==record,f'Previous full record changed: {record["id"]}'
        preserved_previous=len(previous)
    validator = jsonschema.Draft202012Validator(json.loads((ROOT / 'dataset.schema.json').read_text()))
    for r in records:
        validator.validate(r)
    report = validate_dataset(records)
    mixed = [r for r in records if r['construction']['kind'] == 'mixed']
    controls = [r for r in records if r['construction']['kind'] == 'human_control']
    assert len(mixed) == len(controls) == report['sources'] == source_manifest['mixed_documents']
    total=len(mixed);span_total=sum(len(r['replacements']) for r in mixed)
    if version=='1.1.0':
        assert 'expansion' in runs and total==1000
        assert Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in mixed)=={'haiku':300,'sonnet':300,'opus':300,'opus3':100}
    if version=='1.2.0':
        assert {'gpt_general','gpt_ml_papers'} <= set(runs) and total==1400
        assert Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in mixed)=={'haiku':300,'sonnet':300,'opus':300,'opus3':100,'gpt_sol':200,'gpt_luna':200}
    write_jsonl(out / 'records/all.jsonl', records)
    write_jsonl(out / 'records/mixed.jsonl', mixed)
    write_jsonl(out / 'records/human-controls.jsonl', controls)
    write_jsonl(out / 'sources/originals.jsonl', [r['source'] for r in mixed])
    flat = [flat_record(r,cohort_names) for r in records]
    split_counts = {}
    for config in configs:
        split_counts[config] = {}
        for split in SPLITS:
            rows = [r for r in flat if matches(r, config) and r['split'] == split]
            path = out / 'data' / config / f'{split}.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(rows, schema=VIEW_SCHEMA)
            pq.write_table(table, path, compression='zstd')
            assert pq.read_table(path).to_pylist() == rows
            assert 'source_text_sha256' not in table.column_names
            split_counts[config][split] = len(rows)
    (out / 'schema').mkdir()
    shutil.copy2(ROOT / 'dataset.schema.json', out / 'schema/full-record.schema.json')
    (out / 'schema/arrow-schema.txt').write_text(str(VIEW_SCHEMA) + '\n')
    write_json(out / 'manifest/splits.json', split_counts)
    audit_names = [
        'detector-quality-audit.json', 'parent-copy-audit.json', 'source-audit.json',
        'quality-gate.json', 'verification-15.json', 'verification-notes.json',
        'source-check.json', 'source-visual-review.json', 'model-probe-audit.json',
        'browser-scheduling.json', 'parent-copy-quarantine.json',
        'all-scheduling.json', 'probe-scheduling.json',
        'retry-results.json', 'brief-retry-audit.json',
    ]
    for cohort, run in runs.items():
        a = json.loads((run / 'detector-quality-audit.json').read_text())
        p = json.loads((run / 'parent-copy-audit.json').read_text())
        assert a['complete'] and a['integrity_checks'] == 'pass' and not a['prose_review_flags']
        assert not p['flagged_blocks'] and p['covered_sources'] == p['total_sources']
        for name in audit_names:
            path = run / name
            if path.exists():
                write_json(out / 'audits' / cohort / name, clean(json.loads(path.read_text()), counts))
        plan = json.loads((run / 'plan.json').read_text())
        write_json(out / 'provenance' / f'{cohort}-planning.json', clean({
            'original_plan_id': plan['id'], 'original_plan_artifact_sha256': sha(run / 'plan.json'),
            'representation': 'Portable selection/configuration export; source metadata is in sources/originals.jsonl.',
            **{k: plan[k] for k in ['schema_version', 'tool_version', 'prompt_version', 'config', 'splits', 'variants', 'skipped']},
            'source_ids': [s['id'] for s in plan['sources']],
        }, counts))
        parent = json.loads((run / 'parent-copy-policy.json').read_text())
        write_json(out / 'provenance' / f'{cohort}-parent-copy-policy.json', clean(parent, counts))
    write_json(out / 'manifest/upstream-handoff.json', clean(source_manifest, counts))
    write_json(out / 'manifest/portable-metadata-changes.json', dict(counts))
    for directory in ['heterogeneous', 'scripts', 'tests', 'examples']:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and (path.suffix in ['.py', '.toml'] or path.parent==ROOT/'examples' and path.name in ['config.gpt-expansion.json','quotas.gpt-expansion.json']):
                dest = out / 'reproduction' / path.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    for name in ['pyproject.toml', 'README.md', 'dataset.schema.json']:
        shutil.copy2(ROOT / name, out / 'reproduction' / name)
    meta = {
        'language': ['en'], 'license': 'other', 'license_name': 'mixed-source-terms',
        'license_link': 'LICENSE.md', 'size_categories': ['1K<n<10K'],
        'pretty_name': 'Heterogeneous AI Spans — Human/AI Mixed Documents',
        'tags': ['ai-detection', 'span-annotation', 'mixed-authorship', 'model-attribution', 'synthetic-data'],
        'configs': [{'config_name': name, **({'default': True} if name == 'mixed' else {}),
                     'data_files': [{'split': s, 'path': f'data/{name}/{s}.parquet'} for s in SPLITS]}
                    for name in configs],
    }
    writer_counts=Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in mixed)
    writer_table=[]
    identities={'haiku':'claude-haiku-4-5-20251001','sonnet':'claude-sonnet-5-5','opus':'claude-opus-5-5','opus3':'Browser UI label `Opus 3`',
                'gpt_sol':'Requested `gpt-6.1-sol`; served identity unreported',
                'gpt_luna':'Requested `gpt-6-luna`; served identity unreported'}
    writer_order=[('haiku','Haiku'),('sonnet','Sonnet'),('opus','Current Opus'),('opus3','Opus 3'),('gpt_sol','GPT-6.1 Sol'),('gpt_luna','GPT-6 Luna')]
    cohorts=list(runs)
    for writer,label in writer_order:
        if writer not in writer_counts:continue
        values=[sum(r['replacements'][0]['generation']['provenance']['backend']==writer for r in mixed if cohort_names[r['construction']['plan_id']]==c) for c in cohorts]
        writer_table.append(f'| {label} | '+ ' | '.join(map(str,values))+f' | {writer_counts[writer]} | {identities[writer]} |')
    source_counts=Counter(r['source']['dataset'] for r in mixed)
    has_gpt={'gpt_general','gpt_ml_papers'} <= set(runs)
    gpt_section='''### Version 1.2.0 GPT expansion

Adds 100 general documents and 100 distinct pre-2015 ML-paper excerpts for each of GPT-6.1 Sol and GPT-6 Luna: 400 mixed documents plus 400 controls. General passages per writer comprise 10 Gutenberg, 40 Standard Ebooks, 3 WikiText and 47 Beige Book excerpts, reflecting the remaining unused eligible source pool. The 200 additional papers exclude every previously used paper.

GPT writers used Codex subscription authentication with requested IDs `gpt-6.1-sol` and `gpt-6-luna`, and explicit low reasoning effort. The CLI did not report the served model: `reported_model=null`, `model_identity_status="requested_only"`. No served checkpoint is inferred. Briefs still use Haiku, as in previous cohorts. New paper excerpts require at least five continuous clean prose paragraphs, with at most 65% of source characters replaced; the same 3/4/6-paragraph blocks, brief bottleneck and 15% source-copy ceiling apply.

All {{PREVIOUS_RECORD_COUNT}} previous full records are preserved exactly, including IDs, source/generated text, prompts and split assignments. `ml_papers_mixed` now includes both the earlier and GPT paper cohorts; `gpt_ml_papers_mixed` isolates the added 200 papers.''' if has_gpt else ''
    substitutions={
        '{{REPO_ID}}':repo_id,'{{VERSION}}':version,'{{TOTAL}}':str(total),
        '{{RECORD_TOTAL}}':str(len(records)),'{{SPAN_TOTAL}}':str(span_total),
        '{{WRITER_TABLE}}':'\n'.join(writer_table),
        '{{WRITER_HEADER}}':'| Writer | '+' | '.join(cohorts)+' | Mixed total | Recorded identity |\n|---|'+''.join('---:|' for c in cohorts)+'---:|---|',
        '{{EXTRA_COHORT_OPTIONS}}':''.join(f'<option value="{c}">{c.replace("_"," ")}</option>' for c in cohorts if c not in ['original','ml_papers']),
        '{{EXTRA_WRITER_OPTIONS}}':''.join(f'<option value="{w}">{label}</option>' for w,label in writer_order if w in ['gpt_sol','gpt_luna'] and w in writer_counts),
        '{{COHORT_CONFIGS}}':' / '.join(f'`{c}`' for c in cohorts),
        '{{WRITER_COUNT}}':str(len(writer_counts)),
        '{{PREVIOUS_RECORD_COUNT}}':str(preserved_previous),
        '{{GENERATION_BACKENDS}}':'Claude models ran through Claude Code subscription authentication; GPT writers ran through Codex subscription authentication, retaining requested-only attribution when served identity was unreported; Opus 3 ran through the browser UI. No OpenRouter/open-weight generations are included.' if has_gpt else 'Modern Claude models ran through Claude Code subscription authentication; Opus 3 ran through the browser UI. No GPT/OpenRouter/open-weight generations are included.',
        '{{GPT_EXPANSION_SECTION}}':gpt_section.replace('{{PREVIOUS_RECORD_COUNT}}',str(preserved_previous)),
        '{{ADDITIONAL_CONFIG_EXAMPLES}}':f'gpt_general = load_dataset("{repo_id}", "gpt_general_mixed")\ngpt_papers = load_dataset("{repo_id}", "gpt_ml_papers_mixed")' if has_gpt else '',
        '{{ML_EXTRACTION_PARAGRAPHS}}':'The original paper cohort requires at least six continuous usable prose paragraphs; the GPT paper cohort requires at least five.' if has_gpt else 'Papers had to yield a continuous sequence of at least six usable prose paragraphs.',
        '{{PDF_REVIEW_DESCRIPTION}}':'Five actual PDF-page renders were checked for the original paper cohort, and five additional pages for the GPT paper cohort.' if has_gpt else 'Five actual paper PDF-page renders were compared with extraction.',
        '{{SPLIT_TABLE}}':'\n'.join(f'| `{c}` | {v["train"]} | {v["validation"]} | {v["test"]} | {sum(v.values())} |' for c,v in split_counts.items()),
        **{f'{{{{COUNT_{k}}}}}':str(v) for k,v in source_counts.items()},
    }
    def substitute(content):
        for key,value in substitutions.items():content=content.replace(key,value)
        return content
    (out / 'README.md').write_text('---\n' + yaml.safe_dump(meta, sort_keys=False) + '---\n\n' +
        substitute((ROOT / 'packaging/dataset-card.md').read_text()),encoding='utf-8')
    for source, dest in [('schema-guide.md', 'schema/README.md'), ('license.md', 'LICENSE.md'),
                         ('review.html', 'review/index.html'), ('example.py', 'examples/load_and_inspect.py')]:
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(substitute((ROOT / 'packaging' / source).read_text()), encoding='utf-8')
    review_records = [{
        **{k: f[k] for k in ['id', 'split', 'kind', 'cohort', 'generator_backend', 'requested_model','reported_model','model_identity_status',
                            'source_title', 'source_author', 'source_dataset', 'source_reference', 'text', 'source_text', 'spans']},
        'briefs': [v['summary'] for v in r['replacements']],
    } for r, f in zip(records, flat)]
    review_path = out / 'review/index.html'
    review_path.write_text(review_path.read_text().replace('{{RECORDS}}',
        json.dumps(review_records, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')))
    (out / 'assets').mkdir()
    (out / 'assets/overview.svg').write_text(substitute((ROOT / 'packaging/overview.svg').read_text()))
    summary = {
        'release_version': version, 'built_at': datetime.now(timezone.utc).isoformat(),
        'repo_id': repo_id, 'records': len(records), 'mixed_documents': total, 'human_controls': total,
        'distinct_sources': total, 'ai_replacement_spans': span_total,
        'previous_full_records_preserved_exactly':preserved_previous,
        'writer_document_counts': dict(Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in mixed)),
        'source_document_counts': dict(Counter(r['source']['dataset'] for r in mixed)),
        'splits': split_counts, 'integrity': report,
        'checks': ['Original manifest hashes verified', f'Full JSON Schema validation for {len(records)} portable records',
                   'Exact reconstruction and Unicode span coverage', 'Source/group/duplicate split isolation',
                   f'Complete upstream audits; full-parent checks cover {span_total} replacement blocks',
                   f'All {len(configs)*3} Parquet files round-trip without field changes',
                   'Redundant source_text_sha256 column omitted from every Parquet configuration'],
        'schema_changes':{'removed_columns':['source_text_sha256'],'scope':'Parquet records in all configurations; compute from source_text when needed.'},
        'portable_metadata_changes': dict(counts),
        'limitations': source_manifest['limitations'] + [
            'Full parent documents, private model-session URLs, and runtime logs are not distributed.',
            'Source excerpts retain mixed upstream rights; no uniform content license is asserted.',
            'The ML-paper Opus examples were already generated; no further Opus ML run is scheduled.',
        ],
    }
    write_json(out / 'manifest/release.json', summary)
    # Scan data/documents for machine identities and actual private chat URLs.
    scan_files = list(out.rglob('*.json')) + list(out.rglob('*.jsonl')) + [out/'review/index.html', out/'README.md']
    for path in scan_files:
        content = path.read_text()
        assert '/home/riv/' not in content, path
        assert not re.search(r'https://claude\.ai/chat/[0-9a-f-]{20,}', content), path
        assert not re.search(r'\bhf_[A-Za-z0-9]{25,}\b', content), path
    files = sorted(p for p in out.rglob('*') if p.is_file())
    checksum_text = ''.join(f'{sha(p)}  {p.relative_to(out).as_posix()}\n' for p in files)
    (out / 'SHA256SUMS.txt').write_text(checksum_text)
    archive = out / f'heterogeneous-ai-spans-v{version}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in [*files, out / 'SHA256SUMS.txt']:
            z.write(path, 'heterogeneous-ai-spans/' + path.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in files:
            assert hashlib.sha256(z.read('heterogeneous-ai-spans/' + path.relative_to(out).as_posix())).hexdigest() == sha(path)
    (out / 'SHA256SUMS.txt').write_text(checksum_text + f'{sha(archive)}  {archive.name}\n')
    print(json.dumps({'directory': str(out), 'records': len(records), 'splits': split_counts,
                      'files': len(list(out.rglob('*'))), 'archive_mb': round(archive.stat().st_size / 1e6, 2)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repo-id', default='open-text-detector/heterogeneous-ai-spans')
    parser.add_argument('--combined',type=Path,default=COMBINED)
    parser.add_argument('--version',default='1.2.0')
    parser.add_argument('--previous-release',type=Path,help='Require exact preservation of every earlier portable full record.')
    args = parser.parse_args()
    build(args.out.resolve(), args.repo_id,args.combined.resolve(),args.version,args.previous_release)
