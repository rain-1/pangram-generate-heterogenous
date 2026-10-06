"""Combine completed, audited cohorts without changing records or their splits."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
import argparse
import hashlib
import json
import os
from pathlib import Path

from heterogeneous.core import read_jsonl, write_json, write_jsonl, atomic_text
from heterogeneous.pipeline import validate_dataset


def build(runs, out, cohort_names=None):
    if cohort_names is not None and (len(cohort_names) != len(runs) or len(set(cohort_names)) != len(cohort_names)):
        raise ValueError('Provide one unique cohort name for each run')
    records = []; cohorts = []
    for index, run in enumerate(runs):
        audit = json.loads((run / 'detector-quality-audit.json').read_text())
        parent = json.loads((run / 'parent-copy-audit.json').read_text())
        assert audit['complete'] and audit['integrity_checks'] == 'pass'
        assert not audit['prose_review_flags'] and not parent['flagged_blocks']
        assert parent['covered_sources'] == parent['total_sources']
        assert parent['plan_id'] == audit['plan_id']
        data = read_jsonl(run / 'dataset.jsonl')
        assert all(r['construction']['plan_id'] == audit['plan_id'] for r in data)
        records.extend(data)
        name = cohort_names[index] if cohort_names is not None else ('gpt_ml_papers' if 'gpt-ml-papers' in run.name else
                'gpt_general' if 'gpt-general' in run.name else
                'expansion' if 'expansion' in run.name else
                'ml_papers' if 'ml-papers' in run.name else 'original')
        cohorts.append({'name':name,'directory':str(run.resolve()), 'plan_id':audit['plan_id'],
                        'mixed':audit['mixed_documents'], 'controls':audit['human_controls'],
                        'models':audit['models'], 'dataset_sha256':hashlib.sha256((run/'dataset.jsonl').read_bytes()).hexdigest(),
                        'integrity_checks':'pass', 'full_parent_copy_check':'pass'})
    report = validate_dataset(records)
    mixed = [r for r in records if r['construction']['kind']=='mixed']
    controls = [r for r in records if r['construction']['kind']=='human_control']
    counts = Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in mixed)
    expected_counts = Counter()
    for cohort in cohorts:
        expected_counts.update(cohort['models'])
    assert counts == expected_counts
    total = sum(c['mixed'] for c in cohorts)
    assert len(mixed)==len(controls)==len({r['source']['id'] for r in mixed})==total
    write_jsonl(out/'dataset.jsonl', records)
    write_jsonl(out/'mixed.jsonl', mixed)
    write_jsonl(out/'human-controls.jsonl', controls)
    for split in ['train','validation','test']:
        write_jsonl(out/f'{split}.jsonl',(r for r in records if r['split']==split))
    report.update({'created_at':datetime.now(timezone.utc).isoformat(), 'complete':True,
                   'mixed_documents':total,'human_controls':total,'models':dict(counts),'cohorts':cohorts,
                   'checks':['Source, group and normalized duplicate split isolation across all cohorts',
                             f'Exact reconstruction/span integrity for all {len(records)} records',
                             'Complete cohort provenance audits and full-parent copy checks'],
                   'limitations':['Human-origin labels retain documentary candidate status.',
                                  'Opus 3 is attributed to its selected UI label; exact checkpoint unreported.',
                                  'Codex writer IDs are requested-only when the CLI does not report the served model.',
                                  'Generated papers are narrative excerpts, not full papers.',
                                  'Factual drift and output length are not rejection criteria.'],
                   'files':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                            ['dataset.jsonl','mixed.jsonl','human-controls.jsonl','train.jsonl','validation.jsonl','test.jsonl']}})
    write_json(out/'manifest.json', report)
    rows=[]
    for name in counts:
        values=[cohort['models'].get(name,0) for cohort in cohorts]
        rows.append(f'<tr><th>{escape(name)}</th>'+''.join(f'<td>{v}</td>' for v in values)+f'<td>{counts[name]}</td></tr>')
    links=[]
    for run,cohort in zip(runs,cohorts):
        relative=Path(os.path.relpath(run,out)).as_posix()
        label={'original':'Original four-model batch','ml_papers':'Pre-2015 ML papers','expansion':'Three-model expansion',
               'gpt_general':'GPT general documents','gpt_ml_papers':'GPT pre-2015 ML papers'}.get(cohort['name'],cohort['name'].replace('_',' ').title())
        links.append(f'<section><h2>{label}</h2><p>{cohort["mixed"]} mixed documents + {cohort["controls"]} unchanged controls.</p>'
                     f'<a href="{relative}/review.html">Open all colored documents</a> · '
                     f'<a href="{relative}/detector-quality-audit.json">Integrity audit</a> · '
                     f'<a href="{relative}/parent-copy-audit.json">Full-parent copy audit</a></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heterogeneous dataset · Complete handoff</title><style>
body{font:17px/1.6 system-ui;background:#f6f7f2;color:#20332b;margin:0}main{max-width:1050px;margin:35px auto;padding:25px}
h1{font-size:38px;line-height:1.2}a{color:#17623a}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #dbe3dc}
section{padding:20px;margin:20px 0;background:white;border:1px solid #dbe3dc;border-radius:12px}.human{background:#e6f4e9;color:#17623a;padding:6px 12px}.ai{background:#fff0df;color:#974210;padding:6px 12px}small{color:#627168}</style>
<main><h1>'''+str(total)+''' mixed documents, ready to inspect</h1><p>'''+str(total)+''' different source passages, plus '''+str(total)+''' matched unchanged controls. Exact Unicode character spans retain the source author and AI writer.</p>
<p><span class="human">Green · retained human text</span> <span class="ai">Orange · generated AI text</span></p>
<table><thead><tr><th>Writer</th>'''+''.join(f'<th>{escape(c["name"])}</th>' for c in cohorts)+'''<th>Total mixed</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p>Complete span/provenance audits and full-parent copy checks pass. The model probes and paper extraction spot checks are documented in the cohort audits.</p>'''+''.join(links)+'''
<section><h2>Combined downloads</h2><p><a href="mixed.jsonl">'''+str(total)+''' mixed documents · JSONL</a> · <a href="human-controls.jsonl">'''+str(total)+''' controls · JSONL</a> · <a href="dataset.jsonl">All '''+str(len(records))+''' records · JSONL</a> · <a href="manifest.json">Manifest and hashes</a></p><p>Existing train/validation/test assignments are preserved and checked across all cohorts.</p></section>
<p><small>Paper sources were published in 2000–2014. Human-origin labels retain the documented-provenance caveat. Opus 3 attribution is the selected browser UI label; the exact checkpoint is unreported. Factual drift and length differences are nonblocking for this detector dataset.</small></p></main></html>'''
    atomic_text(out/'index.html',page)
    print(json.dumps({'mixed':total,'controls':total,'models':dict(counts),'out':str(out.resolve())},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort-names',nargs='+',help='One unique portable cohort name per input run.')
    a=p.parse_args();build(a.runs,a.out,a.cohort_names)
