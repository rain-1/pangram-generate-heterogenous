"""Validate the requested 15 records and render the direct read-through notes."""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import median

import jsonschema

from heterogeneous.core import atomic_text, digest, read_jsonl, text_hash, write_json, write_jsonl
from heterogeneous.html import render_html
from heterogeneous.pipeline import Runner, load_plan, validate_dataset


def build(run, notes_path):
    def no_generation(*args):raise AssertionError('Verification must never request a new model completion')
    plan=load_plan(run);runner=Runner(run,plan,completion=no_generation);selected=json.loads((run/'verify-five-selection.json').read_text())
    wanted=set(selected['variants']);sources={v['source_id'] for v in plan['variants'] if v['id'] in wanted}
    records=[r for r in read_jsonl(run/'mixed.jsonl') if r['source']['id'] in sources]
    records.sort(key=lambda r:(r['replacements'][0]['generation']['provenance']['backend'],r['source']['id']))
    notes=json.loads(notes_path.read_text());assert {r['id'] for r in records}==set(notes['records'])
    names=Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in records)
    assert names=={'haiku':5,'sonnet':5,'opus':5},names
    assert len({r['source']['id'] for r in records})==15
    schema=json.loads(Path('dataset.schema.json').read_text());validator=jsonschema.Draft202012Validator(schema)
    report=validate_dataset(records);audit=[];ratios=[];compressions=[]
    pin={b['name']:b['model'] for b in plan['config']['generators']}
    for r in records:
        validator.validate(r)
        v=next(v for v in plan['variants'] if v['source_id']==r['source']['id'])
        assert r['source']['text']==runner.sources[v['source_id']]['text']
        assert sum(s['end']-s['start'] for s in r['spans'])==len(r['text'])
        assert all(s['author']['type']=='non_ai' for s in r['spans'] if s['label']=='human')
        assert r['source']['human_verified'] is False and r['source']['binary_target'] is None
        blocks=[]
        assert [b['id'] for b in r['replacements']]==[b['id'] for b in v['blocks']]
        for b,expected in zip(r['replacements'],v['blocks']):
            assert b['source_start']==expected['start'] and b['source_end']==expected['end']
            backend=runner.variant_backends(v)[0]
            summary=runner.summary(v,expected);generation=runner.generation(v,expected,backend)
            assert summary==b['summarization'] and generation==b['generation']
            provenance=generation['provenance']
            assert provenance['reported_model']==pin[backend['name']]
            assert provenance['requested_model']==pin[backend['name']]
            assert provenance['metadata']['model_identity_status']=='reported'
            assert text_hash(generation['prompt'])==provenance['prompt_sha256']
            a=len(b['generated_text'])/b['source_characters'];c=len(b['summary'])/b['source_characters']
            ratios.append(a);compressions.append(c)
            blocks.append({'id':b['id'],'source_characters':b['source_characters'],'generated_characters':len(b['generated_text']),
                           'length_ratio':a,'brief_compression_ratio':c,'summary_quality':summary['quality'],
                           'generation_quality':generation['quality'],'requested_model':provenance['requested_model'],
                           'reported_model':provenance['reported_model'],'generation_request_id':provenance['request_id']})
        audit.append({'record_id':r['id'],'source_id':r['source']['id'],'model':backend['name'],'structural_checks':'pass',
                      'content_review':notes['records'][r['id']],'blocks':blocks})
    counts=Counter(n['status'] for n in notes['records'].values());by=defaultdict(Counter)
    detector_policy=notes.get('evaluation_policy')=='detector_task_fitness'
    for a in audit:by[a['model']][a['content_review']['status']]+=1
    report.update({'plan_id':plan['id'],'models':dict(names),'structural_passes':15,'review_method':notes['method'],
                   'evaluation_policy':notes.get('evaluation_policy','content_fidelity'),
                   'content_counts':dict(counts),'content_by_model':{k:dict(v) for k,v in by.items()},'documents':audit,
                   'replacement_length_ratio':{'min':min(ratios),'median':median(ratios),'max':max(ratios)},
                   'brief_compression_ratio':{'min':min(compressions),'median':median(compressions),'max':max(compressions)},
                   'limitations':notes['limitations'],'recommendations':notes['recommendations']})
    review_summary_path=run/'quality-review-summary.json'
    if review_summary_path.exists():
        report['independent_model_review']=json.loads(review_summary_path.read_text())
    write_jsonl(run/'verification-15.jsonl',records);write_json(run/'verification-15.json',report)
    write_jsonl(run/'verification-usable.jsonl',[r for r in records if notes['records'][r['id']]['status']=='usable'])
    write_jsonl(run/'verification-needs-review.jsonl',[r for r in records if notes['records'][r['id']]['status']=='review'])
    write_jsonl(run/'verification-rejected.jsonl',[r for r in records if notes['records'][r['id']]['status']=='reject'])
    output=run/'verification-15.html';render_html(records,output,15);page=output.read_text()
    page=page.replace('Heterogeneous text dataset · Example','Heterogeneous text dataset · 15-document verification')
    page=page.replace('expanded independently by each model','expanded by the writing model assigned to that source')
    table='<section class="verification"><h2>Verification: five documents per model</h2><p>All 15 pass the exact-span, source reconstruction, cache integrity, schema and reported-model checks. Content judgments below come from a direct assistant read-through, with independent model reviews used as supporting triage.</p><table><thead><tr><th>Model</th><th>Usable</th><th>Review</th><th>Reject</th></tr></thead><tbody>'
    for name in ['haiku','sonnet','opus']:
        c=by[name];label={'haiku':'Haiku 4.5','sonnet':'Sonnet 5.5','opus':'Opus 5.5'}[name]
        table+=f'<tr><td>{escape(label)}</td><td class="good">{c["usable"]}</td><td class="caution">{c["review"]}</td><td class="bad">{c["reject"]}</td></tr>'
    definition=('Usable means suitable prose for detector training with exact labels and verified generation provenance. Factual changes and replacement length are descriptive observations, not rejection criteria.' if detector_policy else 'Usable means no material continuity or factual failure found; minor style and length concerns may remain. Review means edit or further inspection is needed. Reject means regenerate from a corrected brief.')
    table+='</tbody></table><p>'+definition+'</p><p>These are different source passages, and every writer used Haiku-generated briefs. This small review does not rank the writing models.</p><details><summary>Browse all 15 verdicts and jump to each document</summary><table><thead><tr><th>Model</th><th>Source</th><th>Verdict</th></tr></thead><tbody>'
    for r in records:
        n=notes['records'][r['id']];name=r['replacements'][0]['generation']['provenance']['backend']
        colour={'usable':'good','review':'caution','reject':'bad'}[n['status']]
        table+=f'<tr><td>{escape(name.title())}</td><td><a href="#doc-{r["id"]}">{escape(r["source"].get("title",r["source"]["id"]))}</a></td><td class="{colour}">{escape(n["status"])}</td></tr>'
    table+='</tbody></table></details><ul>'
    table+=''.join('<li>'+escape(x)+'</li>' for x in notes['recommendations'])+'</ul></section>'
    page=page.replace('<section class="cards"',table+'<section class="cards"',1)
    page=page.replace('</style>','''
.verification{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin-bottom:28px}.verification table{border-collapse:collapse;width:100%;max-width:650px}.verification th,.verification td{text-align:left;padding:10px;border-bottom:1px solid var(--line)}.good{color:#17623a}.caution{color:#974210}.bad{color:#a12424}.review-notes{padding:18px 28px;border-top:1px solid var(--line);font-size:14px}.review-notes ul{padding-left:22px}.status{font-weight:700}.card details pre{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--ink)}
</style>''')
    for r in records:
        n=notes['records'][r['id']];name=r['replacements'][0]['generation']['provenance']['backend']
        marker=f'<article class="card" data-record-id="{r["id"]}">';start=page.index(marker);end=page.index('</article>',start)
        card=page[start:end];card=card.replace('class="card"',f'class="card" id="doc-{r["id"]}"',1)
        card=card.replace('<h2>Claude</h2>',f'<h2>{escape(name.title())} · {escape(r["source"].get("title",r["source"]["id"]))}</h2>',1)
        statusclass={'usable':'good','review':'caution','reject':'bad'}[n['status']]
        review=f'<div class="review-notes"><p class="status {statusclass}">{escape(n["status"].upper())}</p><ul>'+''.join('<li>'+escape(x)+'</li>' for x in n['findings'])+'</ul></div>'
        originals='<details><summary>Check the original blocks and boundaries</summary>'
        for b in r['replacements']:
            d=json.loads(b['generation']['prompt'].split('\n')[-1]);original=r['source']['text'][b['source_start']:b['source_end']]
            originals+=f'<h3>Original block {escape(b["id"])}</h3><pre>{escape(original)}</pre><h3>Retained preceding context</h3><pre>{escape(d.get("preceding_context",""))}</pre><h3>Retained following context</h3><pre>{escape(d.get("following_context",""))}</pre>'
        originals+='</details>'
        page=page[:start]+card+review+originals+page[end:]
    atomic_text(output,page)
    proceed=detector_policy and counts=={'usable':15}
    reason=('All 15 reviewed documents pass integrity and detector task-fitness checks. User clarified that factual fidelity and length differences are nonblocking.' if proceed else 'Resolve task-fitness or content review failures before scaling under the selected evaluation policy.')
    write_json(run/'quality-gate.json',{'plan_id':plan['id'],'decision':'proceed' if proceed else 'hold','evaluation_policy':notes.get('evaluation_policy','content_fidelity'),'reason':reason,
                                      'paragraph_count_policy':'soft_target' if detector_policy else 'exact',
                                      'verification_report':'verification-15.json'})
    print(json.dumps({'structural_passes':15,'content_counts':dict(counts),'content_by_model':{k:dict(v) for k,v in by.items()},'html':str(output)},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--notes',type=Path,required=True)
    a=p.parse_args();build(a.run,a.notes)
if __name__=='__main__':main()
