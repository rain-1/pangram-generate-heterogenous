"""Generate/review an expansion with resumable subscription-backed caches."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import threading

from heterogeneous.core import digest, read_jsonl, write_json, write_jsonl
from heterogeneous.html import render_html
from heterogeneous.pipeline import Runner, load_plan


def assemble(run):
    plan=load_plan(run);report=Runner(run,plan).assemble(allow_partial=True)
    records=read_jsonl(run/'dataset.jsonl')
    mixed=[r for r in records if r['construction']['kind']=='mixed']
    controls=[r for r in records if r['construction']['kind']=='human_control']
    write_jsonl(run/'mixed.jsonl',mixed);write_jsonl(run/'human-controls.jsonl',controls)
    if mixed:render_html(mixed,run/'review.html',len(mixed))
    selected=set(json.loads((run/'verification-selection.json').read_text())['variants'])
    sources={v['source_id'] for v in plan['variants'] if v['id'] in selected}
    probe=[r for r in mixed if r['source']['id'] in sources]
    if probe:render_html(probe,run/'model-probe-review.html',len(probe))
    print(json.dumps({'mixed':len(mixed),'controls':len(controls),'models':dict(Counter(r['replacements'][0]['generation']['provenance']['backend'] for r in mixed))}),flush=True)
    return report


def process(run,scope,jobs,writer_jobs=None):
    plan=load_plan(run);source_audit=json.loads((run/'source-audit.json').read_text())
    assert source_audit['plan_id']==plan['id']
    assert hashlib.sha256((run/'input.jsonl').read_bytes()).hexdigest()==source_audit['input_sha256']
    if all(source['dataset']=='jmlr_pre2015' for source in plan['sources']):
        from scripts.run_ml_paper_pilot import source_check
        source_check(run)
    policy=json.loads((run/'parent-copy-policy.json').read_text())
    assert policy['plan_id']==plan['id'] and not policy['uncovered_sources']
    for reference in {r['path']:r for r in policy['sources'].values()}.values():
        assert hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest()==reference['sha256']
    if scope=='all':
        audit=json.loads((run/'model-probe-audit.json').read_text())
        assert audit['plan_id']==plan['id'] and audit['decision']=='proceed'
    selected=json.loads((run/'verification-selection.json').read_text())
    assert selected['plan_id']==plan['id']
    wanted=set(selected['variants']);variants=[v for v in plan['variants'] if scope=='all' or v['id'] in wanted]
    runner=Runner(run,plan);runner.semaphores[digest(plan['config']['summarizer'])]=threading.Semaphore(8)
    if writer_jobs is not None:
        if writer_jobs<1:raise ValueError('Writer scheduling concurrency must be positive')
        for backend in plan['config']['generators']:
            runner.semaphores[digest(backend)]=threading.Semaphore(writer_jobs)
    write_json(run/f'{scope}-scheduling.json',{'plan_id':plan['id'],'workers':jobs,'brief_concurrency':8,
        'writer_concurrency':{b['name']:writer_jobs or b['concurrency'] for b in plan['config']['generators']},
        'scope':'Operational scheduling only; frozen model configuration, prompts and cache identity unchanged.'})
    errors=[]
    def work(v):
        for attempt in range(3):
            try:
                for block in v['blocks']:runner.summary(v,block)
                for backend in runner.variant_backends(v):
                    for block in v['blocks']:runner.generation(v,block,backend)
                return v['id']
            except Exception:
                if attempt==2:raise
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures={pool.submit(work,v):v for v in variants}
        for i,future in enumerate(as_completed(futures),1):
            try:future.result()
            except Exception as e:errors.append({'variant':futures[future]['id'],'error':str(e)})
            progress={'scope':scope,'completed_tasks':i,'total':len(variants),'errors':len(errors)}
            write_json(run/f'{scope}-progress.json',progress);write_json(run/f'{scope}-errors.json',errors)
            print(json.dumps(progress),flush=True)
    assemble(run)
    if errors:raise ValueError(f'{len(errors)} variants need retries; successful calls retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--scope',choices=['probe','all','assemble'],default='probe');p.add_argument('--jobs',type=int,default=20)
    p.add_argument('--writer-jobs',type=int,help='Override operational writer scheduling without changing model parameters.')
    a=p.parse_args()
    (a.run/f'{a.scope}.pid').write_text(str(os.getpid())+'\n')
    if a.scope=='assemble':assemble(a.run)
    else:process(a.run,a.scope,a.jobs,a.writer_jobs)
