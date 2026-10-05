"""Generate/review an expansion with resumable subscription-backed caches."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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
from heterogeneous.providers import ProviderError


def bounded_work(variants, work, jobs, completed):
    """Keep only jobs requests in flight; the callback may stop new submissions."""
    pending={};iterator=iter(variants);stopped=False
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        def fill():
            while not stopped and len(pending)<jobs:
                variant=next(iterator,None)
                if variant is None:break
                pending[pool.submit(work,variant)]=variant
        fill()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                variant=pending.pop(future)
                if completed(variant,future):stopped=True
            fill()
    return stopped


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


def process(run,scope,jobs,writer_jobs=None,brief_jobs=8,provider_error_limit=8):
    if jobs<1 or brief_jobs<1 or provider_error_limit<1:raise ValueError('Scheduling values must be positive')
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
    runner=Runner(run,plan);runner.semaphores[digest(plan['config']['summarizer'])]=threading.Semaphore(brief_jobs)
    if writer_jobs is not None:
        if writer_jobs<1:raise ValueError('Writer scheduling concurrency must be positive')
        for backend in plan['config']['generators']:
            runner.semaphores[digest(backend)]=threading.Semaphore(writer_jobs)
    write_json(run/f'{scope}-scheduling.json',{'plan_id':plan['id'],'workers':jobs,'brief_concurrency':brief_jobs,
        'provider_error_limit':provider_error_limit,'submission_policy':'Bounded workers; stop submitting after repeated provider failures.',
        'writer_concurrency':{b['name']:writer_jobs or b['concurrency'] for b in plan['config']['generators']},
        'scope':'Operational scheduling only; frozen model configuration, prompts and cache identity unchanged.'})
    errors=[]
    def work(v):
        for attempt in range(3):
            try:
                for block in v['blocks']:
                    try:runner.summary(v,block)
                    except ProviderError as e:e.backend=plan['config']['summarizer']['name'];raise
                for backend in runner.variant_backends(v):
                    for block in v['blocks']:
                        try:runner.generation(v,block,backend)
                        except ProviderError as e:e.backend=backend['name'];raise
                return v['id']
            except ProviderError:raise
            except Exception:
                if attempt==2:raise
    from collections import Counter
    provider_errors=Counter();completed_tasks=0
    def completed(v,future):
        nonlocal completed_tasks
        completed_tasks+=1
        try:future.result()
        except Exception as e:
            item={'variant':v['id'],'error':str(e)}
            if isinstance(e,ProviderError):
                backend=getattr(e,'backend','unknown');provider_errors[backend]+=1
                item.update(provider_error=True,backend=backend)
            errors.append(item)
        stop=any(n>=provider_error_limit for n in provider_errors.values())
        progress={'scope':scope,'completed_tasks':completed_tasks,'total':len(variants),'errors':len(errors),
                  'paused_for_provider_errors':stop,'provider_errors':dict(provider_errors)}
        write_json(run/f'{scope}-progress.json',progress);write_json(run/f'{scope}-errors.json',errors)
        print(json.dumps(progress),flush=True)
        return stop
    paused=bounded_work(variants,work,jobs,completed)
    assemble(run)
    if paused:raise ValueError('Repeated provider failures; paused submissions, completed calls retained')
    if errors:raise ValueError(f'{len(errors)} variants need retries; successful calls retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--scope',choices=['probe','all','assemble'],default='probe');p.add_argument('--jobs',type=int,default=20)
    p.add_argument('--writer-jobs',type=int,help='Override operational writer scheduling without changing model parameters.')
    p.add_argument('--brief-jobs',type=int,default=8)
    p.add_argument('--provider-error-limit',type=int,default=8)
    a=p.parse_args()
    (a.run/f'{a.scope}.pid').write_text(str(os.getpid())+'\n')
    if a.scope=='assemble':assemble(a.run)
    else:process(a.run,a.scope,a.jobs,a.writer_jobs,a.brief_jobs,a.provider_error_limit)
