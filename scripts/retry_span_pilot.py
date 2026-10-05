"""Retry explicitly failed variants while preserving successful stage caches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from heterogeneous.core import write_json
from heterogeneous.pipeline import Runner, load_plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--errors',type=Path,required=True)
    parser.add_argument('--attempts',type=int,default=3)
    args=parser.parse_args()
    if args.attempts<1:raise ValueError('Attempts must be positive')
    plan=load_plan(args.run);runner=Runner(args.run,plan)
    wanted={e['variant'] for e in json.loads(args.errors.read_text())}
    remaining=[];recovered=[]
    for variant in plan['variants']:
        if variant['id'] not in wanted:continue
        errors=[]
        for attempt in range(args.attempts):
            try:
                for block in variant['blocks']:runner.summary(variant,block)
                for backend in runner.variant_backends(variant):
                    if backend['kind']!='claude-web':
                        for block in variant['blocks']:runner.generation(variant,block,backend)
                recovered.append(variant['id'])
                print(json.dumps({'recovered':variant['id'],'retry_round':attempt+1}),flush=True)
                break
            except Exception as error:
                errors.append(str(error))
                print(json.dumps({'failed':variant['id'],'retry_round':attempt+1,'error':str(error)}),flush=True)
        else:remaining.append({'variant':variant['id'],'errors':errors})
        write_json(args.run/'retry-results.json',{'plan_id':plan['id'],'recovered':recovered,'remaining':remaining})
    if remaining:raise ValueError(f'{len(remaining)} variants still need attention')


if __name__=='__main__':main()
