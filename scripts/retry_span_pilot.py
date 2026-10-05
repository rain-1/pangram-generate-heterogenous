"""Retry explicitly failed variants while preserving successful stage caches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from heterogeneous.core import write_json
from heterogeneous.pipeline import Runner, load_plan, summarize_prompt, summary_quality


class RetryRunner(Runner):
    def __init__(self,run,plan,compact_under=0):
        super().__init__(run,plan)
        self.compact_under=compact_under

    def summary(self,variant,block):
        if not self.compact_under or block['characters']>=self.compact_under:
            return super().summary(variant,block)
        source=self.sources[variant['source_id']];backend=self.plan['config']['summarizer']
        instruction=(f'\nThis is an exceptionally short source block. Condense the shared topic and final outcome '
                     f'into two very compact complete sentences, using at most {int(block["characters"]*.65)} '
                     'characters total. Group the information rather than repeating each line or headline. '
                     'Use only the central topic and contrast; omit titles, quotations, decorative wording and secondary details.')
        result=self.cached_call('summary',block,backend,
            lambda feedback:summarize_prompt(source['text'][block['start']:block['end']],
                self.plan['config']['dataset']['summary_min_sentences'],
                self.plan['config']['dataset']['summary_max_sentences'],feedback)+instruction,
            lambda text:summary_quality(text,block,source,self.plan['config']['dataset']))
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--errors',type=Path,required=True)
    parser.add_argument('--attempts',type=int,default=3)
    parser.add_argument('--compact-briefs-under',type=int,default=0,
                        help='Add recorded compact-sentence instructions for blocks shorter than this character count; checks remain unchanged.')
    args=parser.parse_args()
    if args.attempts<1:raise ValueError('Attempts must be positive')
    if args.compact_briefs_under<0:raise ValueError('Compact threshold must be nonnegative')
    plan=load_plan(args.run);runner=RetryRunner(args.run,plan,args.compact_briefs_under)
    if args.compact_briefs_under:
        write_json(args.run/'brief-retry-audit.json',{'plan_id':plan['id'],'compact_under_characters':args.compact_briefs_under,
            'method':'Append a topic/outcome-only compact-sentence instruction for exceptional short source blocks; the exact accepted prompt is saved in provenance.',
            'checks':'Original sentence, compression and source-copy requirements unchanged.'})
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
