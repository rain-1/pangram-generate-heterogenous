"""Freeze unused passages, retaining prior group splits and source evidence."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random
import re

from heterogeneous.core import digest, normalized_hash, paragraph_spans, read_jsonl, select_blocks, text_hash, write_json, write_jsonl
from heterogeneous.pipeline import make_plan, save_plan
from scripts.prepare_span_pilot import CORE_MANIFEST, REPO_REVISION, file_hash, passages, regional_beige

QUOTAS = {
    'haiku': {'gutenberg_selected': 15, 'standardebooks': 35, 'wikitext2_raw': 25, 'beigebook': 25},
    'sonnet': {'gutenberg_selected': 15, 'standardebooks': 35, 'wikitext2_raw': 25, 'beigebook': 25},
    'opus': {'standardebooks': 100},
}


def parent_range(source):
    metadata = source['upstream_metadata']
    dataset = source['dataset']
    if dataset in ['gutenberg_selected', 'wikitext2_raw']:
        start, end = map(int, source['id'].rsplit('/', 1)[1].split('-'))
        return metadata['parent_document_id'], start, end
    if dataset == 'standardebooks':
        return metadata['document_id'], *source['parent_character_range']
    if dataset == 'beigebook':
        offset = metadata['parent_report_character_range'][0]
        a, b = source['parent_character_range']
        return metadata['parent_report_id'], offset+a, offset+b
    return None


def prepare(run, base, combined, core, seed, quotas=None, generation_config=None):
    if run.exists():
        raise ValueError('Use a fresh run directory; never overwrite a frozen expansion.')
    old_records = read_jsonl(combined/'dataset.jsonl')
    old_sources = {r['source']['id']: r['source'] for r in old_records}
    old_groups = {}; old_hashes = set(); ranges = {}
    for record in old_records:
        source = record['source']
        assert old_groups.setdefault(source['group_id'], record['split']) == record['split']
        old_hashes.add(normalized_hash(source['text']))
    for source in old_sources.values():
        position = parent_range(source)
        if position:
            parent, a, b = position
            ranges.setdefault(parent, []).append((a,b))
    quotas = deepcopy(QUOTAS if quotas is None else quotas)
    expected_models = Counter({name:sum(values.values()) for name,values in quotas.items()})
    total = sum(expected_models.values())
    assert total > 0 and all(count > 0 for count in expected_models.values())
    assert all(isinstance(count,int) and count >= 0 for values in quotas.values() for count in values.values())
    config = deepcopy(generation_config or json.loads((base/'plan.json').read_text())['config'])
    config['generators'] = [b for b in config['generators'] if b['name'] in quotas]
    assert {b['name'] for b in config['generators']} == set(quotas)
    config['dataset']['seed'] = seed
    settings = config['dataset']
    assert file_hash(core/'manifest.json') == CORE_MANIFEST
    for name, expected in json.loads((core/'manifest.json').read_text())['outputs'].items():
        assert file_hash(core/name) == expected
    pools = {name: [] for name in ['gutenberg_selected','standardebooks','wikitext2_raw','beigebook']}
    assert all(set(values) <= set(pools) for values in quotas.values())
    all_core_authors = set(); excluded = Counter()
    def add(source):
        if source['id'] in old_sources:
            excluded['previous_source_id'] += 1; return
        h = normalized_hash(source['text'])
        if h in old_hashes:
            excluded['previous_normalized_text'] += 1; return
        if len(paragraph_spans(source['text'])) < 8 or '\x00' in source['text']:
            excluded['prose_structure'] += 1; return
        source.update(human_verified=False, authorship='human_candidate', binary_target=None, language='en')
        source['sha256'] = text_hash(source['text']); source['normalized_sha256'] = h
        selection_seed = digest([seed, source['id'], source['sha256'], 0])
        if not select_blocks(source['text'], settings, selection_seed):
            excluded['replacement_selection_ineligible'] += 1; return
        position = parent_range(source)
        if position:
            parent, a, b = position
            if any(a < end and begin < b for begin,end in ranges.get(parent, [])):
                excluded['previous_parent_range_overlap'] += 1; return
        if source['group_id'] in old_groups:
            previous = source.get('split')
            source['split'] = old_groups[source['group_id']]
            assert previous is None or previous == source['split']
        pools[source['dataset']].append(source)
    with gzip.open(core/'train.jsonl.gz','rt',encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['dataset']=='gutenberg_selected': all_core_authors.add(row.get('author'))
            if row['dataset'] not in pools: continue
            text=row['text'];assert text_hash(text)==row['text_sha256']
            processed=text.replace('\n','\n\n') if row['dataset']=='wikitext2_raw' else text
            assert processed.split()==text.split()
            add({'id':row['id'],'group_id':row['split_group'],'split':row['split'],
                'reference':f"{(core/'train.jsonl.gz').resolve()}#{row['id']}",
                'text':processed,'author_id':None if row['dataset']=='wikitext2_raw' else row.get('author'),
                'genre':row['genre'],'dataset':row['dataset'],'title':row.get('title'),'license':row['license'],
                'provenance':'Pinned documentary core; source candidate labels retained, no detector filtering.',
                'human_origin_basis':row['human_origin_basis'],'upstream_metadata':{k:v for k,v in row.items() if k!='text'},
                'preprocessing':{'method':'restore paragraph separators from native WikiText line breaks' if processed!=text else 'none',
                                 'original_text_sha256':row['text_sha256'],'whitespace_tokens_unchanged':True,
                                 'source_release_manifest_sha256':CORE_MANIFEST}})
    for row in read_jsonl(base/'sources/standardebooks/records.jsonl'):
        if row['author_names'][0] in all_core_authors: continue
        assert text_hash(row['text'])==row['clean_sha256'] and row['source_revision_date']<='2022-12-31T23:59:59Z'
        for a,b,text in passages(row['text']):
            add({'id':f"standardebooks:{row['source_id']}:{a}-{b}",'group_id':f"book-author/{row['author_names'][0]}",
                'reference':row['source_url'],'text':text,'author_id':row['author_names'][0],'genre':'fiction',
                'dataset':'standardebooks','title':row['title'],'license':row['rights_status'],
                'provenance':row['human_origin_evidence'],
                'human_origin_basis':{'kind':'historical_work_pinned_pre_2023_edition','commit':row['source_revision'],
                                     'commit_date':row['source_revision_date'],'publication_year':row['publication_date']},
                'upstream_metadata':{k:v for k,v in row.items() if k!='text'},'parent_character_range':[a,b]})
    for row in read_jsonl(regional_beige(base/'sources/beigebook')):
        assert text_hash(row['text'])==row['text_sha256']
        for a,b,text in passages(row['text']):
            add({'id':f"{row['id']}:{a}-{b}",'group_id':row['parent_report_id'],'reference':row['url'],'text':text,
                'author_id':row['publisher'],'genre':'professional_finance','dataset':'beigebook','title':row['title'],
                'region':row['region'],'license':row['license'],
                'provenance':'Dated official report; captured revision not independently dated.',
                'human_origin_basis':{'kind':'historical_official_report_candidate','publication_date':row['date'],'captured_version_verified':False},
                'upstream_metadata':{k:v for k,v in row.items() if k!='text'},'parent_character_range':[a,b]})
    eligible={name:len(rows) for name,rows in pools.items()}
    print('unused_eligible',eligible,flush=True)
    rng=random.Random(seed); selected=[]; selected_hashes=set(); groups=Counter()
    for name,pool in pools.items():
        ordered=sorted(pool,key=lambda r:r['id']);rng.shuffle(ordered)
        assignments=[model for model in quotas for _ in range(quotas[model].get(name,0))]
        rng.shuffle(assignments)
        cap={'gutenberg_selected':20,'standardebooks':25,'wikitext2_raw':1,'beigebook':4}[name]
        chosen=[]
        for row in ordered:
            if len(chosen)==len(assignments):break
            if row['normalized_sha256'] in selected_hashes or groups[row['group_id']]>=cap:continue
            row['generator_names']=[assignments[len(chosen)]]
            chosen.append(row);selected_hashes.add(row['normalized_sha256']);groups[row['group_id']]+=1
        assert len(chosen)==len(assignments),(name,len(chosen),len(assignments),eligible)
        selected.extend(chosen)
    rng.shuffle(selected)
    assert len(selected)==total and Counter(r['generator_names'][0] for r in selected)==expected_models
    assert all(r['genre']=='fiction' for r in selected if r['generator_names']==['opus'])
    run.mkdir(parents=True)
    (run/'sources').symlink_to((base/'sources').resolve(),target_is_directory=True)
    write_jsonl(run/'input.jsonl',selected)
    write_json(run/'config.json',config)
    plan=make_plan(run/'input.jsonl',config);assert len(plan['variants'])==total and not plan['skipped']
    save_plan(plan,run)
    for source in plan['sources']:
        if source['group_id'] in old_groups:assert plan['splits'][source['id']]==old_groups[source['group_id']]
    ids=[]
    probe_counts = {}
    for model in quotas:
        candidates=[v for v in plan['variants'] if v['generator_names']==[model]]
        rng.shuffle(candidates)
        picked=[]
        lookup={s['id']:s for s in plan['sources']}
        for dataset in pools:
            if quotas[model].get(dataset,0):
                picked.append(next(v for v in candidates if lookup[v['source_id']]['dataset']==dataset))
        for v in candidates:
            if len(picked)>=min(5,len(candidates)):break
            if v not in picked:picked.append(v)
        probe_counts[model]=len(picked)
        ids.extend(v['id'] for v in picked)
    write_json(run/'verification-selection.json',{'plan_id':plan['id'],'variants':ids,'models':probe_counts})
    write_json(run/'source-audit.json',{'plan_id':plan['id'],'seed':seed,'sources':total,'eligible_unused':eligible,
        'quotas':quotas,'source_assignments':dict(expected_models),'input_sha256':file_hash(run/'input.jsonl'),
        'previous_dataset_sha256':file_hash(combined/'dataset.jsonl'),'core_manifest_sha256':CORE_MANIFEST,
        'source_repo_revision':REPO_REVISION,'exclusions':dict(excluded),
        'sampling':'Seeded shuffle within unused eligible source strata; explicit quotas and per-group caps.',
        'checks':['Pinned core/export hashes','Distinct unused source IDs and normalized text','No overlap with previous parent ranges',
                  'Previously used source-group splits inherited','Only core training export sampled','Opus restricted to fiction',
                  'Candidate authorship labels preserved; no detector filtering'],
        'generation_policy':'Facts, length and paragraph counts remain nonblocking; exact spans, provenance, useful prose and copy checks required.'})
    write_json(run/'quality-gate.json',{'plan_id':plan['id'],'decision':'proceed','scope':'probe',
        'evaluation_policy':'detector_task_fitness','paragraph_count_policy':'soft_target',
        'full_generation_requires':f'matching successful {len(ids)}-document model-probe-audit.json',
        'reason':'Apply the established detector-task policy to the probe; bulk generation still requires review.'})
    print(json.dumps({'plan_id':plan['id'],'sources':total,'blocks':sum(len(v['blocks']) for v in plan['variants']),
                      'model_counts':dict(Counter(s['generator_names'][0] for s in selected)),
                      'splits':dict(Counter(plan['splits'].values()))},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('runs/heterogeneous-pilot-20261005-v2'))
    p.add_argument('--combined',type=Path,default=Path('runs/heterogeneous-20261005-combined'))
    p.add_argument('--core',type=Path,default=Path('../pretraining-datawork/data/corpus-human-diverse-core-v1'))
    p.add_argument('--quotas',type=Path,help='JSON mapping backend names to collection counts.')
    p.add_argument('--config',type=Path,help='Normalized JSON generation config; omitted uses the base plan.')
    p.add_argument('--seed',type=int,default=2026100502);a=p.parse_args()
    prepare(a.run,a.base,a.combined,a.core,a.seed,
            json.loads(a.quotas.read_text()) if a.quotas else None,
            json.loads(a.config.read_text()) if a.config else None)
