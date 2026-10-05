"""Prepare a reproducible, provenance-preserving heterogeneous-text pilot.

This script samples source candidates; it does not certify human authorship.
Model generation is performed separately by heterogeneous.Runner.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from heterogeneous.config import load_config
from heterogeneous.core import normalized_hash, paragraph_spans, select_blocks, text_hash, write_json, write_jsonl

CORE_MANIFEST = '2f8a39152727a7bfb589d5b193df787c0ebc6038b0a9afe7023ff14667958ba6'
REPO_REVISION = 'b618ca42b85fcb4cd8ddcc60cf18ed957262cd6a'


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def acquire_beige(root):
    """Keep whole report narratives and exact downloaded HTML, not short chunks."""
    output = root / 'records.jsonl'
    if output.exists():
        return
    raw = root / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    base = 'https://www.federalreserve.gov'
    headers = {'User-Agent': 'pangram-at-home-research/1.0 (dated public document collection)'}
    def get(url):
        r = requests.get(url, headers=headers, timeout=60); r.raise_for_status(); return r.content
    urls = set()
    for year in range(2019, 2023):
        url = f'{base}/monetarypolicy/beigebook{year}.htm'
        b = get(url); (raw / f'archive-{year}.html').write_bytes(b)
        soup = BeautifulSoup(b, 'html.parser')
        urls.update(urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                    if re.search(r'/monetarypolicy/beigebook\d{6,8}\.htm$', urljoin(url, a['href'])))
    def report(url):
        b = get(url); soup = BeautifulSoup(b, 'html.parser')
        title = soup.title.get_text(' ', strip=True)
        match = re.search(r'([A-Z][a-z]+ \d{1,2}, \d{4})$', title)
        if not match: raise ValueError(f'No report date: {url}')
        date = datetime.strptime(match[1], '%B %d, %Y').date().isoformat()
        if date > '2022-12-31': raise ValueError('Unexpected recent report')
        (raw / f'report-{date}.html').write_bytes(b)
        for node in soup.select('header, footer, nav, script, style, table, .social, .printOnly'):
            node.decompose()
        main = soup.find('main') or soup.find(id='article') or soup.find(id='content') or soup.body
        paragraphs = []
        for node in main.find_all('p'):
            text = re.sub(r'\s+', ' ', node.get_text(' ', strip=True)).strip()
            if len(text.split()) >= 20 and not text.lower().startswith(('for media', 'contact', 'last update')):
                paragraphs.append(text)
        text = '\n\n'.join(paragraphs)
        if len(paragraphs) < 8: raise ValueError(f'Too little report prose: {url}')
        return {'id': f'beigebook:{date}', 'text': text, 'date': date, 'title': title,
                'url': url, 'raw_sha256': text_hash_bytes(b), 'text_sha256': text_hash(text),
                'publisher': 'Board of Governors of the Federal Reserve System',
                'license': 'Board website public domain unless otherwise indicated; attribution and third-party caveats retained',
                'rights_url': f'{base}/disclaimer.htm', 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        records = sorted(executor.map(report, sorted(urls)), key=lambda r: r['id'])
    write_jsonl(output, records)
    write_json(root / 'manifest.json', {'reports': len(records), 'records_sha256': file_hash(output),
               'provenance': 'Dated official 2019–2022 release archives; current captured revision not independently dated.',
               'collection_script_sha256': file_hash(Path(__file__))})
    print(f'Acquired {len(records)} Beige Book reports.', flush=True)


def text_hash_bytes(b):
    return hashlib.sha256(b).hexdigest()


def regional_beige(root):
    """Recover district boundaries from saved HTML, keeping original prose exact."""
    target = root / 'regional-records.jsonl'
    if target.exists(): return target
    records = []
    for line in (root / 'records.jsonl').open():
        report = json.loads(line)
        raw_path = root / 'raw' / f"report-{report['date']}.html"
        assert file_hash(raw_path) == report['raw_sha256']
        soup = BeautifulSoup(raw_path.read_bytes(), 'html.parser')
        main = soup.find(id='article') or soup.find('main')
        current = None; sections = []; paragraphs = []
        for node in main.find_all(['h4', 'p']):
            if node.name == 'h4':
                if current and paragraphs: sections.append((current, paragraphs))
                current = node.get_text(' ', strip=True); paragraphs = []
            elif current:
                text = re.sub(r'\s+', ' ', node.get_text(' ', strip=True)).strip()
                if len(text.split()) >= 20 and not text.lower().startswith(('for media', 'contact', 'last update')):
                    paragraphs.append(text)
        if current and paragraphs: sections.append((current, paragraphs))
        if len(sections) != 12: raise ValueError(f'Expected twelve district sections: {report["id"]}, got {len(sections)}')
        for region, paragraphs in sections:
            text = '\n\n'.join(paragraphs)
            start = report['text'].find(text)
            if start < 0: raise ValueError('Regional prose does not match archived whole-report extraction')
            slug = re.sub('[^a-z0-9]+', '-', region.lower()).strip('-')
            records.append({**report, 'id': report['id'] + ':' + slug, 'region': region,
                            'parent_report_id': report['id'], 'parent_report_character_range': [start, start+len(text)],
                            'text': text, 'text_sha256': text_hash(text)})
    write_jsonl(target, records)
    return target


def passages(text, max_chars=12000):
    units = paragraph_spans(text); start = 0
    for i in range(1, len(units) + 1):
        end = units[i-1][1]
        if i == len(units) or units[i][1] - units[start][0] > max_chars:
            if i - start >= 8:
                yield units[start][0], end, text[units[start][0]:end]
            start = i


def prepare(core, run, config, seed):
    if (run / 'input.jsonl').exists(): raise ValueError('Refusing to overwrite a frozen sample.')
    assert file_hash(core / 'manifest.json') == CORE_MANIFEST
    manifest = json.loads((core / 'manifest.json').read_text())
    for p, h in manifest['outputs'].items(): assert file_hash(core / p) == h, p
    settings = load_config(config)['dataset']
    pools = {name: [] for name in ['gutenberg_selected', 'wikitext2_raw', 'standardebooks', 'beigebook']}
    originals = {}
    def add(pool, source, original):
        if '\x00' in source['text'] or len(paragraph_spans(source['text'])) < 8: return
        if not select_blocks(source['text'], settings, text_hash(source['id'])): return
        source.update(human_verified=False, authorship='human_candidate', binary_target=None, language='en')
        pools[pool].append(source); originals[source['id']] = original
    with gzip.open(core / 'train.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line); text = row['text']; assert text_hash(text) == row['text_sha256']
            if row['dataset'] not in pools: continue
            processed = text.replace('\n', '\n\n') if row['dataset'] == 'wikitext2_raw' else text
            assert processed.split() == text.split()
            metadata = {k: v for k, v in row.items() if k != 'text'}
            author = None if row['dataset'] in ['hansard', 'wikitext2_raw'] else row.get('author')
            source = {'id': row['id'], 'group_id': row['split_group'], 'split': row['split'],
                      'reference': f"{(core/'train.jsonl.gz').resolve()}#{row['id']}",
                      'text': processed, 'author_id': author, 'genre': row['genre'], 'license': row['license'],
                      'dataset': row['dataset'], 'title': row.get('title'),
                      'provenance': 'Pinned DATA_HANDOFF documentary core; integrity verified, authorship remains a candidate claim.',
                      'human_origin_basis': row['human_origin_basis'], 'upstream_metadata': metadata,
                      'preprocessing': {'method': 'restore paragraph separators from WikiText native line breaks' if processed != text else 'none',
                                        'original_text_sha256': row['text_sha256'], 'whitespace_tokens_unchanged': True,
                                        'source_release_manifest_sha256': CORE_MANIFEST}}
            add(row['dataset'], source, row)
    standard = run / 'sources/standardebooks/records.jsonl'
    core_authors = {r['author_id'] for r in pools['gutenberg_selected']}
    with standard.open() as f:
        for line in f:
            row = json.loads(line)
            if row['author_names'][0] in core_authors: continue
            assert text_hash(row['text']) == row['clean_sha256']
            assert row['source_revision_date'] <= '2022-12-31T23:59:59Z'
            for a, b, text in passages(row['text']):
                sid = f"standardebooks:{row['source_id']}:{a}-{b}"
                add('standardebooks', {'id': sid, 'group_id': f"book-author/{row['author_names'][0]}",
                    'reference': row['source_url'], 'text': text, 'author_id': row['author_names'][0],
                    'genre': 'fiction', 'dataset': 'standardebooks', 'title': row['title'],
                    'license': row['rights_status'], 'provenance': row['human_origin_evidence'],
                    'human_origin_basis': {'kind': 'historical_work_pinned_pre_2023_edition', 'commit': row['source_revision'],
                                          'commit_date': row['source_revision_date'], 'publication_year': row['publication_date']},
                    'upstream_metadata': {k: v for k, v in row.items() if k != 'text'}, 'parent_character_range': [a, b]},
                    {'id': sid, 'text': text, 'parent_id': row['document_id'], 'character_range': [a,b], 'text_sha256': text_hash(text)})
    with regional_beige(run/'sources/beigebook').open() as f:
        for line in f:
            row=json.loads(line); assert text_hash(row['text']) == row['text_sha256']
            for a,b,text in passages(row['text']):
                sid=f"{row['id']}:{a}-{b}"
                add('beigebook', {'id':sid, 'group_id':row['parent_report_id'], 'reference':row['url'], 'text':text,
                    'author_id':row['publisher'], 'genre':'professional_finance', 'dataset':'beigebook', 'title':row['title'],
                    'region':row['region'],
                    'license':row['license'], 'provenance':'Dated official Federal Reserve narrative; current captured revision not independently dated.',
                    'human_origin_basis':{'kind':'historical_official_report_candidate','publication_date':row['date'],'captured_version_verified':False},
                    'upstream_metadata':{k:v for k,v in row.items() if k!='text'},'parent_character_range':[a,b]},
                    {'id':sid,'text':text,'parent_id':row['id'],'character_range':[a,b],'text_sha256':text_hash(text)})
    rng=random.Random(seed); selected=[]; seen_hashes=set(); groups=Counter(); inventory=[]
    print('Eligible source passages:', {name: len(pool) for name,pool in pools.items()}, flush=True)
    for name,pool in pools.items():
        ordered=sorted(pool,key=lambda r:r['id']);rng.shuffle(ordered);picked=[]
        cap={'gutenberg_selected':20,'standardebooks':12,'wikitext2_raw':1,'beigebook':4}[name]
        for row in ordered:
            h=normalized_hash(row['text'])
            if h in seen_hashes or groups[row['group_id']]>=cap:continue
            picked.append(row);seen_hashes.add(h);groups[row['group_id']]+=1
            if len(picked)==100:break
        if len(picked)!=100:raise ValueError(f'{name}: only {len(picked)} eligible selections; candidates={len(pool)}')
        for index,row in enumerate(picked):
            row['generator_names']=[['haiku','sonnet','opus','opus3'][index%4]]
        selected.extend(picked);inventory.append({'dataset':name,'eligible_passages':len(pool),'sampled':len(picked),'group_cap':cap})
    rng.shuffle(selected)
    # Existing train partitions remain train. New groups use deterministic plan splitting.
    write_jsonl(run/'input.jsonl',selected)
    write_jsonl(run/'source-originals.jsonl',(originals[r['id']] for r in selected))
    write_json(run/'source-audit.json',{'seed':seed,'sources':len(selected),'inventory':inventory,
        'source_assignments':dict(Counter(r['generator_names'][0] for r in selected)),
        'sampling':'Uniform seeded shuffle within each eligible dataset stratum; four equal strata, explicit parent/author caps; not a global uniform sample.',
        'excluded_datasets':{'hansard':'Only two passages passed multi-paragraph prose eligibility; headings and Q&A unsuitable for this pilot.'},
        'input_sha256':file_hash(run/'input.jsonl'),'originals_sha256':file_hash(run/'source-originals.jsonl'),
        'core_manifest_sha256':CORE_MANIFEST,'source_repo_revision':REPO_REVISION,
        'heldout_policy':'Only the core training export was sampled; its validation/test remain untouched.',
        'authorship':'Documented human-origin candidates; no detector filtering or measured purity guarantee.',
        'checks':['Pinned core and all file hashes','Unchanged upstream candidate labels','Exact text hashes','No normalized exact duplicates',
                  'Paragraph and eligible replacement checks','Whitespaced-only WikiText paragraph restoration','Parent/author sampling caps']})
    print(json.dumps({'sources':len(selected),'inventory':inventory,'source_assignments':dict(Counter(r['generator_names'][0] for r in selected))},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--core',type=Path,default=Path('../pretraining-datawork/data/corpus-human-diverse-core-v1'))
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--seed',type=int,default=20261005)
    args=parser.parse_args();acquire_beige(args.run/'sources/beigebook');prepare(args.core,args.run,args.config,args.seed)

if __name__=='__main__':main()
