"""Collect dated JMLR PDFs and recover narrative paragraph excerpts.

No model is used for extraction. Raw PDFs, layout XML and paragraph page/bbox
locations are retained; historical publication is evidence, not certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import shutil
import unicodedata
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests

from heterogeneous.config import load_config
from heterogeneous.core import normalized_hash, select_blocks, text_hash, write_json, write_jsonl
from heterogeneous.pipeline import make_plan, save_plan

NS = {'h': 'http://www.w3.org/1999/xhtml'}
HEADERS = {'User-Agent': 'pangram-at-home-research/1.0 (historical ML paper corpus)'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    if not path.exists():
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path.read_bytes()


def inventory(root):
    papers = []
    for volume in range(1, 16):
        url = f'https://jmlr.org/papers/v{volume}/'
        path = root / f'volume-{volume}.html'
        soup = BeautifulSoup(fetch(url, path), 'html.parser')
        for link in soup.find_all('a', href=True):
            if link.get_text(strip=True).lower().strip('[]') != 'pdf':
                continue
            dd = link.find_parent('dd')
            if not dd:
                continue
            dt = dd.find_parent('dt') or dd.find_previous_sibling('dt')
            title = ''.join(str(x) for x in dt.contents if isinstance(x, str)).strip() if dt else ''
            title = BeautifulSoup(title, 'html.parser').get_text(' ', strip=True)
            author = dd.find('i')
            details = dd.get_text(' ', strip=True)
            year = re.search(r',\s*(20\d\d)\.', details)
            if not title or not author or not year or int(year[1]) >= 2015:
                continue
            pdf_url = urljoin(url, link['href']).replace('http://', 'https://')
            pid = Path(pdf_url).stem
            abstract = next((a for a in dd.find_all('a', href=True) if a.get_text(strip=True).strip('[]') == 'abs'), None)
            papers.append({'id': f'jmlr:v{volume}:{pid}', 'paper_id': pid,
                           'title': title, 'authors': author.get_text(' ', strip=True),
                           'year': int(year[1]), 'volume': volume, 'pdf_url': pdf_url,
                           'reference': urljoin(url, abstract['href']) if abstract else pdf_url,
                           'index_url': url, 'index_sha256': sha(path), 'index_citation': details.split('[abs]')[0].strip()})
    return papers


def prose(text):
    words = text.split()
    if len(words) < 35 or len(words) > 450:
        return False
    if not re.match(r'[A-Z“"]', text) or not re.search(r'[.!?][”"]?$', text):
        return False
    if re.match(r'(Figure|Table|Algorithm|Lemma|Theorem|Proof|Corollary|Proposition|References|Acknowledg)', text):
        return False
    if re.search(r'\(cid:|\ufffd|[\x00-\x08\x0b-\x1f]', text):
        return False
    if re.search(r'[∑∏∫∀∃≤≥∈∉⊆⊗∞]|(?:^|\s)[=+_]\s', text):
        return False
    normal = sum(bool(re.fullmatch(r'[A-Za-z][A-Za-z\-’\']*[.,;:!?)]?', w.strip('(“"'))) for w in words)
    return normal / len(words) >= .78


def extract(path):
    xml = path.with_suffix('.xml')
    if not xml.exists():
        subprocess.run(['pdftotext', '-bbox-layout', str(path), str(xml)], check=True, capture_output=True)
    # Some historical font encodings emit XML-forbidden control characters.
    # Preserve raw XML, replace controls in the parse view with a visible marker,
    # and reject every prose paragraph containing that marker.
    clean_xml = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '\ufffd', xml.read_text())
    tree = ET.fromstring(clean_xml)
    metadata = {m.attrib['name']: m.attrib.get('content') for m in tree.findall('.//h:meta', NS)}
    pages = tree.findall('.//h:page', NS)
    first_text = ' '.join(w.text or '' for w in pages[0].findall('.//h:word', NS))
    if not re.search(r'Journal of Machine Learning Research.*?\b(20\d\d)\b', first_text):
        raise ValueError('PDF title-page publication year not found')
    published = int(re.search(r'Journal of Machine Learning Research.*?\b(20\d\d)\b', first_text)[1])
    if published >= 2015:
        raise ValueError('PDF publication is too recent')
    # Layout paragraphs may contain several indented paragraphs. Split at body
    # indentation, blank-line gaps and headings; join continuations across pages.
    paragraphs = []
    current = []
    locations = []
    started = False
    finished = False
    previous = None
    def flush():
        nonlocal current, locations
        if current:
            text = ''
            for line in current:
                if text.endswith('-') and re.match(r'[a-z]', line):
                    text = text[:-1] + line
                else:
                    text += (' ' if text else '') + line
            paragraphs.append({'text': unicodedata.normalize('NFC', text), 'locations': locations,
                               'eligible_prose': prose(text)})
        current, locations = [], []
    for page_number, page in enumerate(pages, 1):
        lines = []
        for line in page.findall('.//h:line', NS):
            text = ' '.join(w.text or '' for w in line.findall('h:word', NS)).strip()
            y = float(line.attrib['yMin'])
            if (not text or re.fullmatch(r'\d+', text) or re.match(r'^[c©\ufffd].*\b20\d\d\b', text)
                    or y < 72 or float(line.attrib['yMax']) > float(page.attrib['height']) - 48):
                continue
            lines.append({'text': text, 'page': page_number,
                          **{k: float(line.attrib[k]) for k in ('xMin', 'yMin', 'xMax', 'yMax')}})
        lines.sort(key=lambda row: (round(row['yMin'], 1), row['xMin']))
        long_lines = [row['xMin'] for row in lines if len(row['text']) > 65]
        baseline = sorted(long_lines)[len(long_lines)//4] if long_lines else 90
        for line in lines:
            text = line['text']
            heading = bool(re.match(r'^\d+(?:\.\d+)*\.?\s+[A-Z]', text) and len(text.split()) < 16)
            if not started:
                if re.match(r'^1\.?\s+(Introduction|Overview|Motivation|Background)', text, re.I):
                    started = True
                continue
            if text == 'References' or re.match(r'^\d+\.?\s+References$', text):
                finished = True
                break
            if heading:
                flush()
                paragraphs.append({'text': text, 'locations': [line], 'eligible_prose': False})
                previous = None
                continue
            if previous and current:
                same_page = previous['page'] == page_number
                gap = line['yMin'] - previous['yMax'] if same_page else 0
                indent = line['xMin'] > baseline + 7
                previous_end = bool(re.search(r'[.!?][”"]?$', previous['text']))
                if (indent and previous_end) or (same_page and gap > 9) or (not same_page and indent and previous_end):
                    flush()
            current.append(text)
            locations.append(line)
            previous = line
        if finished:
            break
    flush()
    return {'paragraphs': paragraphs, 'pdf_metadata': metadata, 'pdf_publication_year': published,
            'pdf_pages': len(pages), 'layout_xml_sha256': sha(xml)}


def candidates(paper, raw, minimum_paragraphs=6):
    path = raw / f"v{paper['volume']}-{paper['paper_id']}.pdf"
    fetch(paper['pdf_url'], path)
    if not path.read_bytes().startswith(b'%PDF-'):
        raise ValueError('Downloaded file is not a PDF')
    parsed = extract(path)
    if parsed['pdf_publication_year'] != paper['year']:
        raise ValueError('Index and PDF publication years disagree')
    # Select continuous runs of eligible prose; never join text across an
    # omitted heading, formula, caption or rejected paragraph.
    runs = []; segment = []
    for index, paragraph in enumerate(parsed['paragraphs']):
        if paragraph['eligible_prose']:
            segment.append(index)
        else:
            if len(segment) >= minimum_paragraphs:
                runs.append(segment)
            segment = []
    if len(segment) >= minimum_paragraphs:
        runs.append(segment)
    document = {**paper, **parsed, 'pdf_sha256': sha(path), 'pdf_path': str(path.resolve())}
    write_json(path.with_suffix('.json'), document)
    return document, runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--minimum-paragraphs', type=int, default=6)
    parser.add_argument('--exclude', type=Path, help='Existing full-record JSONL; exclude its papers and exact source texts.')
    parser.add_argument('--source-cache', type=Path, help='Reuse cached PDF/XML/index files without changing the cache.')
    args = parser.parse_args()
    run = args.run; raw = run / 'sources/jmlr/raw'; raw.mkdir(parents=True, exist_ok=True)
    if (run / 'plan.json').exists():
        raise ValueError('Refusing to replace a frozen plan')
    config = load_config(args.config)
    config['dataset']['seed'] = args.seed
    names = [backend['name'] for backend in config['generators']]
    if args.count < 1 or args.count % len(names):
        raise ValueError('Count must be positive and divisible by the number of writers')
    old_groups = set(); old_hashes = set()
    if args.exclude:
        from heterogeneous.core import read_jsonl
        for record in read_jsonl(args.exclude):
            old_groups.add(record['source']['group_id'])
            old_hashes.add(normalized_hash(record['source']['text']))
    if args.source_cache:
        for source in args.source_cache.iterdir():
            target = raw/source.name
            if not source.is_file() or target.exists():continue
            if source.suffix=='.json':shutil.copy2(source,target)
            elif source.suffix in ['.pdf','.xml','.html']:target.symlink_to(source.resolve())
    rng = random.Random(args.seed)
    complete_inventory = inventory(raw)
    papers = sorted((paper for paper in complete_inventory if paper['id'] not in old_groups), key=lambda p: p['id'])
    rng.shuffle(papers)
    write_jsonl(run / 'paper-inventory.jsonl', papers)
    selected = []; rejected = []; seen = set()
    # Process fixed batches: reproducibility does not depend on worker timing.
    for offset in range(0, len(papers), 24):
        def work(paper):
            try:
                doc, segments = candidates(paper, raw, args.minimum_paragraphs)
                return paper, doc, segments, None
            except Exception as error:
                return paper, None, [], str(error)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(work, papers[offset:offset+24]))
        for paper, doc, segments, error in results:
            if len(selected) >= args.count:
                break
            if error or not segments:
                rejected.append({'paper_id': paper['id'], 'reason': error or f'No run of {args.minimum_paragraphs} clean prose paragraphs'})
                continue
            rng.shuffle(segments)
            segment = segments[0]
            # Whole paragraphs; select at most ~12k chars without splitting one.
            start = rng.randrange(max(1, len(segment) - args.minimum_paragraphs + 1))
            indices = []; char_count = 0
            for index in segment[start:]:
                length = len(doc['paragraphs'][index]['text'])
                if len(indices) >= args.minimum_paragraphs and char_count + length > 12000:
                    break
                indices.append(index); char_count += length + 2
            text = '\n\n'.join(doc['paragraphs'][index]['text'] for index in indices)
            sid = f"{paper['id']}:paragraphs:{indices[0]}-{indices[-1]+1}"
            if normalized_hash(text) in seen or normalized_hash(text) in old_hashes or not select_blocks(text, config['dataset'], text_hash(sid)):
                rejected.append({'paper_id': paper['id'], 'reason': 'Duplicate or no eligible replacement block'})
                continue
            seen.add(normalized_hash(text))
            name = names[len(selected) % len(names)]
            selected.append({'id': sid, 'group_id': paper['id'], 'reference': paper['reference'],
                'text': text, 'author_id': paper['authors'], 'genre': 'machine_learning_research', 'language': 'en',
                'dataset': 'jmlr_pre2015', 'title': paper['title'], 'generator_names': [name],
                'human_verified': False, 'authorship': 'human_candidate', 'binary_target': None,
                'license': 'JMLR author copyright; journal CC-BY policy recorded, historical paper-specific license unconfirmed',
                'provenance': 'Original journal PDF publication year agrees with dated volume index; deterministic extraction without a model.',
                'human_origin_basis': {'kind': 'historical_journal_paper_pre2015', 'publication_year': paper['year'],
                    'pdf_publication_year': doc['pdf_publication_year'], 'pdf_metadata': doc['pdf_metadata'],
                    'pdf_sha256': doc['pdf_sha256'], 'certainty': 'strong historical evidence, not independently certified'},
                'upstream_metadata': {k: v for k, v in doc.items() if k != 'paragraphs'},
                'paragraph_indices': indices,
                'preprocessing': {'method': 'Poppler layout; paragraph indentation/gaps; join wrapped lines; remove line-end hyphen before lowercase continuation; NFC',
                    'selected_paragraphs': [doc['paragraphs'][index] for index in indices],
                    'no_model_used': True, 'excluded_material': 'Headers, footers, headings, equations/captions and paragraphs failing prose checks'}})
        write_json(run / 'collection-progress.json', {'selected': len(selected), 'target': args.count,
            'processed_candidates': min(offset+24, len(papers)), 'rejected': len(rejected)})
        print(json.dumps({'selected': len(selected), 'processed': offset+len(results), 'rejected': len(rejected)}), flush=True)
        if len(selected) >= args.count:
            break
    write_jsonl(run / 'input.jsonl', selected)
    write_json(run / 'paper-exclusions.json', rejected)
    if len(selected) != args.count:
        raise ValueError(f'Only {len(selected)} eligible papers, expected {args.count}')
    rng.shuffle(selected); write_jsonl(run / 'input.jsonl', selected)
    plan = make_plan(run / 'input.jsonl', config)
    assert not plan['skipped']
    assert not {r['group_id'] for r in selected} & old_groups
    assert Counter(r['generator_names'][0] for r in selected)=={name:args.count//len(names) for name in names}
    save_plan(plan, run)
    write_json(run / 'source-audit.json', {'plan_id': plan['id'], 'sources': len(selected),
        'unique_papers': len({r['group_id'] for r in selected}),
        'models': dict(Counter(r['generator_names'][0] for r in selected)),
        'years': dict(Counter(r['human_origin_basis']['publication_year'] for r in selected)),
        'seed': args.seed, 'input_sha256': sha(run / 'input.jsonl'),
        'excluded_previous_papers':sum(p['id'] in old_groups for p in complete_inventory),
        'previous_dataset_sha256':sha(args.exclude) if args.exclude else None,
        'collection_script_sha256': sha(Path(__file__)), 'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'sampling': 'Seeded shuffled eligible JMLR papers from volumes 1–15; one continuous prose excerpt per distinct paper; model assignment round-robin before final shuffle.',
        'checks': ['Publication year <2015 in both journal index and PDF', 'PDF/XML hashes retained',
            'No model used for text recovery', f'Continuous run of ≥{args.minimum_paragraphs} eligible prose paragraphs',
            'No normalized exact duplicate excerpts', 'One paper per source; group split in frozen plan',
            'Previously used papers and exact source texts excluded when a previous dataset is supplied'],
        'limitations': ['Historical dates support human origin but do not certify every word',
            'PDF paragraph/dehyphenation recovery requires visual sample review',
            'Prose excerpts exclude most mathematics; not a full-paper corpus',
            'Current retrieved PDF revision is not independently archived/datetested'],
        'license_policy_url': 'https://jmlr.org/author-info.html'})


if __name__ == '__main__':
    main()
