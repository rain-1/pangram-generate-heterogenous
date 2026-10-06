"""Inspect full records and verify spans after extracting the release ZIP.

    python examples/load_and_inspect.py --verify-all
    python examples/load_and_inspect.py --model opus3 --limit 2

Requires only the Python standard library. Model names include haiku/sonnet/opus/opus3/gpt_sol/gpt_luna.
"""
import argparse
import hashlib
import json
from pathlib import Path


def validate(r):
    text = r['text']
    assert hashlib.sha256(text.encode('utf-8')).hexdigest() == r['text_sha256']
    assert r['offset_unit'] == 'unicode_codepoint' and r['end_exclusive'] is True
    source = r['source']['text']
    assert hashlib.sha256(source.encode('utf-8')).hexdigest() == r['source']['sha256']
    cursor = 0
    for s in r['spans']:
        assert s['start'] == cursor and s['end'] > cursor
        if s['label'] == 'human':
            assert text[s['start']:s['end']] == source[s['source_start']:s['source_end']]
        cursor = s['end']
    assert cursor == len(text)
    pieces = []; cursor = 0
    for replacement in sorted(r['replacements'], key=lambda x: x['source_start']):
        pieces.extend([source[cursor:replacement['source_start']], replacement['generated_text']])
        cursor = replacement['source_end']
    pieces.append(source[cursor:])
    assert ''.join(pieces) == text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'records/all.jsonl')
    p.add_argument('--verify-all', action='store_true')
    p.add_argument('--model', choices=['haiku','sonnet','opus','opus3','gpt_sol','gpt_luna'])
    p.add_argument('--limit', type=int, default=1)
    a = p.parse_args(); checked = shown = 0
    with a.input.open(encoding='utf-8') as handle:
        for line in handle:
            r=json.loads(line)
            if a.verify_all: validate(r); checked += 1
            writer=next((s['author'].get('backend') for s in r['spans'] if s['label']=='ai'),None)
            if shown < a.limit and writer and (a.model is None or writer==a.model):
                print(r['id'],r['split'],writer,r['source'].get('title'))
                for s in r['spans']:
                    print(s['label'],f'[{s["start"]}:{s["end"]}]',r['text'][s['start']:s['end']][:150])
                shown += 1
    if a.verify_all: print(f'Passed hash, span and reconstruction checks for {checked} records.')


if __name__ == '__main__': main()
