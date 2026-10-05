"""Pinned full-parent copy checks; parent text is never sent in a prompt."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import unicodedata

from .core import words


def normalized_words(text):
    text = unicodedata.normalize('NFKC', text).translate(str.maketrans({'’': "'", '‘': "'", '‐': '-', '‑': '-'}))
    return [word.casefold() for word in words(text)]


@lru_cache(maxsize=256)
def parent_ngrams(path, expected_hash, n, size, mtime):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError('Full-parent copy reference integrity mismatch')
    tokens = normalized_words(data.decode('utf-8-sig'))
    return frozenset(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def load_policy(run, plan):
    path = run / 'parent-copy-policy.json'
    if not path.exists():
        return {}
    policy = json.loads(path.read_text())
    if policy['plan_id'] != plan['id']:
        raise ValueError('Full-parent copy policy belongs to another plan')
    if set(policy['sources']) - {s['id'] for s in plan['sources']}:
        raise ValueError('Unknown source in full-parent copy policy')
    return policy['sources']


def parent_fraction(text, reference, n):
    path = Path(reference['path']); stat = path.stat()
    source = parent_ngrams(str(path), reference['sha256'], n, stat.st_size, stat.st_mtime_ns)
    tokens = normalized_words(text)
    covered = set()
    for i in range(len(tokens)-n+1):
        if tuple(tokens[i:i+n]) in source:
            covered.update(range(i,i+n))
    return len(covered)/len(tokens) if tokens else 0.0


def check_parent(text, source_id, policy, settings):
    reference = policy.get(source_id)
    if not reference:
        return {}
    fraction = parent_fraction(text, reference, settings['copy_ngram'])
    if fraction > settings['max_copy_fraction']:
        raise ValueError(f"passage copies too much full-parent wording ({fraction:.1%}; maximum {settings['max_copy_fraction']:.1%}); compose entirely new prose and do not quote or reconstruct any known chapter or published passage, including material outside the selected excerpt")
    return {'full_parent_copy_fraction': fraction, 'full_parent_sha256': reference['sha256']}
