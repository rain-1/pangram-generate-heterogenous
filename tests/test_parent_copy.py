import json

import pytest

from heterogeneous.core import copy_fraction, text_hash
from heterogeneous.parent_copy import check_parent, load_policy, parent_fraction


def test_detects_memorized_chapter_outside_selected_excerpt(tmp_path):
    excerpt = 'Newspapers discussed the recent city election and the voting arrangements.'
    memorized = "She couldn't find the lantern, and the narrow passage grew darker with every step. Beyond the locked door, she heard the quiet sound of someone turning the pages of a book."
    parent = tmp_path / 'book.txt'
    text = excerpt + '\n\n' + memorized.replace("couldn't", 'couldn’t')
    parent.write_text(text)
    reference = {'path': str(parent), 'sha256': text_hash(text)}
    assert copy_fraction(memorized, excerpt, 8) == 0
    assert parent_fraction(memorized, reference, 8) == 1
    with pytest.raises(ValueError, match='full-parent wording'):
        check_parent(memorized, 'source', {'source': reference}, {'copy_ngram':8, 'max_copy_fraction':.15})


def test_parent_reference_tampering_is_detected(tmp_path):
    parent = tmp_path / 'book.txt'
    parent.write_text('A dated historical document was archived with its original words intact.')
    reference = {'path': str(parent), 'sha256': text_hash(parent.read_text())}
    assert parent_fraction('Fresh prose about a different subject entirely.', reference, 8) == 0
    parent.write_text('The original file has been replaced with a different and longer document.')
    with pytest.raises(ValueError, match='integrity mismatch'):
        parent_fraction('Fresh prose about a different subject entirely.', reference, 8)


def test_parent_policy_cannot_apply_to_another_plan(tmp_path):
    (tmp_path / 'parent-copy-policy.json').write_text(json.dumps({'plan_id':'old', 'sources':{}}))
    with pytest.raises(ValueError, match='another plan'):
        load_policy(tmp_path, {'id':'new', 'sources':[]})
