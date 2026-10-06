"""Synthetic short-block retries retain the normal acceptance checks and cache."""
import pytest

from heterogeneous.config import DEFAULT_DATASET
from heterogeneous.core import text_hash
from scripts.retry_span_pilot import RetryRunner


def test_compact_retry_records_its_prompt_and_still_rejects_an_uncompressed_brief(tmp_path):
    source={'id':'fixture','text':'A town has a broken fountain and a visiting magician.\n\nThe magician fails to restore the water supply.\n\nA newcomer succeeds and the villagers rejoice.'}
    block={'id':'fixture-block','start':0,'end':len(source['text']),'characters':len(source['text'])}
    backend={'name':'briefs','kind':'codex','model':'synthetic-test','concurrency':1,'retries':0}
    plan={'id':'fixture-plan','sources':[source],'config':{'dataset':{**DEFAULT_DATASET,'summary_min_sentences':2,'quality_attempts':2},'summarizer':backend}}
    runner=RetryRunner(tmp_path,plan,compact_under=240)
    calls=[]
    def completion(prompt_config,prompt):
        calls.append(prompt)
        text=('This description is deliberately too long to condense the very short source passage, with excess background that the model should not need to repeat. It also gives more unnecessary detail about the story.' if len(calls)==1
              else 'A specialist fails at a repair. An outsider restores the service.')
        return {'text':text,'provenance':{'prompt_sha256':text_hash(prompt)}}
    runner.completion=completion
    accepted=runner.summary({'source_id':'fixture'},block)
    assert accepted['quality_attempt']==2
    assert len(accepted['text'])<block['characters']
    assert 'description must be shorter' in accepted['rejected_attempts'][0]
    assert 'exceptionally short source block' in accepted['prompt']
    assert 'Previous attempt failed' in accepted['prompt']
    assert len(calls)==2
    assert runner.summary({'source_id':'fixture'},block)==accepted
    assert len(calls)==2
def test_author_citations_do_not_inflate_brief_sentence_count():
    from heterogeneous.core import sentence_spans

    text = ('Ye et al. (2010) described a classifier based on call features. '
            'It first finds candidates and then improves precision. '
            'Dai et al. (2009) instead collected runtime features for classification.')
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == [
        'Ye et al. (2010) described a classifier based on call features.',
        'It first finds candidates and then improves precision.',
        'Dai et al. (2009) instead collected runtime features for classification.',
    ]



