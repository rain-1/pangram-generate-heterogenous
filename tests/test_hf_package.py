"""Synthetic fixtures for the released training representation, never corpus data."""
import hashlib

import pytest

pytest.importorskip('pyarrow')
pytest.importorskip('yaml')
pytest.importorskip('jsonschema')

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.package_hf_dataset import VIEW_SCHEMA, flat_record, matches


def fixture(mixed=False):
    original='Retained 🐧 source.'
    author={'type':'ai','backend':'haiku','requested_model':'fixture-writer','reported_model':'fixture-writer',
            'metadata':{'model_identity_status':'reported'}} if mixed else {'type':'non_ai','id':'fixture-author'}
    return {'id':'synthetic-example','split':'test','text':original,'text_sha256':hashlib.sha256(original.encode()).hexdigest(),
            'offset_unit':'unicode_codepoint','end_exclusive':True,'ai_fraction_chars':1.0 if mixed else 0.0,
            'construction':{'plan_id':'fixture-plan','kind':'mixed' if mixed else 'human_control'},
            'source':{'id':'fixture-source','group_id':'fixture-group','dataset':'standardebooks','author_id':'fixture-author',
                      'title':'Synthetic fixture','reference':'test://fixture','license':'test fixture only','human_verified':False,
                      'text':original,'sha256':hashlib.sha256(original.encode()).hexdigest()},
            'spans':[{'start':0,'end':len(original),'label':'ai' if mixed else 'human','source_start':0,'source_end':len(original),
                      'replacement_id':'fixture-replacement' if mixed else None,'author':author}]}


@pytest.mark.parametrize('mixed',[False,True])
def test_released_parquet_omits_redundant_source_hash_and_preserves_text(tmp_path,mixed):
    original=fixture(mixed);flat=flat_record(original,{'fixture-plan':'expansion'})
    assert 'source_text_sha256' not in flat and 'source_text_sha256' not in VIEW_SCHEMA.names
    assert flat['cohort']=='expansion'
    path=tmp_path/'fixture.parquet';pq.write_table(pa.Table.from_pylist([flat],schema=VIEW_SCHEMA),path)
    row=pq.read_table(path).to_pylist()[0]
    assert row==flat and row['source_text']==original['source']['text']
    assert hashlib.sha256(row['source_text'].encode()).hexdigest()==original['source']['sha256']
    assert row['text'][row['spans'][0]['start']:row['spans'][0]['end']]==row['text']
    assert row['generator_backend']==('haiku' if mixed else None)


def test_expansion_subset_does_not_silently_enter_original_or_paper_cohorts():
    mixed=flat_record(fixture(True),{'fixture-plan':'expansion'})
    control=flat_record(fixture(False),{'fixture-plan':'expansion'})
    assert matches(mixed,'mixed') and matches(mixed,'all') and matches(mixed,'expansion_mixed')
    assert not matches(mixed,'original_mixed') and not matches(mixed,'ml_papers_mixed')
    assert matches(control,'human_controls') and not matches(control,'expansion_mixed')


@pytest.mark.parametrize('cohort',['gpt_general','gpt_ml_papers'])
def test_gpt_subsets_keep_requested_identity_and_include_papers_in_paper_view(tmp_path,cohort):
    record=fixture(True)
    author=record['spans'][0]['author']
    author.update(backend='gpt_sol',requested_model='gpt-6.1-sol',reported_model=None,
                  metadata={'model_identity_status':'requested_only'})
    record['source']['dataset']='jmlr_pre2015' if cohort=='gpt_ml_papers' else 'standardebooks'
    row=flat_record(record,{'fixture-plan':cohort})
    path=tmp_path/'gpt.parquet';pq.write_table(pa.Table.from_pylist([row],schema=VIEW_SCHEMA),path)
    loaded=pq.read_table(path).to_pylist()[0]
    assert loaded['requested_model']=='gpt-6.1-sol' and loaded['reported_model'] is None
    assert loaded['model_identity_status']==loaded['spans'][0]['model_identity_status']=='requested_only'
    assert matches(loaded,f'{cohort}_mixed')
    assert matches(loaded,'ml_papers_mixed')==(cohort=='gpt_ml_papers')
    assert not matches(loaded,'original_mixed') and not matches(loaded,'expansion_mixed')


@pytest.mark.parametrize('cohort',['overnight_batch_01','overnight_batch_02','overnight_batch_03'])
def test_overnight_configurations_preserve_cohort_and_paper_membership(cohort):
    original=fixture(True)
    original['source']['dataset']='jmlr_pre2015'
    row=flat_record(original,{'fixture-plan':cohort})
    assert matches(row,'overnight_mixed') and matches(row,f'{cohort}_mixed')
    assert matches(row,'ml_papers_mixed') and matches(row,'mixed')
    assert not matches(row,'original_mixed') and not matches(row,'gpt_ml_papers_mixed')
    other='overnight_batch_02' if cohort=='overnight_batch_01' else 'overnight_batch_01'
    assert not matches(row,f'{other}_mixed')
    control=flat_record(fixture(False),{'fixture-plan':cohort})
    assert not matches(control,'overnight_mixed') and not matches(control,f'{cohort}_mixed')
