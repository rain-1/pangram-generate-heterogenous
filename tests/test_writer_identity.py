"""Audit requested-only attribution without inventing a served model identity."""
import pytest

pytest.importorskip('jsonschema')
from scripts.audit_span_pilot import validate_writer_identity


@pytest.mark.parametrize('reported,status',[(None,'requested_only'),('gpt-6.1-sol','reported')])
def test_codex_accepts_explicit_request_with_honest_identity_status(reported,status):
    validate_writer_identity({'kind':'codex','model':'gpt-6.1-sol'},
        {'requested_model':'gpt-6.1-sol','reported_model':reported,'metadata':{'model_identity_status':status}})


@pytest.mark.parametrize('kind,requested,reported,status',[
    ('codex','gpt-6.1-sol',None,'reported'),
    ('codex','gpt-6-luna',None,'requested_only'),
    ('codex','gpt-6.1-sol','other-model','reported'),
    ('codex','gpt-6.1-sol','gpt-6.1-sol','requested_only'),
    ('claude','gpt-6.1-sol',None,'requested_only'),
])
def test_missing_or_mismatched_attribution_is_rejected(kind,requested,reported,status):
    with pytest.raises(AssertionError):
        validate_writer_identity({'kind':kind,'model':'gpt-6.1-sol'},
            {'requested_model':requested,'reported_model':reported,'metadata':{'model_identity_status':status}})
