import json
from pathlib import Path
import pytest
Draft202012Validator = pytest.importorskip("jsonschema", reason="JSON Schema conformance requires the optional jsonschema test dependency").Draft202012Validator
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'schemas/nl-plan.schema.json').read_text())
VALIDATOR = Draft202012Validator(SCHEMA)

def test_profile_and_envelope():
    Draft202012Validator.check_schema(SCHEMA)
    profile = json.loads((ROOT / 'standard/semantic-profile.json').read_text())
    assert not profile['compilation_executes']
    for status, payload in [('clarify', {'question': 'Ποιο αρχείο;'}), ('unsupported', {'reason': 'Brak operacji.'})]:
        VALIDATOR.validate({'schema': profile['envelope'], 'status': status, **payload})
    call = {'kind': 'call', 'operation': 'proc://media/images/convert/v1', 'arguments': {'source': 'zażółć.svg'}, 'digest': 'a' * 64}
    for plan in [call, {'kind': 'sequence', 'steps': [call, call]}]:
        VALIDATOR.validate({'schema': profile['envelope'], 'status': 'ok', 'plan': plan})

@pytest.mark.parametrize('value', [
    {'schema': 'wellmanifest.nl-plan/v1', 'status': 'ok', 'plan': {'kind': 'call', 'operation': 'invented', 'arguments': {}, 'digest': 'a'*64}},
    {'schema': 'wellmanifest.nl-plan/v1', 'status': 'clarify', 'question': 'x', 'plan': {}},
    {'schema': 'wellmanifest.nl-plan/v1', 'status': 'unsupported', 'reason': ''},
    {'schema': 'wellmanifest.nl-plan/v1', 'status': 'ok', 'plan': {'kind': 'sequence', 'steps': []}},
    {'schema': 'wellmanifest.nl-plan/v1', 'status': 'ok', 'plan': {'kind': 'condition'}},
])
def test_invalid_structural_plans(value):
    assert list(VALIDATOR.iter_errors(value))
