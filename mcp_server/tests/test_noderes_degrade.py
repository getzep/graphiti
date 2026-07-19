"""NodeResolutions must degrade, not drop.

The dedupe LLM intermittently answers with a bare NodeDuplicate rather than the wrapper object.
With `entity_resolutions` required that raised inside add_episode and the episode was lost with no
retry (measured 2026-07-16: 2 of 8 episodes gone). Losing a dedup round is recoverable; losing the
episode is not.
"""

import pytest
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate, NodeResolutions
from pydantic import ValidationError


def test_bare_node_duplicate_response_does_not_raise():
    """The exact shape that was killing episodes in production."""
    llm_response = {'id': 0, 'name': 'concurrent worker test', 'duplicate_candidate_id': -1}

    resolutions = NodeResolutions(**llm_response)

    assert resolutions.entity_resolutions == [], 'omitted resolutions must mean "resolve nothing"'


def test_empty_response_does_not_raise():
    assert NodeResolutions().entity_resolutions == []
    assert NodeResolutions(**{}).entity_resolutions == []


def test_well_formed_response_still_parses():
    """Relaxing the field must not weaken the happy path."""
    llm_response = {
        'entity_resolutions': [
            {'id': 0, 'name': 'Jizo', 'duplicate_candidate_id': -1},
            {'id': 1, 'name': 'ChaiKlang', 'duplicate_candidate_id': 3},
        ]
    }

    resolutions = NodeResolutions(**llm_response)

    assert len(resolutions.entity_resolutions) == 2
    assert resolutions.entity_resolutions[0].name == 'Jizo'
    assert resolutions.entity_resolutions[1].duplicate_candidate_id == 3


def test_malformed_resolution_entries_still_rejected():
    """Only the wrapper is relaxed — a resolution missing its own required fields is still a bug
    we want to hear about, not silently coerce."""
    with pytest.raises(ValidationError):
        NodeResolutions(entity_resolutions=[{'id': 0}])  # no name / duplicate_candidate_id


def test_node_duplicate_fields_unchanged():
    nd = NodeDuplicate(id=2, name='SomTor', duplicate_candidate_id=-1)
    assert (nd.id, nd.name, nd.duplicate_candidate_id) == (2, 'SomTor', -1)
