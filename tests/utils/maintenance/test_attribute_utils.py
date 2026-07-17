import logging

import pytest
from pydantic import BaseModel, Field

from graphiti_core.utils.maintenance.attribute_utils import (
    DEFAULT_ATTRIBUTE_MAX_LENGTH,
    LIST_TOTAL_LENGTH_MULTIPLIER,
    apply_capped_attributes,
    cap_string_attributes,
)


class _Person(BaseModel):
    phones: str | None = Field(
        default=None,
        description='Phone numbers, comma-separated, with type tag where known',
    )
    industry: str | None = Field(default=None, description='Industry sector')
    description: str | None = Field(
        default=None,
        max_length=2000,
        description='Long-form description',
    )
    aliases: list[str] | None = Field(default=None, description='Alternate names')


def test_short_string_passes_through():
    response = {'phones': '415-555-0142', 'industry': 'SaaS'}
    kept, dropped = cap_string_attributes(response, _Person)
    assert kept == response
    assert dropped == set()


def test_overlong_string_dropped():
    rant = 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH * 10)
    response = {'phones': rant, 'industry': 'SaaS'}
    kept, dropped = cap_string_attributes(response, _Person)
    assert 'phones' not in kept, 'over-cap field must be dropped, not truncated'
    assert kept['industry'] == 'SaaS'
    assert dropped == {'phones'}


def test_explicit_max_length_honored():
    long_desc = 'a' * 1500
    kept, dropped = cap_string_attributes({'description': long_desc}, _Person)
    assert kept['description'] == long_desc
    assert dropped == set()


def test_explicit_max_length_enforced_when_exceeded():
    kept, dropped = cap_string_attributes({'description': 'a' * 2500}, _Person)
    assert 'description' not in kept
    assert dropped == {'description'}


def test_non_string_fields_pass_through():
    class Mixed(BaseModel):
        amount: float | None = None
        notes: str | None = None

    response = {'amount': 42.5, 'notes': 'short note'}
    kept, dropped = cap_string_attributes(response, Mixed)
    assert kept == response
    assert dropped == set()


def test_list_with_one_overlong_element_drops_field():
    bleed = 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH + 1)
    response = {'aliases': ['Sammy', bleed, 'Sam R.']}
    kept, dropped = cap_string_attributes(response, _Person)
    assert 'aliases' not in kept
    assert dropped == {'aliases'}


def test_list_aggregate_length_caught():
    """Many just-under-cap elements collectively exceed max_len * multiplier."""
    item = 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH - 1)
    response = {'aliases': [item] * (LIST_TOTAL_LENGTH_MULTIPLIER + 2)}
    kept, dropped = cap_string_attributes(response, _Person)
    assert 'aliases' not in kept
    assert dropped == {'aliases'}


def test_list_short_passes_through():
    response = {'aliases': ['Sammy', 'Sam R.', 'S. Rivera']}
    kept, dropped = cap_string_attributes(response, _Person)
    assert kept == response
    assert dropped == set()


def test_required_field_overcap_kept_with_warning(caplog):
    """A required string field with an over-cap value cannot be dropped without
    failing Pydantic validation; the helper retains the bleed and logs louder."""

    class StrictPerson(BaseModel):
        full_name: str = Field(..., description='Full legal name')

    bleed = 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH + 50)
    with caplog.at_level(logging.WARNING):
        kept, dropped = cap_string_attributes(
            {'full_name': bleed},
            StrictPerson,
            prompt_name='extract_nodes.extract_attributes',
            entity_uuid='abc-123',
            group_id='tenant-x',
        )
    # Bleed is retained, NOT dropped — Pydantic validation would otherwise fail.
    assert kept == {'full_name': bleed}
    assert dropped == set()
    matching = [
        rec for rec in caplog.records if 'attribute_length_cap_skipped_required' in rec.message
    ]
    assert matching, 'expected a warning about the required-field carve-out'
    assert all(rec.levelno == logging.WARNING for rec in matching)


def test_logs_info_on_drop_with_uuid_and_group_id_no_pii(caplog):
    response = {'phones': 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH + 1)}
    with caplog.at_level(logging.INFO):
        cap_string_attributes(
            response,
            _Person,
            prompt_name='extract_nodes.extract_attributes',
            entity_uuid='abc-123-uuid',
            group_id='tenant-x',
        )
    matching = [rec for rec in caplog.records if 'attribute_length_cap_exceeded' in rec.message]
    assert matching, 'expected an info-level log on cap drop'
    assert all(rec.levelno == logging.INFO for rec in matching)
    assert any(
        'phones' in rec.message
        and 'abc-123-uuid' in rec.message
        and 'tenant-x' in rec.message
        and 'reason=per_item' in rec.message
        for rec in matching
    )
    assert not any('Sam Rivera' in rec.message for rec in matching)


def test_log_aggregate_trigger_reports_aggregate_length_and_cap(caplog):
    """When list-aggregate fires, length= must be the total (not max element)
    and cap= must be the aggregate cap, so DataDog operators see the breach
    directly instead of the misleading length=under-cap cap=per-item view."""
    item = 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH - 1)
    items = [item] * (LIST_TOTAL_LENGTH_MULTIPLIER + 2)
    expected_total = sum(len(i) for i in items)
    expected_cap = DEFAULT_ATTRIBUTE_MAX_LENGTH * LIST_TOTAL_LENGTH_MULTIPLIER
    with caplog.at_level(logging.INFO):
        cap_string_attributes(
            {'aliases': items},
            _Person,
            prompt_name='extract_nodes.extract_attributes',
            entity_uuid='node-uuid',
            group_id='tenant-x',
        )
    matching = [rec for rec in caplog.records if 'attribute_length_cap_exceeded' in rec.message]
    assert len(matching) == 1
    msg = matching[0].message
    assert 'reason=aggregate' in msg
    assert f'length={expected_total}' in msg
    assert f'cap={expected_cap}' in msg
    # Sanity: aggregate length must exceed the aggregate cap (otherwise the test
    # is wrong, not the helper).
    assert expected_total > expected_cap


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv('GRAPHITI_ATTRIBUTE_MAX_LENGTH', '50')
    long_value = 'x' * 100
    kept, dropped = cap_string_attributes({'phones': long_value}, _Person)
    assert 'phones' not in kept
    assert dropped == {'phones'}


def test_env_var_invalid_falls_back_to_default(monkeypatch, caplog):
    monkeypatch.setenv('GRAPHITI_ATTRIBUTE_MAX_LENGTH', 'not-a-number')
    short_value = 'x' * 100
    with caplog.at_level(logging.WARNING):
        kept, dropped = cap_string_attributes({'phones': short_value}, _Person)
    assert kept == {'phones': short_value}
    assert dropped == set()


def test_env_var_explicit_max_length_still_wins(monkeypatch):
    monkeypatch.setenv('GRAPHITI_ATTRIBUTE_MAX_LENGTH', '50')
    long_desc = 'a' * 1500
    kept, dropped = cap_string_attributes({'description': long_desc}, _Person)
    assert kept == {'description': long_desc}
    assert dropped == set()


@pytest.mark.parametrize('bad_value', ['0', '-10', '   ', ''])
def test_env_var_non_positive_or_blank_falls_back(monkeypatch, bad_value):
    monkeypatch.setenv('GRAPHITI_ATTRIBUTE_MAX_LENGTH', bad_value)
    kept, dropped = cap_string_attributes({'phones': 'x' * 100}, _Person)
    assert kept == {'phones': 'x' * 100}
    assert dropped == set()


def test_invalid_env_warning_dedups_across_calls(monkeypatch, caplog):
    from graphiti_core.utils.maintenance import attribute_utils

    monkeypatch.setattr(attribute_utils, '_warned_invalid_env', set())
    monkeypatch.setenv('GRAPHITI_ATTRIBUTE_MAX_LENGTH', 'garbage')

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            cap_string_attributes({'phones': 'x' * 50}, _Person)

    invalid_warnings = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING and 'GRAPHITI_ATTRIBUTE_MAX_LENGTH' in rec.message
    ]
    assert len(invalid_warnings) == 1


# --- apply_capped_attributes (unified node/edge merge helper) -----------------


def test_apply_overlay_keeps_prior_for_omitted_and_dropped_fields():
    prior = {'phones': '415-555-0142', 'industry': 'Software'}
    bleed = 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH + 1)
    llm = {'phones': bleed}  # industry omitted; phones over-cap
    merged, dropped = apply_capped_attributes(llm, _Person, prior, merge_mode='overlay')
    assert merged['phones'] == '415-555-0142'  # cap-dropped; prior preserved
    assert merged['industry'] == 'Software'  # LLM-omitted; prior preserved
    assert dropped == {'phones'}


def test_apply_overlay_legitimate_update_passes_through():
    prior = {'phones': '415-555-0142', 'industry': 'Software'}
    llm = {'industry': 'SaaS'}
    merged, dropped = apply_capped_attributes(llm, _Person, prior, merge_mode='overlay')
    assert merged == {'phones': '415-555-0142', 'industry': 'SaaS'}
    assert dropped == set()


def test_apply_replace_clears_omitted_but_preserves_dropped():
    prior = {'phones': '415-555-0142', 'industry': 'Software'}
    bleed = 'x' * (DEFAULT_ATTRIBUTE_MAX_LENGTH + 1)
    llm = {'phones': bleed}  # industry omitted; phones over-cap
    merged, dropped = apply_capped_attributes(llm, _Person, prior, merge_mode='replace')
    assert merged['phones'] == '415-555-0142'  # cap-dropped; prior restored
    assert 'industry' not in merged  # LLM-omitted; cleared
    assert dropped == {'phones'}


def test_apply_replace_legitimate_update_replaces_prior_and_clears_omitted():
    prior = {'phones': '415-555-0142', 'industry': 'Software'}
    llm = {'industry': 'SaaS'}
    merged, dropped = apply_capped_attributes(llm, _Person, prior, merge_mode='replace')
    # phones was omitted by the LLM → cleared per replace semantics.
    assert merged == {'industry': 'SaaS'}
    assert dropped == set()
