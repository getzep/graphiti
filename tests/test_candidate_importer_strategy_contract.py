"""Contract tests for candidate importer lane strategy coverage.

Guardrail: every candidate-generating lane must have explicit importer strategy
coverage via lane kind (Entity vs OM-native).
"""

import pytest

from scripts.candidate_importer_strategy import (
    CANDIDATE_LANE_KIND_BY_GROUP,
    LANE_KIND_ENTITY,
    LANE_KIND_OM_NATIVE,
    QUERY_STRATEGY_ENTITY_RELATES_TO,
    QUERY_STRATEGY_OM_NATIVE,
    candidate_generating_lanes_from_policy,
    missing_candidate_lane_query_paths,
    resolve_query_strategy_for_graph,
    strategy_for_candidate_lane,
)


def test_candidate_generating_lanes_have_query_path_contract() -> None:
    """Policy candidate lanes must all be strategy-covered."""
    lanes = candidate_generating_lanes_from_policy()
    missing = missing_candidate_lane_query_paths(lanes)
    assert missing == [], (
        'candidate-generating policy contains lane(s) with no importer strategy: '
        f'{missing}'
    )


def test_live_policy_covers_om_and_entity_lane_types() -> None:
    """Current candidate lanes exercise both OM-native and Entity paths."""
    lanes = set(candidate_generating_lanes_from_policy())

    assert 's1_observational_memory' in lanes
    assert strategy_for_candidate_lane('s1_observational_memory') == QUERY_STRATEGY_OM_NATIVE
    assert CANDIDATE_LANE_KIND_BY_GROUP['s1_observational_memory'] == LANE_KIND_OM_NATIVE

    for lane in ('s1_sessions_main', 's1_chatgpt_history'):
        assert lane in lanes
        assert CANDIDATE_LANE_KIND_BY_GROUP[lane] == LANE_KIND_ENTITY
        assert strategy_for_candidate_lane(lane) == QUERY_STRATEGY_ENTITY_RELATES_TO


def test_unknown_candidate_lane_fails_contract_check() -> None:
    """New candidate lanes must be wired explicitly before CI goes green."""
    lanes = ['s1_sessions_main', 's1_observational_memory', 's1_future_lane']
    missing = missing_candidate_lane_query_paths(lanes)
    assert missing == ['s1_future_lane']


def test_strategy_for_candidate_lane_raises_for_unknown_lane() -> None:
    with pytest.raises(KeyError):
        strategy_for_candidate_lane('s1_future_lane')


def test_non_candidate_lane_uses_entity_strategy_fallback() -> None:
    assert resolve_query_strategy_for_graph('s1_curated_refs') == QUERY_STRATEGY_ENTITY_RELATES_TO
