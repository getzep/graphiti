"""Lane-type-aware query strategy contract for candidate importers.

This module intentionally captures only generic dispatch/coverage guardrails:
- candidate lane -> lane kind (Entity vs OM-native)
- lane kind -> importer query strategy

It does *not* embed private anchors, subject-ID schemes, or policy-specific
mapping rules.
"""

from __future__ import annotations

from collections.abc import Iterable

from truth.candidates import LANE_CANDIDATES_ELIGIBLE

LANE_KIND_ENTITY = 'entity'
LANE_KIND_OM_NATIVE = 'om_native'

QUERY_STRATEGY_ENTITY_RELATES_TO = 'entity_relates_to'
QUERY_STRATEGY_OM_NATIVE = 'om_native'

CANDIDATE_LANE_KIND_BY_GROUP: dict[str, str] = {
    's1_sessions_main': LANE_KIND_ENTITY,
    's1_chatgpt_history': LANE_KIND_ENTITY,
    's1_observational_memory': LANE_KIND_OM_NATIVE,
    's1_memory_day1': LANE_KIND_ENTITY,
}

QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND: dict[str, str] = {
    LANE_KIND_ENTITY: QUERY_STRATEGY_ENTITY_RELATES_TO,
    LANE_KIND_OM_NATIVE: QUERY_STRATEGY_OM_NATIVE,
}


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or '').strip()
        if not normalized or normalized in seen:
            continue
        out.append(normalized)
        seen.add(normalized)
    return out


def candidate_generating_lanes_from_policy() -> list[str]:
    """Return candidate-generating lanes from the unified lane policy contract."""
    return sorted(_ordered_unique(LANE_CANDIDATES_ELIGIBLE))


def missing_candidate_lane_query_paths(candidate_lanes: Iterable[str]) -> list[str]:
    """Return candidate lanes that lack lane-kind or strategy wiring."""
    missing: list[str] = []
    for lane in _ordered_unique(candidate_lanes):
        lane_kind = CANDIDATE_LANE_KIND_BY_GROUP.get(lane)
        if lane_kind is None:
            missing.append(lane)
            continue
        if lane_kind not in QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND:
            missing.append(lane)
    return missing


def validate_candidate_lane_query_contract(candidate_lanes: Iterable[str]) -> None:
    """Raise when candidate lanes are missing importer strategy coverage."""
    missing = missing_candidate_lane_query_paths(candidate_lanes)
    if missing:
        raise ValueError(
            'Missing importer query strategy for candidate-generating lane(s): '
            f'{sorted(missing)}. '
            'Update CANDIDATE_LANE_KIND_BY_GROUP and/or '
            'QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND.'
        )


def strategy_for_candidate_lane(group_id: str) -> str:
    """Resolve importer strategy for a candidate-generating lane."""
    lane_kind = CANDIDATE_LANE_KIND_BY_GROUP.get(group_id)
    if lane_kind is None:
        raise KeyError(f'Unknown candidate lane: {group_id}')

    strategy = QUERY_STRATEGY_BY_CANDIDATE_LANE_KIND.get(lane_kind)
    if strategy is None:
        raise KeyError(f'No query strategy for candidate lane kind: {lane_kind}')

    return strategy


def resolve_query_strategy_for_graph(group_id: str) -> str:
    """Resolve query strategy for any graph lane.

    Candidate-generating lanes use explicit lane-kind dispatch.
    Non-candidate lanes default to the generic Entity path.
    """
    if group_id in LANE_CANDIDATES_ELIGIBLE:
        return strategy_for_candidate_lane(group_id)
    return QUERY_STRATEGY_ENTITY_RELATES_TO
