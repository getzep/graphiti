"""Contract tests for hardened sessions ontology lanes.

These tests ensure the PUBLIC ontology config keeps the finalized sessions schema
for production and FR-11 pilot session lanes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'extraction_ontologies.yaml'
ROOT_CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'extraction_ontologies.yaml'

TARGET_LANES = [
    's1_sessions_main',
    's1_chatgpt_history',
    's1_pilot_fr11_20260227',
    's1_pilot_fr11_20260228',
    's1_pilot_fr11_20260228_v2',
]

EXPECTED_ENTITY_TYPES = {
    'Person',
    'Organization',
    'Location',
    'Event',
    'Artifact',
    'System',
    'Preference',
    'Requirement',
    'Procedure',
}

EXPECTED_EDGE_TYPES = {
    'PRODUCED',
    'MODIFIED',
    'DEPLOYED_TO',
    'USES',
    'INTEGRATES_WITH',
    'DEPENDS_ON',
    'LOCATED_AT',
    'ATTENDED',
    'PART_OF',
    'PREFERS',
    'REQUIRES',
}

FORBIDDEN_GENERIC_EDGES = {'RELATES_TO', 'MENTIONS', 'DISCUSSED'}


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8')) or {}


def _lane(data: dict, lane_id: str) -> dict:
    lane = data.get(lane_id)
    assert isinstance(lane, dict), f'Missing or invalid lane: {lane_id}'
    return lane


def test_sessions_lanes_are_explicitly_present():
    data = _load_config()
    for lane_id in TARGET_LANES:
        assert lane_id in data, f'Expected explicit ontology block for {lane_id}'


def test_sessions_lanes_use_finalized_schema():
    data = _load_config()

    for lane_id in TARGET_LANES:
        lane = _lane(data, lane_id)

        assert lane.get('extraction_mode') == 'constrained_soft'

        entity_names = {
            str(entry.get('name'))
            for entry in lane.get('entity_types', [])
            if isinstance(entry, dict)
        }
        assert entity_names == EXPECTED_ENTITY_TYPES

        edge_names = {
            str(entry.get('name'))
            for entry in lane.get('relationship_types', [])
            if isinstance(entry, dict)
        }
        assert edge_names == EXPECTED_EDGE_TYPES
        assert not (edge_names & FORBIDDEN_GENERIC_EDGES)


def test_sessions_lanes_prompt_guidance_includes_hardening_rules():
    data = _load_config()

    for lane_id in TARGET_LANES:
        lane = _lane(data, lane_id)
        guidance = str(lane.get('intent_guidance') or '')
        lower = guidance.lower()

        # No generic edge families.
        assert 'relates_to' in lower
        assert 'mentions' in lower
        assert 'discussed' in lower

        # Floating-node extraction rule.
        assert 'floating' in lower
        assert 'mention-only' in lower or 'mention only' in lower

        # Guardrails to suppress taxonomy spam.
        for concept in ('preference', 'requirement', 'procedure', 'event'):
            assert concept in lower

        # Required type aliases.
        compact = lower.replace(' ', '')
        assert 'agent->person' in compact
        assert 'task/ticket->artifact' in compact
        assert 'environment->system' in compact


def test_top_level_and_mcp_server_ontology_files_are_identical():
    assert ROOT_CONFIG_PATH.exists(), f'Missing top-level config: {ROOT_CONFIG_PATH}'
    assert CONFIG_PATH.exists(), f'Missing mcp_server config: {CONFIG_PATH}'
    assert ROOT_CONFIG_PATH.read_text(encoding='utf-8') == CONFIG_PATH.read_text(encoding='utf-8')
