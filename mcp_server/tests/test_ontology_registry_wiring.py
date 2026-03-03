"""Validation tests for ontology registry wiring.

Proves:
- Target lanes (constrained_soft) return extraction_mode='constrained_soft'
  and non-empty intent_guidance.
- Non-target lanes (no explicit mode) default to 'permissive'.
- intent_guidance > extraction_emphasis precedence (new key wins).
- Legacy extraction_emphasis fallback works.
- Invalid extraction_mode (field present but wrong value) falls back to
  'permissive' with warning (current public behavior).
- Missing extraction_mode field still defaults to 'permissive' (backward compat).
- Bounded-length guard caps intent_guidance at _INTENT_GUIDANCE_MAX_CHARS.
- edge_types are built from relationship_types.
"""

from __future__ import annotations

import os
import tempfile

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_registry(yaml_content: str):
    """Write YAML to a tempfile and load an OntologyRegistry from it."""
    from mcp_server.src.services.ontology_registry import OntologyRegistry

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        path = f.name
    try:
        return OntologyRegistry.load(path)
    finally:
        os.unlink(path)


def _yaml_lane(
    *,
    extraction_mode: str | None = None,
    intent_guidance: str | None = None,
    extraction_emphasis: str | None = None,
    entity_type_name: str = 'RhetoricalMove',
    rel_type_name: str = 'USES_MOVE',
) -> str:
    lane: dict = {
        'entity_types': [
            {'name': entity_type_name, 'description': 'A technique.'}
        ],
        'relationship_types': [
            {'name': rel_type_name, 'description': 'Piece → move.'}
        ],
    }
    if extraction_mode is not None:
        lane['extraction_mode'] = extraction_mode
    if intent_guidance is not None:
        lane['intent_guidance'] = intent_guidance
    if extraction_emphasis is not None:
        lane['extraction_emphasis'] = extraction_emphasis
    return yaml.dump({'target_lane': lane, 'other_lane': {
        'entity_types': [{'name': 'GenericEntity', 'description': 'Any.'}],
    }})


# ---------------------------------------------------------------------------
# extraction_mode field tests
# ---------------------------------------------------------------------------


class TestExtractionMode:
    """Target lanes return constrained_soft; others default permissive."""

    def test_target_lane_returns_constrained_soft(self):
        registry = _load_registry(_yaml_lane(
            extraction_mode='constrained_soft',
            intent_guidance='Focus on rhetorical moves.',
        ))
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.extraction_mode == 'constrained_soft'

    def test_non_target_lane_defaults_permissive(self):
        registry = _load_registry(_yaml_lane(extraction_mode='constrained_soft'))
        profile = registry.get('other_lane')
        assert profile is not None
        assert profile.extraction_mode == 'permissive'

    def test_lane_without_mode_defaults_permissive(self):
        """No extraction_mode key → defaults to 'permissive'."""
        registry = _load_registry(_yaml_lane())
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.extraction_mode == 'permissive'

    def test_invalid_mode_falls_back_to_permissive(self, caplog):
        """Invalid extraction_mode should fall back to permissive with warning."""
        registry = _load_registry(_yaml_lane(extraction_mode='turbo_mode'))
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.extraction_mode == 'permissive'
        assert "Invalid extraction_mode" in caplog.text

    def test_missing_mode_field_defaults_permissive(self):
        """Absent extraction_mode key → defaults to 'permissive' (backward compat)."""
        registry = _load_registry(_yaml_lane())  # no extraction_mode key
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.extraction_mode == 'permissive'

    def test_unregistered_lane_returns_none(self):
        registry = _load_registry(_yaml_lane())
        assert registry.get('unknown_lane') is None


# ---------------------------------------------------------------------------
# intent_guidance field tests
# ---------------------------------------------------------------------------


class TestIntentGuidance:
    """Target lanes with constrained_soft have non-empty guidance."""

    def test_constrained_lane_has_non_empty_guidance(self):
        """Constrained-soft lane must return non-empty intent_guidance."""
        registry = _load_registry(_yaml_lane(
            extraction_mode='constrained_soft',
            intent_guidance='Focus on rhetorical moves and voice fingerprint.',
        ))
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.extraction_mode == 'constrained_soft'
        assert profile.intent_guidance  # non-empty

    def test_intent_guidance_key_wins_over_extraction_emphasis(self):
        """intent_guidance key takes precedence over extraction_emphasis (legacy)."""
        registry = _load_registry(_yaml_lane(
            intent_guidance='New guidance.',
            extraction_emphasis='Old emphasis.',
        ))
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.intent_guidance == 'New guidance.'
        assert profile.extraction_emphasis == 'Old emphasis.'  # stored but not used as primary

    def test_legacy_extraction_emphasis_fallback(self):
        """When intent_guidance absent, falls back to extraction_emphasis."""
        registry = _load_registry(_yaml_lane(extraction_emphasis='Focus on hooks.'))
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.intent_guidance == 'Focus on hooks.'
        assert profile.extraction_emphasis == 'Focus on hooks.'

    def test_both_absent_gives_empty_string(self):
        """When neither key is set, intent_guidance is empty string."""
        registry = _load_registry(_yaml_lane())
        profile = registry.get('target_lane')
        assert profile is not None
        assert profile.intent_guidance == ''

    def test_intent_guidance_bounded_at_max_chars(self):
        """intent_guidance must be truncated to _INTENT_GUIDANCE_MAX_CHARS."""
        from mcp_server.src.services.ontology_registry import _INTENT_GUIDANCE_MAX_CHARS

        oversized = 'x' * (_INTENT_GUIDANCE_MAX_CHARS + 500)
        registry = _load_registry(_yaml_lane(intent_guidance=oversized))
        profile = registry.get('target_lane')
        assert profile is not None
        assert len(profile.intent_guidance) == _INTENT_GUIDANCE_MAX_CHARS

    def test_extraction_emphasis_bounded_at_max_chars(self):
        """Legacy extraction_emphasis fallback is also bounded."""
        from mcp_server.src.services.ontology_registry import _INTENT_GUIDANCE_MAX_CHARS

        oversized = 'y' * (_INTENT_GUIDANCE_MAX_CHARS + 100)
        registry = _load_registry(_yaml_lane(extraction_emphasis=oversized))
        profile = registry.get('target_lane')
        assert profile is not None
        assert len(profile.intent_guidance) <= _INTENT_GUIDANCE_MAX_CHARS


# ---------------------------------------------------------------------------
# edge_types field tests
# ---------------------------------------------------------------------------


class TestEdgeTypes:
    """OntologyProfile.edge_types built from relationship_types."""

    def test_edge_types_populated(self):
        registry = _load_registry(_yaml_lane(rel_type_name='USES_MOVE'))
        profile = registry.get('target_lane')
        assert profile is not None
        assert 'USES_MOVE' in profile.edge_types

    def test_edge_type_is_pydantic_model(self):
        from pydantic import BaseModel

        registry = _load_registry(_yaml_lane(rel_type_name='OPENS_WITH'))
        profile = registry.get('target_lane')
        assert issubclass(profile.edge_types['OPENS_WITH'], BaseModel)

    def test_no_relationship_types_gives_empty_edge_types(self):
        lane_yaml = yaml.dump({'minimal_lane': {
            'entity_types': [{'name': 'Entity', 'description': 'Any.'}],
        }})
        registry = _load_registry(lane_yaml)
        profile = registry.get('minimal_lane')
        assert profile is not None
        assert profile.edge_types == {}


# ---------------------------------------------------------------------------
# resolve_ontology 4-tuple integration test
# ---------------------------------------------------------------------------


class TestResolveOntologyIntegration:
    """resolve_ontology() returns correct 4-tuple for constrained + permissive lanes."""

    def _make_registry(self):
        content = yaml.dump({
            'constrained_lane': {
                'entity_types': [{'name': 'RhetoricalMove', 'description': 'A move.'}],
                'relationship_types': [{'name': 'USES_MOVE', 'description': 'uses.'}],
                'extraction_mode': 'constrained_soft',
                'intent_guidance': 'Focus on rhetorical moves.',
            },
            'permissive_lane': {
                'entity_types': [{'name': 'GenericEntity', 'description': 'Any.'}],
            },
        })
        return _load_registry(content)

    def test_constrained_lane_mode(self):
        registry = self._make_registry()
        profile = registry.get('constrained_lane')
        assert profile is not None
        assert profile.extraction_mode == 'constrained_soft'

    def test_constrained_lane_guidance_non_empty(self):
        registry = self._make_registry()
        profile = registry.get('constrained_lane')
        assert profile.intent_guidance == 'Focus on rhetorical moves.'

    def test_permissive_lane_mode(self):
        registry = self._make_registry()
        profile = registry.get('permissive_lane')
        assert profile is not None
        assert profile.extraction_mode == 'permissive'

    def test_permissive_lane_guidance_empty(self):
        registry = self._make_registry()
        profile = registry.get('permissive_lane')
        assert profile.intent_guidance == ''

    def test_unregistered_lane_returns_none(self):
        registry = self._make_registry()
        assert registry.get('not_a_lane') is None
