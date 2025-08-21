"""
Tests for LLM client utility functions.
"""

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS
from graphiti_core.llm_client.utils import resolve_max_tokens


class TestResolveMaxTokens:
    """Comprehensive tests for the resolve_max_tokens utility function."""

    def test_precedence_order_all_parameters_provided(self):
        """Test that requested_max_tokens has highest precedence when all parameters are provided."""
        result = resolve_max_tokens(
            requested_max_tokens=1000,
            config_max_tokens=2000,
            instance_max_tokens=3000,
            default_max_tokens=4000,
        )
        assert result == 1000

    def test_config_takes_precedence_over_instance_and_default(self):
        """Test that config_max_tokens takes precedence over instance and default."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=2000,
            instance_max_tokens=3000,
            default_max_tokens=4000,
        )
        assert result == 2000

    def test_instance_takes_precedence_over_default(self):
        """Test that instance_max_tokens takes precedence over default."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=None,
            instance_max_tokens=3000,
            default_max_tokens=4000,
        )
        assert result == 3000

    def test_default_fallback_when_all_none(self):
        """Test that default_max_tokens is used when all other values are None."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=None,
            instance_max_tokens=None,
            default_max_tokens=4000,
        )
        assert result == 4000

    def test_config_ignored_when_equals_default(self):
        """Test that config_max_tokens is ignored when it equals the default."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=8192,  # Same as default
            instance_max_tokens=3000,
            default_max_tokens=8192,
        )
        assert result == 3000  # Should use instance, not config

    def test_config_ignored_when_equals_default_falls_back_to_default(self):
        """Test fallback to default when config equals default and instance is None."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=8192,  # Same as default
            instance_max_tokens=None,
            default_max_tokens=8192,
        )
        assert result == 8192

    def test_zero_values_are_treated_as_valid(self):
        """Test that zero values are treated as valid, not None."""
        result = resolve_max_tokens(
            requested_max_tokens=0,
            config_max_tokens=2000,
            instance_max_tokens=3000,
            default_max_tokens=4000,
        )
        assert result == 0  # Zero should be respected as highest precedence

    def test_config_zero_different_from_default(self):
        """Test that config=0 is treated as different from non-zero default."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=0,  # Different from default
            instance_max_tokens=3000,
            default_max_tokens=8192,
        )
        assert result == 0  # Config should win even if it's zero

    def test_realistic_scenario_user_sets_high_limit(self):
        """Test realistic scenario where user sets a high token limit in config."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=16000,  # User wants more tokens
            instance_max_tokens=None,
            default_max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert result == 16000

    def test_realistic_scenario_runtime_override(self):
        """Test realistic scenario where runtime parameter overrides everything."""
        result = resolve_max_tokens(
            requested_max_tokens=32000,  # Runtime override
            config_max_tokens=16000,
            instance_max_tokens=4000,
            default_max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert result == 32000

    def test_realistic_scenario_constructor_fallback(self):
        """Test realistic scenario where constructor parameter is used as fallback."""
        result = resolve_max_tokens(
            requested_max_tokens=None,
            config_max_tokens=DEFAULT_MAX_TOKENS,  # Same as default, ignored
            instance_max_tokens=4000,  # Constructor parameter
            default_max_tokens=DEFAULT_MAX_TOKENS,
        )
        assert result == 4000

    def test_edge_case_negative_values(self):
        """Test that negative values are handled (though they shouldn't occur in practice)."""
        result = resolve_max_tokens(
            requested_max_tokens=-1,
            config_max_tokens=2000,
            instance_max_tokens=3000,
            default_max_tokens=4000,
        )
        assert result == -1  # Negative should still have highest precedence

    def test_default_parameters(self):
        """Test function with default parameters."""
        result = resolve_max_tokens()
        assert result == DEFAULT_MAX_TOKENS

    def test_only_requested_provided(self):
        """Test when only requested_max_tokens is provided."""
        result = resolve_max_tokens(requested_max_tokens=5000)
        assert result == 5000

    def test_only_config_provided_different_from_default(self):
        """Test when only config_max_tokens is provided and different from default."""
        result = resolve_max_tokens(config_max_tokens=16000)
        assert result == 16000

    def test_only_config_provided_same_as_default(self):
        """Test when only config_max_tokens is provided but equals default."""
        result = resolve_max_tokens(config_max_tokens=DEFAULT_MAX_TOKENS)
        assert result == DEFAULT_MAX_TOKENS

    def test_only_instance_provided(self):
        """Test when only instance_max_tokens is provided."""
        result = resolve_max_tokens(instance_max_tokens=4000)
        assert result == 4000 