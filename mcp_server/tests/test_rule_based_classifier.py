#!/usr/bin/env python3
"""Tests for rule-based memory classifier."""

import sys
import tempfile
import json
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from classifiers.rule_based import RuleBasedClassifier
from classifiers.base import MemoryCategory, ClassificationResult
from utils.project_config import ProjectConfig


def create_test_config(shared_group_ids=None, shared_entity_types=None, shared_patterns=None):
    """Helper to create test ProjectConfig."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {"group_id": "test-project"}
        if shared_group_ids:
            config_data["shared_group_ids"] = shared_group_ids
        if shared_entity_types:
            config_data["shared_entity_types"] = shared_entity_types
        if shared_patterns:
            config_data["shared_patterns"] = shared_patterns
        json.dump(config_data, f)
        config_path = Path(f.name)

    return ProjectConfig(
        group_id="test-project",
        config_path=config_path,
        shared_group_ids=shared_group_ids or [],
        shared_entity_types=shared_entity_types or [],
        shared_patterns=shared_patterns or []
    )


def test_classify_shared_preference():
    """Test classification when episode contains Preference entity type."""
    print("Test: Classify episode with Preference entity type")
    classifier = RuleBasedClassifier()
    config = create_test_config(shared_entity_types=["Preference"])

    async def run_test():
        result = await classifier.classify("User preference: 4-space indentation", config)
        assert result.category == MemoryCategory.SHARED
        assert result.confidence >= 0.7
        assert "Preference" in result.reasoning
        print(f"  ✓ Classified as SHARED: {result.reasoning}")

    asyncio.run(run_test())


def test_classify_shared_procedure():
    """Test classification when episode contains Procedure entity type."""
    print("\nTest: Classify episode with Procedure entity type")
    classifier = RuleBasedClassifier()
    config = create_test_config(shared_entity_types=["Procedure", "Preference"])

    async def run_test():
        result = await classifier.classify(
            "Standard procedure: run tests before committing",
            config
        )
        assert result.category == MemoryCategory.SHARED
        assert "Procedure" in result.reasoning
        print(f"  ✓ Classified as SHARED: {result.reasoning}")

    asyncio.run(run_test())


def test_classify_project_specific():
    """Test classification when episode has no shared types."""
    print("\nTest: Classify project-specific episode")
    classifier = RuleBasedClassifier()
    config = create_test_config(shared_entity_types=["Preference"])

    async def run_test():
        result = await classifier.classify(
            "The API endpoint is at /api/v1/users",
            config
        )
        assert result.category == MemoryCategory.PROJECT_SPECIFIC
        assert result.confidence == 0.6
        assert "no shared" in result.reasoning.lower()
        print(f"  ✓ Classified as PROJECT_SPECIFIC: {result.reasoning}")

    asyncio.run(run_test())


def test_classify_with_pattern():
    """Test classification using shared patterns."""
    print("\nTest: Classify using shared patterns")
    classifier = RuleBasedClassifier()
    config = create_test_config(
        shared_patterns=["偏好", "习惯"]
    )

    async def run_test():
        result = await classifier.classify(
            "用户的代码风格偏好很重要",
            config
        )
        assert result.category == MemoryCategory.SHARED
        assert "偏好" in result.reasoning
        print(f"  ✓ Classified as SHARED using pattern: {result.reasoning}")

    asyncio.run(run_test())


def test_classify_with_custom_confidence():
    """Test classifier with custom confidence values."""
    print("\nTest: Custom confidence values")
    classifier = RuleBasedClassifier(confidence_shared=0.9, confidence_project=0.8)
    config = create_test_config(shared_entity_types=["Preference"])

    async def run_test():
        result_shared = await classifier.classify("User preference noted", config)
        assert result_shared.confidence == 0.9

        result_project = await classifier.classify("API endpoint configured", config)
        assert result_project.confidence == 0.8

        print("  ✓ Custom confidence values applied correctly")

    asyncio.run(run_test())


def test_supports_strategy():
    """Test strategy support checking."""
    print("\nTest: Strategy support")
    classifier = RuleBasedClassifier()

    assert classifier.supports("simple") is True
    assert classifier.supports("rule_based") is True
    assert classifier.supports("default") is True
    assert classifier.supports("llm_based") is False
    assert classifier.supports("smart_split") is False

    print("  ✓ Strategy support checking works correctly")


def test_confidence_threshold():
    """Test confidence threshold."""
    print("\nTest: Confidence threshold")
    classifier = RuleBasedClassifier()

    assert classifier.get_confidence_threshold() == 0.5

    # Test is_confident with results
    result_low = ClassificationResult(
        category=MemoryCategory.SHARED,
        confidence=0.4,
        reasoning="Test"
    )
    result_high = ClassificationResult(
        category=MemoryCategory.SHARED,
        confidence=0.8,
        reasoning="Test"
    )

    assert classifier.is_confident(result_low) is False
    assert classifier.is_confident(result_high) is True

    print("  ✓ Confidence threshold works correctly")


def test_default_shared_types():
    """Test that default shared types are used when config is empty."""
    print("\nTest: Default shared types")
    classifier = RuleBasedClassifier()
    config = create_test_config()  # No shared_entity_types configured

    async def run_test():
        # Should still detect "Preference" as it's in defaults
        result = await classifier.classify("User preference noted", config)
        assert result.category == MemoryCategory.SHARED
        print("  ✓ Default shared types applied when config is empty")

    asyncio.run(run_test())


def test_multiple_shared_types():
    """Test classification with multiple shared entity types."""
    print("\nTest: Multiple shared entity types")
    classifier = RuleBasedClassifier()
    config = create_test_config(
        shared_entity_types=["Preference", "Procedure", "Requirement", "Topic"]
    )

    async def run_test():
        # Test each type
        result1 = await classifier.classify("User preference: dark mode", config)
        assert result1.category == MemoryCategory.SHARED

        result2 = await classifier.classify("Procedure: run make test", config)
        assert result2.category == MemoryCategory.SHARED

        result3 = await classifier.classify("Requirement: must support Python 3.10", config)
        assert result3.category == MemoryCategory.SHARED

        print("  ✓ Multiple shared types all detected correctly")

    asyncio.run(run_test())


def test_case_insensitive_matching():
    """Test that matching is case-insensitive."""
    print("\nTest: Case-insensitive matching")
    classifier = RuleBasedClassifier()
    config = create_test_config(shared_entity_types=["Preference"])

    async def run_test():
        # Various cases should all match
        variations = [
            "User PREFERENCE noted",
            "user preference set",
            "PREFERENCE: dark mode"
        ]

        for episode in variations:
            result = await classifier.classify(episode, config)
            assert result.category == MemoryCategory.SHARED, f"Failed for: {episode}"

        print("  ✓ Case-insensitive matching works correctly")

    asyncio.run(run_test())


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Rule-Based Classifier Tests")
    print("=" * 60)

    tests = [
        test_classify_shared_preference,
        test_classify_shared_procedure,
        test_classify_project_specific,
        test_classify_with_pattern,
        test_classify_with_custom_confidence,
        test_supports_strategy,
        test_confidence_threshold,
        test_default_shared_types,
        test_multiple_shared_types,
        test_case_insensitive_matching,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
