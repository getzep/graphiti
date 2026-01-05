#!/usr/bin/env python3
"""Tests for LLM-based memory classifier."""

import sys
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from classifiers.llm_based import LLMClassifier
from classifiers.base import MemoryCategory, ClassificationResult
from utils.project_config import ProjectConfig


def create_test_config(shared_entity_types=None, shared_patterns=None):
    """Helper to create test ProjectConfig."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {"group_id": "test-project"}
        if shared_entity_types:
            config_data["shared_entity_types"] = shared_entity_types
        if shared_patterns:
            config_data["shared_patterns"] = shared_patterns
        json.dump(config_data, f)
        config_path = Path(f.name)

    return ProjectConfig(
        group_id="test-project",
        config_path=config_path,
        shared_entity_types=shared_entity_types or [],
        shared_patterns=shared_patterns or []
    )


async def test_llm_classifier_shared():
    """Test LLM classifier classifying as shared."""
    print("\nTest: LLM classifier classifies as SHARED")
    
    mock_llm = AsyncMock()
    # Mock structured output response
    mock_llm.generate_structured_output = AsyncMock(
        return_value='{"category": "SHARED", "confidence": 0.9, "reasoning": "User preference about coding style"}'
    )
    
    classifier = LLMClassifier(llm_client=mock_llm)
    config = create_test_config(shared_entity_types=["Preference"])
    
    result = await classifier.classify("User preference: I prefer 4-space indentation", config)
    
    assert result.category == MemoryCategory.SHARED
    assert result.confidence == 0.9
    assert "preference" in result.reasoning.lower()
    print("  ✓ Correctly classified as SHARED")


async def test_llm_classifier_project_specific():
    """Test LLM classifier classifying as project-specific."""
    print("\nTest: LLM classifier classifies as PROJECT_SPECIFIC")
    
    mock_llm = AsyncMock()
    mock_llm.generate_structured_output = AsyncMock(
        return_value='{"category": "PROJECT_SPECIFIC", "confidence": 0.85, "reasoning": "Project-specific API endpoint"}'
    )
    
    classifier = LLMClassifier(llm_client=mock_llm)
    config = create_test_config()
    
    result = await classifier.classify("The API endpoint is at /api/v1/users", config)
    
    assert result.category == MemoryCategory.PROJECT_SPECIFIC
    assert result.confidence == 0.85
    print("  ✓ Correctly classified as PROJECT_SPECIFIC")


async def test_llm_classifier_mixed():
    """Test LLM classifier classifying as mixed."""
    print("\nTest: LLM classifier classifies as MIXED")
    
    mock_llm = AsyncMock()
    # First call for classification, second for splitting
    mock_llm.generate_structured_output = AsyncMock(
        side_effect=[
            '{"category": "MIXED", "confidence": 0.75, "reasoning": "Contains both preference and project detail"}',
            '{"shared_part": "User prefers dark mode", "project_part": "Project uses React"}'
        ]
    )
    
    classifier = LLMClassifier(llm_client=mock_llm, confidence_threshold=0.6)
    config = create_test_config()
    
    result = await classifier.classify("User prefers dark mode. Project uses React with TypeScript.", config)
    
    assert result.category == MemoryCategory.MIXED
    assert result.shared_part == "User prefers dark mode"
    assert result.project_part == "Project uses React"
    print("  ✓ Correctly classified as MIXED with content split")


async def test_llm_classifier_chat_interface():
    """Test LLM classifier using chat interface."""
    print("\nTest: LLM classifier using chat interface")
    
    mock_llm = MagicMock()
    # Don't have generate_structured_output, but have chat
    delattr(mock_llm, 'generate_structured_output')
    mock_llm.chat = AsyncMock(
        return_value='{"category": "SHARED", "confidence": 0.95, "reasoning": "Coding convention"}'
    )
    
    classifier = LLMClassifier(llm_client=mock_llm)
    config = create_test_config()
    
    result = await classifier.classify("Convention: use 4 spaces for indentation", config)
    
    assert result.category == MemoryCategory.SHARED
    assert mock_llm.chat.called
    print("  ✓ Chat interface works correctly")


async def test_llm_classifier_fallback_on_error():
    """Test LLM classifier fallback on error."""
    print("\nTest: LLM classifier fallback on error")
    
    mock_llm = AsyncMock()
    mock_llm.generate_structured_output = AsyncMock(side_effect=Exception("LLM error"))
    
    classifier = LLMClassifier(llm_client=mock_llm)
    config = create_test_config()
    
    result = await classifier.classify("Some content", config)
    
    # Should fallback to PROJECT_SPECIFIC on error
    assert result.category == MemoryCategory.PROJECT_SPECIFIC
    assert "LLM classification failed" in result.reasoning
    print("  ✓ Fallback to PROJECT_SPECIFIC on error")


async def test_llm_classifier_split_content_only_when_mixed():
    """Test that content splitting only happens for MIXED category."""
    print("\nTest: Content splitting only for MIXED category")
    
    mock_llm = AsyncMock()
    mock_llm.generate_structured_output = AsyncMock(
        return_value='{"category": "SHARED", "confidence": 0.9, "reasoning": "User preference"}'
    )
    
    classifier = LLMClassifier(llm_client=mock_llm, confidence_threshold=0.6)
    config = create_test_config()
    
    result = await classifier.classify("User preference: dark mode", config)
    
    # Should not call LLM again for splitting since it's SHARED not MIXED
    assert mock_llm.generate_structured_output.call_count == 1
    assert result.shared_part == ""
    assert result.project_part == ""
    print("  ✓ Content splitting only happens for MIXED")


async def test_llm_classifier_supports_strategy():
    """Test supports method."""
    print("\nTest: LLM classifier supports strategy")
    
    mock_llm = AsyncMock()
    classifier = LLMClassifier(llm_client=mock_llm)
    
    assert classifier.supports("llm_based") is True
    assert classifier.supports("llm") is True
    assert classifier.supports("smart") is True
    assert classifier.supports("rule_based") is False
    assert classifier.supports("simple") is False
    
    print("  ✓ Strategy support checking works correctly")


async def test_llm_classifier_confidence_threshold():
    """Test confidence threshold."""
    print("\nTest: LLM classifier confidence threshold")
    
    mock_llm = AsyncMock()
    classifier = LLMClassifier(llm_client=mock_llm, confidence_threshold=0.7)
    
    assert classifier.get_confidence_threshold() == 0.7
    
    # Test is_confident with results
    result_low = ClassificationResult(
        category=MemoryCategory.SHARED,
        confidence=0.6,
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


async def test_llm_classifier_parse_error_handling():
    """Test handling of malformed LLM responses."""
    print("\nTest: LLM classifier handles malformed responses")
    
    mock_llm = AsyncMock()
    mock_llm.generate_structured_output = AsyncMock(
        return_value='invalid json {{{'
    )
    
    classifier = LLMClassifier(llm_client=mock_llm)
    config = create_test_config()
    
    result = await classifier.classify("Some content", config)
    
    # Should fallback to PROJECT_SPECIFIC on parse error
    assert result.category == MemoryCategory.PROJECT_SPECIFIC
    print("  ✓ Malformed response handled with fallback")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running LLM Classifier Tests")
    print("=" * 60)

    async def run_async_tests():
        tests = [
            test_llm_classifier_shared,
            test_llm_classifier_project_specific,
            test_llm_classifier_mixed,
            test_llm_classifier_chat_interface,
            test_llm_classifier_fallback_on_error,
            test_llm_classifier_split_content_only_when_mixed,
            test_llm_classifier_supports_strategy,
            test_llm_classifier_confidence_threshold,
            test_llm_classifier_parse_error_handling,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                await test()
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

    return asyncio.run(run_async_tests())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
