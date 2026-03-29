#!/usr/bin/env python3
"""Tests for memory classifier base classes and interfaces."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from classifiers.base import ClassificationResult, MemoryCategory, MemoryClassifier
from utils.project_config import ProjectConfig


# Mock implementation for testing
class MockClassifier(MemoryClassifier):
    """Mock classifier for testing interface"""

    async def classify(
        self, episode_body: str, project_config: ProjectConfig
    ) -> ClassificationResult:
        return ClassificationResult(
            category=MemoryCategory.PROJECT_SPECIFIC,
            confidence=0.8,
            reasoning='Mock classification',
        )

    def supports(self, strategy: str) -> bool:
        return strategy == 'mock'


class HighThresholdClassifier(MockClassifier):
    """Mock classifier with high confidence threshold"""

    def get_confidence_threshold(self) -> float:
        return 0.9


def test_memory_category_enum():
    """Test MemoryCategory enum values."""
    print('Test: MemoryCategory enum')
    assert MemoryCategory.PROJECT_SPECIFIC.value == 'project_specific'
    assert MemoryCategory.SHARED.value == 'shared'
    assert MemoryCategory.MIXED.value == 'mixed'
    print('  ✓ MemoryCategory enum has correct values')


def test_classification_result_creation():
    """Test creating ClassificationResult."""
    print('\nTest: ClassificationResult creation')
    result = ClassificationResult(
        category=MemoryCategory.SHARED, confidence=0.8, reasoning='Test reasoning'
    )
    assert result.category == MemoryCategory.SHARED
    assert result.confidence == 0.8
    assert result.reasoning == 'Test reasoning'
    assert result.is_shared is True
    assert result.is_project_specific is False
    print('  ✓ ClassificationResult created successfully')


def test_classification_result_invalid_confidence():
    """Test ClassificationResult with invalid confidence."""
    print('\nTest: ClassificationResult with invalid confidence')
    try:
        ClassificationResult(
            category=MemoryCategory.SHARED,
            confidence=1.5,  # Invalid
            reasoning='Test',
        )
        raise AssertionError('Should have raised ValueError')
    except ValueError as e:
        assert 'Confidence must be between 0.0 and 1.0' in str(e)
        print('  ✓ Invalid confidence raises ValueError')


def test_classification_result_properties():
    """Test ClassificationResult category properties."""
    print('\nTest: ClassificationResult category properties')

    # Project specific
    result_ps = ClassificationResult(category=MemoryCategory.PROJECT_SPECIFIC, confidence=0.7)
    assert result_ps.is_project_specific is True
    assert result_ps.is_shared is False

    # Shared
    result_s = ClassificationResult(category=MemoryCategory.SHARED, confidence=0.7)
    assert result_s.is_shared is True
    assert result_s.is_project_specific is False

    # Mixed
    result_m = ClassificationResult(category=MemoryCategory.MIXED, confidence=0.7)
    assert result_m.is_shared is True
    assert result_m.is_project_specific is True

    print('  ✓ Category properties work correctly')


def test_classification_result_mixed_content():
    """Test ClassificationResult with mixed content parts."""
    print('\nTest: ClassificationResult with mixed content')
    result = ClassificationResult(
        category=MemoryCategory.MIXED,
        confidence=0.6,
        shared_part='User prefers 4 spaces',
        project_part='Project configured with ESLint',
    )
    assert result.shared_part == 'User prefers 4 spaces'
    assert result.project_part == 'Project configured with ESLint'
    print('  ✓ Mixed content parts stored correctly')


def test_classifier_interface():
    """Test that classifier interface is properly defined."""
    print('\nTest: MemoryClassifier interface')

    classifier = MockClassifier()

    # Check it's an instance of ABC
    assert isinstance(classifier, MemoryClassifier)

    # Check supports method
    assert classifier.supports('mock') is True
    assert classifier.supports('other') is False

    # Check classify method exists and is awaitable
    import inspect

    assert inspect.iscoroutinefunction(classifier.classify)

    print('  ✓ MemoryClassifier interface properly defined')


def test_classifier_confidence_threshold():
    """Test confidence threshold methods."""
    print('\nTest: Classifier confidence threshold')

    # Default threshold
    default_classifier = MockClassifier()
    assert default_classifier.get_confidence_threshold() == 0.5

    result_low = ClassificationResult(category=MemoryCategory.SHARED, confidence=0.4)
    result_high = ClassificationResult(category=MemoryCategory.SHARED, confidence=0.8)

    assert default_classifier.is_confident(result_low) is False
    assert default_classifier.is_confident(result_high) is True

    # High threshold classifier
    high_classifier = HighThresholdClassifier()
    assert high_classifier.get_confidence_threshold() == 0.9

    assert high_classifier.is_confident(result_low) is False
    assert high_classifier.is_confident(result_high) is False

    result_very_high = ClassificationResult(category=MemoryCategory.SHARED, confidence=0.95)
    assert high_classifier.is_confident(result_very_high) is True

    print('  ✓ Confidence threshold methods work correctly')


def run_all_tests():
    """Run all tests."""
    print('=' * 60)
    print('Running Memory Classifier Base Tests')
    print('=' * 60)

    tests = [
        test_memory_category_enum,
        test_classification_result_creation,
        test_classification_result_invalid_confidence,
        test_classification_result_properties,
        test_classification_result_mixed_content,
        test_classifier_interface,
        test_classifier_confidence_threshold,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f'  ✗ FAILED: {e}')
            failed += 1
        except Exception as e:
            print(f'  ✗ ERROR: {e}')
            import traceback

            traceback.print_exc()
            failed += 1

    print('\n' + '=' * 60)
    print(f'Results: {passed} passed, {failed} failed')
    print('=' * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
