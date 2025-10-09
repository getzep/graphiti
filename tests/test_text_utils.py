"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence


def test_truncate_at_sentence_short_text():
    """Test that short text is returned unchanged."""
    text = 'This is a short sentence.'
    result = truncate_at_sentence(text, 100)
    assert result == text


def test_truncate_at_sentence_empty():
    """Test that empty text is handled correctly."""
    assert truncate_at_sentence('', 100) == ''
    assert truncate_at_sentence(None, 100) is None


def test_truncate_at_sentence_exact_length():
    """Test text at exactly max_chars."""
    text = 'A' * 100
    result = truncate_at_sentence(text, 100)
    assert result == text


def test_truncate_at_sentence_with_period():
    """Test truncation at sentence boundary with period."""
    text = 'First sentence. Second sentence. Third sentence. Fourth sentence.'
    result = truncate_at_sentence(text, 40)
    assert result == 'First sentence. Second sentence.'
    assert len(result) <= 40


def test_truncate_at_sentence_with_question():
    """Test truncation at sentence boundary with question mark."""
    text = 'What is this? This is a test. More text here.'
    result = truncate_at_sentence(text, 30)
    assert result == 'What is this? This is a test.'
    assert len(result) <= 32


def test_truncate_at_sentence_with_exclamation():
    """Test truncation at sentence boundary with exclamation mark."""
    text = 'Hello world! This is exciting. And more text.'
    result = truncate_at_sentence(text, 30)
    assert result == 'Hello world! This is exciting.'
    assert len(result) <= 32


def test_truncate_at_sentence_no_boundary():
    """Test truncation when no sentence boundary exists before max_chars."""
    text = 'This is a very long sentence without any punctuation marks near the beginning'
    result = truncate_at_sentence(text, 30)
    assert len(result) <= 30
    assert result.startswith('This is a very long sentence')


def test_truncate_at_sentence_multiple_periods():
    """Test with multiple sentence endings."""
    text = 'A. B. C. D. E. F. G. H.'
    result = truncate_at_sentence(text, 10)
    assert result == 'A. B. C.'
    assert len(result) <= 10


def test_truncate_at_sentence_strips_trailing_whitespace():
    """Test that trailing whitespace is stripped."""
    text = 'First sentence.   Second sentence.'
    result = truncate_at_sentence(text, 20)
    assert result == 'First sentence.'
    assert not result.endswith(' ')


def test_max_summary_chars_constant():
    """Test that MAX_SUMMARY_CHARS is set to expected value."""
    assert MAX_SUMMARY_CHARS == 250


def test_truncate_at_sentence_realistic_summary():
    """Test with a realistic entity summary."""
    text = (
        'John is a software engineer who works at a tech company in San Francisco. '
        'He has been programming for over 10 years and specializes in Python and distributed systems. '
        'John enjoys hiking on weekends and is learning to play guitar. '
        'He graduated from MIT with a degree in computer science.'
    )
    result = truncate_at_sentence(text, MAX_SUMMARY_CHARS)
    assert len(result) <= MAX_SUMMARY_CHARS
    # Should keep complete sentences
    assert result.endswith('.')
    # Should include at least the first sentence
    assert 'John is a software engineer' in result
