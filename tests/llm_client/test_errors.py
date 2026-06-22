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

# Running tests: pytest -xvs tests/llm_client/test_errors.py

import pytest

from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError, RefusalError


class TestRateLimitError:
    """Tests for the RateLimitError class."""

    def test_default_message(self):
        """Test that the default message is set correctly."""
        error = RateLimitError()
        assert error.message == 'Rate limit exceeded. Please try again later.'
        assert str(error) == 'Rate limit exceeded. Please try again later.'

    def test_custom_message(self):
        """Test that a custom message can be set."""
        custom_message = 'Custom rate limit message'
        error = RateLimitError(custom_message)
        assert error.message == custom_message
        assert str(error) == custom_message


class TestRefusalError:
    """Tests for the RefusalError class."""

    def test_message_required(self):
        """Test that a message is required for RefusalError."""
        with pytest.raises(TypeError):
            # Intentionally not providing the required message parameter
            RefusalError()  # type: ignore

    def test_message_assignment(self):
        """Test that the message is assigned correctly."""
        message = 'The LLM refused to respond to this prompt.'
        error = RefusalError(message=message)  # Add explicit keyword argument
        assert error.message == message
        assert str(error) == message


class TestEmptyResponseError:
    """Tests for the EmptyResponseError class."""

    def test_message_required(self):
        """Test that a message is required for EmptyResponseError."""
        with pytest.raises(TypeError):
            # Intentionally not providing the required message parameter
            EmptyResponseError()  # type: ignore

    def test_message_assignment(self):
        """Test that the message is assigned correctly."""
        message = 'The LLM returned an empty response.'
        error = EmptyResponseError(message=message)  # Add explicit keyword argument
        assert error.message == message
        assert str(error) == message


if __name__ == '__main__':
    pytest.main(['-v', 'test_errors.py'])
