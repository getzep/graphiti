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

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig


class MockLLMClient(LLMClient):
    """Concrete implementation of LLMClient for testing"""

    async def _generate_response(self, messages, response_model=None):
        return {'content': 'test'}


def test_clean_input():
    client = MockLLMClient(LLMConfig())

    test_cases = [
        # Basic text should remain unchanged
        ('Hello World', 'Hello World'),
        # Control characters should be removed
        ('Hello\x00World', 'HelloWorld'),
        # Newlines, tabs, returns should be preserved
        ('Hello\nWorld\tTest\r', 'Hello\nWorld\tTest\r'),
        # Invalid Unicode should be removed
        ('Hello\udcdeWorld', 'HelloWorld'),
        # Zero-width characters should be removed
        ('Hello\u200bWorld', 'HelloWorld'),
        ('Test\ufeffWord', 'TestWord'),
        # Multiple issues combined
        ('Hello\x00\u200b\nWorld\udcde', 'Hello\nWorld'),
        # Empty string should remain empty
        ('', ''),
        # Form feed and other control characters from the error case
        ('{"edges":[{"relation_typ...\f\x04Hn\\?"}]}', '{"edges":[{"relation_typ...Hn\\?"}]}'),
        # More specific control character tests
        ('Hello\x0cWorld', 'HelloWorld'),  # form feed \f
        ('Hello\x04World', 'HelloWorld'),  # end of transmission
        # Combined JSON-like string with control characters
        ('{"test": "value\f\x00\x04"}', '{"test": "value"}'),
    ]

    for input_str, expected in test_cases:
        assert client._clean_input(input_str) == expected, f'Failed for input: {repr(input_str)}'
