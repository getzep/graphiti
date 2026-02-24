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

import os

import pytest

from graphiti_core.llm_client.cache import LLMCache


@pytest.fixture
def cache(tmp_path):
    """Create an LLMCache using a temporary directory."""
    c = LLMCache(str(tmp_path / 'test_cache'))
    yield c
    c.close()


class TestLLMCache:
    def test_get_missing_key_returns_none(self, cache):
        """Test that getting a nonexistent key returns None."""
        assert cache.get('nonexistent') is None

    def test_set_and_get(self, cache):
        """Test basic set and get round-trip."""
        value = {'content': 'hello', 'tokens': 42}
        cache.set('key1', value)
        assert cache.get('key1') == value

    def test_set_overwrites_existing(self, cache):
        """Test that setting the same key overwrites the previous value."""
        cache.set('key1', {'version': 1})
        cache.set('key1', {'version': 2})
        assert cache.get('key1') == {'version': 2}

    def test_multiple_keys(self, cache):
        """Test storing and retrieving multiple distinct keys."""
        cache.set('a', {'val': 1})
        cache.set('b', {'val': 2})
        cache.set('c', {'val': 3})

        assert cache.get('a') == {'val': 1}
        assert cache.get('b') == {'val': 2}
        assert cache.get('c') == {'val': 3}

    def test_complex_nested_value(self, cache):
        """Test that complex nested JSON structures survive round-trip."""
        value = {
            'choices': [{'message': {'role': 'assistant', 'content': 'test'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
            'nested': {'a': [1, 2, 3], 'b': None, 'c': True},
        }
        cache.set('complex', value)
        assert cache.get('complex') == value

    def test_non_serializable_value_is_skipped(self, cache):
        """Test that non-JSON-serializable values are silently skipped."""
        cache.set('bad', {'func': lambda x: x})  # type: ignore
        assert cache.get('bad') is None

    def test_corrupted_entry_returns_none(self, cache):
        """Test that a corrupted (non-JSON) cache entry returns None."""
        # Directly insert invalid JSON into the database
        cache._conn.execute(
            'INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)',
            ('corrupt', 'not valid json{{{'),
        )
        cache._conn.commit()
        assert cache.get('corrupt') is None

    def test_creates_directory(self, tmp_path):
        """Test that LLMCache creates the directory if it doesn't exist."""
        cache_dir = str(tmp_path / 'nested' / 'dir' / 'cache')
        c = LLMCache(cache_dir)
        try:
            assert os.path.isdir(cache_dir)
            assert os.path.isfile(os.path.join(cache_dir, 'cache.db'))
        finally:
            c.close()

    def test_persistence_across_instances(self, tmp_path):
        """Test that data persists when opening a new LLMCache on the same directory."""
        cache_dir = str(tmp_path / 'persist_cache')
        c1 = LLMCache(cache_dir)
        c1.set('persist_key', {'data': 'survives'})
        c1.close()

        c2 = LLMCache(cache_dir)
        try:
            assert c2.get('persist_key') == {'data': 'survives'}
        finally:
            c2.close()

    def test_close_and_del(self, tmp_path):
        """Test that close() and __del__ don't raise exceptions."""
        c = LLMCache(str(tmp_path / 'close_test'))
        c.close()
        # Calling close again via __del__ should not raise
        c.__del__()
