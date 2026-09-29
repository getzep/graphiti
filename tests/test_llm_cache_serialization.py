"""LLMCache must skip any value it cannot serialize, not only some of them."""

import json

import pytest

from graphiti_core.llm_client.cache import LLMCache


@pytest.fixture()
def cache(tmp_path):
    cache = LLMCache(str(tmp_path / "cache"))
    yield cache
    cache.close()


def test_round_trip(cache):
    cache.set("key", {"value": 42})

    assert cache.get("key") == {"value": 42}


def test_unserializable_object_is_skipped(cache):
    cache.set("key", {"value": object()})

    assert cache.get("key") is None


def test_circular_reference_is_skipped(cache):
    """json.dumps raises ValueError (not TypeError) for a circular value.

    The cache documents that only JSON-serializable data is stored, so a value
    that cannot be serialized has to be skipped here instead of escaping to the
    caller as an unhandled ValueError.
    """
    value: dict = {}
    value["self"] = value

    cache.set("key", {"value": value})

    assert cache.get("key") is None


def test_earlier_entries_are_untouched_by_a_skipped_value(cache):
    cache.set("first", {"value": 1})
    circular: dict = {}
    circular["self"] = circular

    cache.set("second", {"value": circular})

    assert cache.get("first") == {"value": 1}
    assert cache.get("second") is None
