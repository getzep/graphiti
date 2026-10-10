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

import json
from datetime import date, datetime, timezone

from graphiti_core.prompts.prompt_helpers import to_prompt_json


def test_to_prompt_json_serializes_datetime():
    """datetime values serialize as ISO-8601 strings instead of raising.

    Regression test: entity/edge attribute dicts routinely carry datetime
    values (e.g. an entity attribute extracted as a date). Serializing them
    for a summary/extraction prompt previously raised
    `TypeError: Object of type datetime is not JSON serializable`.
    """
    dt = datetime(2024, 3, 15, 9, 30, tzinfo=timezone.utc)
    result = to_prompt_json({'signed_at': dt, 'name': 'Acme contract'})

    parsed = json.loads(result)
    assert parsed['name'] == 'Acme contract'
    assert parsed['signed_at'] == dt.isoformat()


def test_to_prompt_json_serializes_date():
    """`date` (not just `datetime`) also serializes as ISO-8601."""
    result = to_prompt_json({'effective': date(2024, 1, 1)})
    assert json.loads(result)['effective'] == '2024-01-01'


def test_to_prompt_json_serializes_nested_datetime():
    """datetimes nested inside lists/dicts are handled too."""
    dt = datetime(2024, 3, 15, 9, 30, tzinfo=timezone.utc)
    result = to_prompt_json({'events': [{'at': dt}]})
    assert json.loads(result)['events'][0]['at'] == dt.isoformat()


def test_to_prompt_json_preserves_plain_data():
    """Already-serializable data is unaffected by the datetime fallback."""
    data = {'facts': ['a', 'b'], 'count': 3, 'nested': {'k': None}}
    assert json.loads(to_prompt_json(data)) == data


def test_to_prompt_json_preserves_unicode_and_indent():
    """`ensure_ascii=False` keeps non-ASCII readable; `indent` still applies."""
    result = to_prompt_json({'city': 'Brüssel'}, indent=2)
    assert 'Brüssel' in result  # not \u-escaped
    assert '\n' in result  # indent applied
