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

from types import SimpleNamespace

from graphiti_core.graphiti import _in_requested_uuid_order


def test_in_requested_uuid_order_follows_caller_sequence():
    items = [
        SimpleNamespace(uuid='c', name='third'),
        SimpleNamespace(uuid='a', name='first'),
        SimpleNamespace(uuid='b', name='second'),
    ]

    ordered = _in_requested_uuid_order(items, ['a', 'missing', 'b', 'c'])

    assert [item.uuid for item in ordered] == ['a', 'b', 'c']
