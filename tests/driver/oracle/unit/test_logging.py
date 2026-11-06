# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from graphiti_core.driver.oracle.logging import operation_fields


def test_operation_fields_expose_metrics_without_secrets_or_payloads() -> None:
    fields = operation_fields(
        operation='entity_node.save',
        duration_ms=12.5,
        result_count=1,
        retry_count=0,
        oracle_code=12514,
    )

    assert fields == {
        'operation': 'entity_node.save',
        'duration_ms': 12.5,
        'result_count': 1,
        'retry_count': 0,
        'oracle_code': 12514,
    }
    serialized = repr(fields)
    assert 'password' not in serialized
    assert 'embedding' not in serialized
    assert 'content' not in serialized
