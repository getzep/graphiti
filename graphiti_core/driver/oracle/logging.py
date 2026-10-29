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

from __future__ import annotations


def operation_fields(
    *,
    operation: str,
    duration_ms: float,
    result_count: int,
    retry_count: int,
    oracle_code: int | None,
) -> dict[str, str | float | int | None]:
    """Build the allowlisted fields emitted for one driver operation."""

    return {
        'operation': operation,
        'duration_ms': duration_ms,
        'result_count': result_count,
        'retry_count': retry_count,
        'oracle_code': oracle_code,
    }
