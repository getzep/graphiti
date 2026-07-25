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

from typing import TypeGuard

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.query_executor import QueryExecutor


def should_route_group_ids_to_databases(
    executor: QueryExecutor,
    group_ids: list[str] | None,
) -> TypeGuard[GraphDriver]:
    if not isinstance(executor, GraphDriver):
        return False
    if executor.provider != GraphProvider.FALKORDB or not group_ids:
        return False
    return any(executor.should_route_group_id_to_database(group_id) for group_id in group_ids)
