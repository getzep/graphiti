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

import re

import pytest

from graphiti_core.driver.oracle.errors import InvalidTraversalDepth
from graphiti_core.driver.oracle.search.traversal import build_bfs_sql
from graphiti_core.driver.oracle.session import validate_oracle_statement


@pytest.mark.parametrize('depth', [-1, 0, 9, 100, True])
def test_depth_outside_one_to_eight_is_rejected(depth: int) -> None:
    with pytest.raises(InvalidTraversalDepth):
        build_bfs_sql(max_depth=depth)


def test_depth_is_rendered_only_after_integer_validation() -> None:
    sql, binds = build_bfs_sql(max_depth=3)

    assert '{1,3}' in sql
    assert re.search(r'\$[A-Za-z_]', sql) is None
    assert 'max_depth' not in binds
    assert 'GRAPH_TABLE' in sql
    assert 'path_edge.group_id = origin.group_id' in sql
    assert 'origin.group_id = reached.group_id' in sql
    validate_oracle_statement(sql)


def test_graph_name_is_an_allowlisted_identifier() -> None:
    with pytest.raises(ValueError, match='graph_name'):
        build_bfs_sql(max_depth=2, graph_name='GRAPHITI_GRAPH) WHERE 1=1 --')
