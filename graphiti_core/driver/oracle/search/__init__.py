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

"""Safe Oracle Text, VECTOR, and SQL/PGQ search building blocks."""

from .filters import OracleWhere, build_edge_where, build_node_where
from .text import build_oracle_text_query
from .traversal import build_bfs_sql, validate_traversal_depth

__all__ = [
    'OracleWhere',
    'build_edge_where',
    'build_node_where',
    'build_oracle_text_query',
    'build_bfs_sql',
    'validate_traversal_depth',
]
