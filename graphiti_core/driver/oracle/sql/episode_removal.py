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

GET_REMOVAL_NODES = """
SELECT * FROM graphiti_nodes
WHERE group_id = :group_id
ORDER BY uuid
""".strip()

GET_REMOVAL_EDGES = """
SELECT * FROM graphiti_edges
WHERE group_id = :group_id
ORDER BY uuid
""".strip()

LOCK_REMOVAL_NODES = """
SELECT uuid FROM graphiti_nodes
WHERE group_id = :group_id
ORDER BY uuid
FOR UPDATE
""".strip()

LOCK_REMOVAL_EDGES = """
SELECT uuid FROM graphiti_edges
WHERE group_id = :group_id
ORDER BY uuid
FOR UPDATE
""".strip()
