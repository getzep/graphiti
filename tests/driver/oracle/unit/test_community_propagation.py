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

import subprocess
import sys


def test_two_node_cycle_finishes_as_one_community() -> None:
    code = (
        'from graphiti_core.driver.oracle.community import Neighbor, label_propagation; '
        "graph={'a':[Neighbor(node_uuid='b', edge_count=2)], "
        "'b':[Neighbor(node_uuid='a', edge_count=2)]}; "
        'print(label_propagation(graph))'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert result.stdout.strip() == "[['a', 'b']]"
