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

import os

import pytest
from dotenv import load_dotenv

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.helpers import lucene_sanitize

load_dotenv()

HAS_NEO4J = False
HAS_FALKORDB = False
if os.getenv('DISABLE_NEO4J') is None:
    try:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        HAS_NEO4J = True
    except ImportError:
        pass

if os.getenv('DISABLE_FALKORDB') is None:
    try:
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        HAS_FALKORDB = True
    except ImportError:
        pass

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'test')

FALKORDB_HOST = os.getenv('FALKORDB_HOST', 'localhost')
FALKORDB_PORT = os.getenv('FALKORDB_PORT', '6379')
FALKORDB_USER = os.getenv('FALKORDB_USER', None)
FALKORDB_PASSWORD = os.getenv('FALKORDB_PASSWORD', None)


def get_driver(driver_name: str) -> GraphDriver:
    if driver_name == 'neo4j':
        return Neo4jDriver(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
        )
    elif driver_name == 'falkordb':
        return FalkorDriver(
            host=FALKORDB_HOST,
            port=int(FALKORDB_PORT),
            username=FALKORDB_USER,
            password=FALKORDB_PASSWORD,
        )
    else:
        raise ValueError(f'Driver {driver_name} not available')


drivers: list[str] = []
if HAS_NEO4J:
    drivers.append('neo4j')
if HAS_FALKORDB:
    drivers.append('falkordb')


def test_lucene_sanitize():
    # Call the function with test data
    queries = [
        (
            'This has every escape character + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /',
            '\\This has every escape character \\+ \\- \\&\\& \\|\\| \\! \\( \\) \\{ \\} \\[ \\] \\^ \\" \\~ \\* \\? \\: \\\\ \\/',
        ),
        ('this has no escape characters', 'this has no escape characters'),
    ]

    for query, assert_result in queries:
        result = lucene_sanitize(query)
        assert assert_result == result


if __name__ == '__main__':
    pytest.main([__file__])
