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

import pytest


def test_neo4j_sanitize():
    """Test Neo4j sanitization by importing the driver and using its sanitize method."""
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    
    # Create a driver instance - if it fails, we'll handle it gracefully
    try:
        driver = Neo4jDriver(uri="bolt://localhost:7687", user="test", password="test")
    except Exception:
        # If we can't create a real driver, skip this test
        pytest.skip("Neo4j driver connection failed - skipping sanitize test")
    
    # Test the sanitize method
    queries = [
        (
            'This has every escape character + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /',
            '\\This has every escape character \\+ \\- \\&\\& \\|\\| \\! \\( \\) \\{ \\} \\[ \\] \\^ \\" \\~ \\* \\? \\: \\\\ \\/',
        ),
        ('this has no escape characters', 'this has no escape characters'),
    ]

    for query, expected_result in queries:
        result = driver.sanitize(query)
        assert expected_result == result


def test_falkordb_sanitize():
    """Test FalkorDB sanitization by importing the driver and using its sanitize method."""
    try:
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        driver = FalkorDriver()
    except ImportError:
        pytest.skip("FalkorDB not installed - skipping sanitize test")
    except Exception:
        # If we can't create a real driver, skip this test
        pytest.skip("FalkorDB driver connection failed - skipping sanitize test")
    
    # Test the sanitize method - FalkorDB replaces special chars with spaces
    queries = [
        (
            'This has special characters: ,.<>{}[]"\':;!@#$%^&*()-+=~',
            'This has special characters',  # All special chars replaced with spaces, then collapsed
        ),
        ('this has no special characters', 'this has no special characters'),
        ('word1,word2.word3', 'word1 word2 word3'),  # Separators become spaces
        ('keep_underscores_intact', 'keep_underscores_intact'),  # Underscores are NOT replaced
    ]

    for query, expected_result in queries:
        result = driver.sanitize(query)
        assert expected_result == result


# Keep the drivers and get_driver functions for other tests that import from this file
drivers = ['neo4j', 'falkordb']

def get_driver(driver_name: str):
    """Helper function to get a driver instance for testing"""
    if driver_name == 'neo4j':
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        return Neo4jDriver(uri="bolt://localhost:7687", user="neo4j", password="test")
    elif driver_name == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        return FalkorDriver()
    else:
        raise ValueError(f"Unknown driver: {driver_name}")


if __name__ == '__main__':
    pytest.main([__file__])
