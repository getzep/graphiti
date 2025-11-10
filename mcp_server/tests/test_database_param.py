#!/usr/bin/env python3
"""Test Neo4j database parameter configuration."""

import os
import sys
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_neo4j_database_parameter():
    """Test that Neo4j database parameter is included in configuration."""
    from src.config.schema import GraphitiConfig
    from src.services.factories import DatabaseDriverFactory

    print('\n' + '='*70)
    print('Testing Neo4j Database Parameter Configuration')
    print('='*70 + '\n')

    # Test 1: Default database value
    print('Test 1: Default database value')
    config = GraphitiConfig()
    db_config = DatabaseDriverFactory.create_config(config.database)

    assert 'database' in db_config, 'Database parameter missing from config!'
    print(f'  ✓ Database parameter present in config')
    print(f'  ✓ Default database value: {db_config["database"]}')
    assert db_config['database'] == 'neo4j', f'Expected default "neo4j", got {db_config["database"]}'
    print(f'  ✓ Default value matches expected: neo4j\n')

    # Test 2: Environment variable override
    print('Test 2: Environment variable override')
    os.environ['NEO4J_DATABASE'] = 'graphiti'
    config2 = GraphitiConfig()
    db_config2 = DatabaseDriverFactory.create_config(config2.database)

    assert 'database' in db_config2, 'Database parameter missing from config!'
    print(f'  ✓ Database parameter present in config')
    print(f'  ✓ Overridden database value: {db_config2["database"]}')
    assert db_config2['database'] == 'graphiti', f'Expected "graphiti", got {db_config2["database"]}'
    print(f'  ✓ Environment override works correctly\n')

    # Clean up
    del os.environ['NEO4J_DATABASE']

    # Test 3: Verify all required parameters are present
    print('Test 3: Verify all required Neo4j parameters')
    required_params = ['uri', 'user', 'password', 'database']
    for param in required_params:
        assert param in db_config, f'Required parameter "{param}" missing!'
        print(f'  ✓ {param}: present')

    print('\n' + '='*70)
    print('✅ All database parameter tests passed!')
    print('='*70)
    print('\nSummary:')
    print('  - database parameter is included in Neo4j config')
    print('  - Default value is "neo4j"')
    print('  - Environment variable NEO4J_DATABASE override works')
    print('  - All required parameters (uri, user, password, database) present')
    print()


if __name__ == '__main__':
    try:
        test_neo4j_database_parameter()
    except AssertionError as e:
        print(f'\n❌ Test failed: {e}\n')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Unexpected error: {e}\n')
        import traceback
        traceback.print_exc()
        sys.exit(1)
