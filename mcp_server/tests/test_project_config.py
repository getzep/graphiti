#!/usr/bin/env python3
"""Tests for project configuration detection."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.project_config import find_project_config


def _with_env_dir(env_dir: Path, func):
    """Helper to run a function with GRAPHITI_PROJECT_DIR set."""
    old_env = os.environ.get('GRAPHITI_PROJECT_DIR')
    try:
        os.environ['GRAPHITI_PROJECT_DIR'] = str(env_dir)
        return func()
    finally:
        if old_env is None:
            os.environ.pop('GRAPHITI_PROJECT_DIR', None)
        else:
            os.environ['GRAPHITI_PROJECT_DIR'] = old_env


def test_find_project_config_in_current_dir():
    """Test finding config in directory specified by GRAPHITI_PROJECT_DIR."""
    print('Test: Finding config in GRAPHITI_PROJECT_DIR')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()  # Resolve to handle symlinks
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text(json.dumps({'group_id': 'test-project'}))

        old_env = os.environ.get('GRAPHITI_PROJECT_DIR')
        try:
            os.environ['GRAPHITI_PROJECT_DIR'] = str(tmpdir)
            config = find_project_config()
            assert config is not None, (
                f'Expected to find config but got None (tmpdir={tmpdir}, file_exists={config_file.exists()})'
            )
            assert config.group_id == 'test-project', (
                f"Expected group_id='test-project' but got '{config.group_id}'"
            )
            assert config.project_root == tmpdir, (
                f'Expected project_root={tmpdir} but got {config.project_root}'
            )
            print(f'  ✓ Found config: group_id={config.group_id}')
        finally:
            if old_env is None:
                os.environ.pop('GRAPHITI_PROJECT_DIR', None)
            else:
                os.environ['GRAPHITI_PROJECT_DIR'] = old_env


def test_find_project_config_in_parent_dir():
    """Test finding config in parent directory via upward search."""
    print('\nTest: Finding config in parent directory (upward search)')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text(json.dumps({'group_id': 'parent-project'}))

        # Create subdirectory
        subdir = tmpdir / 'subdir' / 'nested'
        subdir.mkdir(parents=True)

        old_env = os.environ.get('GRAPHITI_PROJECT_DIR')
        try:
            # Set env var to nested directory - should search upward and find parent config
            os.environ['GRAPHITI_PROJECT_DIR'] = str(subdir)
            config = find_project_config()
            assert config is not None
            assert config.group_id == 'parent-project'
            print(f'  ✓ Found parent config: group_id={config.group_id}')
        finally:
            if old_env is None:
                os.environ.pop('GRAPHITI_PROJECT_DIR', None)
            else:
                os.environ['GRAPHITI_PROJECT_DIR'] = old_env


def test_no_config_found():
    """Test behavior when GRAPHITI_PROJECT_DIR is not set."""
    print('\nTest: Returns None when GRAPHITI_PROJECT_DIR not set')
    old_env = os.environ.get('GRAPHITI_PROJECT_DIR')
    try:
        # Clear the environment variable
        os.environ.pop('GRAPHITI_PROJECT_DIR', None)
        config = find_project_config()
        assert config is None
        print('  ✓ Correctly returned None when GRAPHITI_PROJECT_DIR not set')
    finally:
        if old_env is not None:
            os.environ['GRAPHITI_PROJECT_DIR'] = old_env


def test_no_config_in_env_dir():
    """Test behavior when config does not exist in GRAPHITI_PROJECT_DIR."""
    print('\nTest: No config found in GRAPHITI_PROJECT_DIR')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Don't create any .graphiti.json file

        old_env = os.environ.get('GRAPHITI_PROJECT_DIR')
        try:
            os.environ['GRAPHITI_PROJECT_DIR'] = str(tmpdir)
            config = find_project_config()
            assert config is None
            print('  ✓ Correctly returned None when no config exists in env dir')
        finally:
            if old_env is None:
                os.environ.pop('GRAPHITI_PROJECT_DIR', None)
            else:
                os.environ['GRAPHITI_PROJECT_DIR'] = old_env


def test_invalid_config_json():
    """Test handling of invalid JSON."""
    print('\nTest: Invalid JSON in config file')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text('invalid json {{{')

        def test():
            config = find_project_config()
            assert config is None
            print('  ✓ Correctly returned None for invalid JSON')

        _with_env_dir(tmpdir, test)


def test_missing_group_id():
    """Test handling of config without group_id."""
    print('\nTest: Config missing group_id field')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text(json.dumps({'description': 'no group id'}))

        def test():
            config = find_project_config()
            assert config is None
            print('  ✓ Correctly returned None when group_id is missing')

        _with_env_dir(tmpdir, test)


def test_config_with_description():
    """Test loading config with description."""
    print('\nTest: Config with description')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        test_data = {'group_id': 'test-with-desc', 'description': 'Test project description'}
        config_file.write_text(json.dumps(test_data))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.group_id == 'test-with-desc'
            assert config.description == 'Test project description'
            print(f'  ✓ Loaded config with description: {config.description}')

        _with_env_dir(tmpdir, test)


def test_config_with_invalid_description():
    """Test handling of invalid description type."""
    print('\nTest: Config with invalid description type')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text(json.dumps({'group_id': 'test-project', 'description': 123}))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.group_id == 'test-project'
            assert config.description is None  # Should be reset to None
            print('  ✓ Invalid description type handled correctly')

        _with_env_dir(tmpdir, test)


def test_config_with_invalid_group_id_type():
    """Test handling of invalid group_id type."""
    print('\nTest: Config with invalid group_id type')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text(json.dumps({'group_id': 12345}))

        def test():
            config = find_project_config()
            assert config is None
            print('  ✓ Correctly returned None when group_id is not a string')

        _with_env_dir(tmpdir, test)


def test_nested_configs_uses_closest():
    """Test that nested configs use the closest one."""
    print('\nTest: Nested config files - uses closest')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create parent config
        parent_config = tmpdir / '.graphiti.json'
        parent_config.write_text(json.dumps({'group_id': 'parent-project'}))

        # Create subdirectory with its own config
        subdir = tmpdir / 'subdir'
        subdir.mkdir()
        child_config = subdir / '.graphiti.json'
        child_config.write_text(json.dumps({'group_id': 'child-project'}))

        # Search from a nested directory
        nested = subdir / 'nested'
        nested.mkdir()

        def test():
            config = find_project_config()
            assert config is not None
            assert config.group_id == 'child-project'  # Should use child, not parent
            print(f'  ✓ Used closest config: group_id={config.group_id}')

        _with_env_dir(nested, test)


def test_config_with_shared_group_ids():
    """Test loading config with shared_group_ids."""
    print('\nTest: Config with shared_group_ids')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        test_data = {'group_id': 'my-app', 'shared_group_ids': ['user-common', 'team-standards']}
        config_file.write_text(json.dumps(test_data))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.shared_group_ids == ['user-common', 'team-standards']
            assert config.has_shared_config is True
            print(f'  ✓ Loaded shared_group_ids: {config.shared_group_ids}')

        _with_env_dir(tmpdir, test)


def test_config_with_shared_entity_types():
    """Test loading config with shared_entity_types."""
    print('\nTest: Config with shared_entity_types')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        test_data = {
            'group_id': 'my-app',
            'shared_entity_types': ['Preference', 'Procedure', 'Requirement'],
        }
        config_file.write_text(json.dumps(test_data))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.shared_entity_types == ['Preference', 'Procedure', 'Requirement']
            print(f'  ✓ Loaded shared_entity_types: {config.shared_entity_types}')

        _with_env_dir(tmpdir, test)


def test_config_with_shared_patterns():
    """Test loading config with shared_patterns."""
    print('\nTest: Config with shared_patterns')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        test_data = {
            'group_id': 'my-app',
            'shared_patterns': ['用户.*偏好', 'programming style', 'coding convention'],
        }
        config_file.write_text(json.dumps(test_data))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.shared_patterns == [
                '用户.*偏好',
                'programming style',
                'coding convention',
            ]
            print(f'  ✓ Loaded shared_patterns: {config.shared_patterns}')

        _with_env_dir(tmpdir, test)


def test_config_with_write_strategy():
    """Test loading config with write_strategy."""
    print('\nTest: Config with write_strategy')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        test_data = {'group_id': 'my-app', 'write_strategy': 'smart_split'}
        config_file.write_text(json.dumps(test_data))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.write_strategy == 'smart_split'
            print(f'  ✓ Loaded write_strategy: {config.write_strategy}')

        _with_env_dir(tmpdir, test)


def test_config_with_all_shared_fields():
    """Test loading config with all shared configuration fields."""
    print('\nTest: Config with all shared fields')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        test_data = {
            'group_id': 'my-app',
            'description': 'My application',
            'shared_group_ids': ['user-common'],
            'shared_entity_types': ['Preference'],
            'shared_patterns': ['偏好'],
            'write_strategy': 'simple',
        }
        config_file.write_text(json.dumps(test_data))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.group_id == 'my-app'
            assert config.description == 'My application'
            assert config.shared_group_ids == ['user-common']
            assert config.shared_entity_types == ['Preference']
            assert config.shared_patterns == ['偏好']
            assert config.write_strategy == 'simple'
            assert config.has_shared_config is True
            print('  ✓ Loaded all shared fields correctly')

        _with_env_dir(tmpdir, test)


def test_config_with_invalid_shared_group_ids_type():
    """Test handling of invalid shared_group_ids type."""
    print('\nTest: Config with invalid shared_group_ids type')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text(json.dumps({'group_id': 'my-app', 'shared_group_ids': 'not-a-list'}))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.shared_group_ids == []  # Should default to empty list
            print('  ✓ Invalid shared_group_ids type handled correctly')

        _with_env_dir(tmpdir, test)


def test_config_defaults_when_shared_fields_missing():
    """Test that shared fields default to empty/when not specified."""
    print('\nTest: Config defaults when shared fields missing')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / '.graphiti.json'
        config_file.write_text(json.dumps({'group_id': 'minimal-app'}))

        def test():
            config = find_project_config()
            assert config is not None
            assert config.shared_group_ids == []
            assert config.shared_entity_types == []
            assert config.shared_patterns == []
            assert config.write_strategy == 'simple'  # Default value
            assert config.has_shared_config is False
            print('  ✓ Shared fields defaulted correctly')

        _with_env_dir(tmpdir, test)


def test_graphiti_project_dir_env_search_upward():
    """Test that GRAPHITI_PROJECT_DIR still searches upward from that directory."""
    print('\nTest: GRAPHITI_PROJECT_DIR searches upward from env dir')
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create parent config (the actual project root)
        project_root = tmpdir / 'project-root'
        project_root.mkdir()
        root_config = project_root / '.graphiti.json'
        root_config.write_text(
            json.dumps({'group_id': 'actual-project', 'description': 'Root project'})
        )

        # Create nested directory (where env var points)
        nested_dir = project_root / 'nested' / 'deep'

        old_env = os.environ.get('GRAPHITI_PROJECT_DIR')
        try:
            os.environ['GRAPHITI_PROJECT_DIR'] = str(nested_dir)

            # Should search upward from nested_dir and find root config
            config = find_project_config()
            assert config is not None
            assert config.group_id == 'actual-project'
            assert config.description == 'Root project'
            print(f'  ✓ Found config by searching upward from env dir: group_id={config.group_id}')

        finally:
            # Restore old environment variable
            if old_env is None:
                os.environ.pop('GRAPHITI_PROJECT_DIR', None)
            else:
                os.environ['GRAPHITI_PROJECT_DIR'] = old_env


def run_all_tests():
    """Run all tests."""
    print('=' * 60)
    print('Running Project Configuration Tests')
    print('=' * 60)

    tests = [
        test_find_project_config_in_current_dir,
        test_find_project_config_in_parent_dir,
        test_no_config_found,
        test_no_config_in_env_dir,
        test_invalid_config_json,
        test_missing_group_id,
        test_config_with_description,
        test_config_with_invalid_description,
        test_config_with_invalid_group_id_type,
        test_nested_configs_uses_closest,
        test_config_with_shared_group_ids,
        test_config_with_shared_entity_types,
        test_config_with_shared_patterns,
        test_config_with_write_strategy,
        test_config_with_all_shared_fields,
        test_config_with_invalid_shared_group_ids_type,
        test_config_defaults_when_shared_fields_missing,
        test_graphiti_project_dir_env_search_upward,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f'  ✗ FAILED: {e}')
            failed += 1
        except Exception as e:
            print(f'  ✗ ERROR: {e}')
            failed += 1

    print('\n' + '=' * 60)
    print(f'Results: {passed} passed, {failed} failed')
    print('=' * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
