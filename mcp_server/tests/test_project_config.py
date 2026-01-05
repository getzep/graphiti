#!/usr/bin/env python3
"""Tests for project configuration detection."""

import json
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.project_config import find_project_config, load_project_config, ProjectConfig


def test_find_project_config_in_current_dir():
    """Test finding config in current directory."""
    print("Test: Finding config in current directory")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir).resolve()  # Resolve to handle symlinks
        config_file = tmpdir / ".graphiti.json"
        config_file.write_text(json.dumps({"group_id": "test-project"}))

        config = find_project_config(tmpdir)
        assert config is not None, f"Expected to find config but got None (tmpdir={tmpdir}, file_exists={config_file.exists()})"
        assert config.group_id == "test-project", f"Expected group_id='test-project' but got '{config.group_id}'"
        assert config.project_root == tmpdir, f"Expected project_root={tmpdir} but got {config.project_root}"
        print(f"  ✓ Found config: group_id={config.group_id}")


def test_find_project_config_in_parent_dir():
    """Test finding config in parent directory."""
    print("\nTest: Finding config in parent directory")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / ".graphiti.json"
        config_file.write_text(json.dumps({"group_id": "parent-project"}))

        # Create subdirectory
        subdir = tmpdir / "subdir" / "nested"
        subdir.mkdir(parents=True)

        config = find_project_config(subdir)
        assert config is not None
        assert config.group_id == "parent-project"
        print(f"  ✓ Found parent config: group_id={config.group_id}")


def test_no_config_found():
    """Test behavior when no config exists."""
    print("\nTest: No config found")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = find_project_config(Path(tmpdir))
        assert config is None
        print("  ✓ Correctly returned None when no config exists")


def test_invalid_config_json():
    """Test handling of invalid JSON."""
    print("\nTest: Invalid JSON in config file")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / ".graphiti.json"
        config_file.write_text("invalid json {{{")

        config = find_project_config(tmpdir)
        assert config is None
        print("  ✓ Correctly returned None for invalid JSON")


def test_missing_group_id():
    """Test handling of config without group_id."""
    print("\nTest: Config missing group_id field")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / ".graphiti.json"
        config_file.write_text(json.dumps({"description": "no group id"}))

        config = find_project_config(tmpdir)
        assert config is None
        print("  ✓ Correctly returned None when group_id is missing")


def test_config_with_description():
    """Test loading config with description."""
    print("\nTest: Config with description")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / ".graphiti.json"
        test_data = {"group_id": "test-with-desc", "description": "Test project description"}
        config_file.write_text(json.dumps(test_data))

        config = find_project_config(tmpdir)
        assert config is not None
        assert config.group_id == "test-with-desc"
        assert config.description == "Test project description"
        print(f"  ✓ Loaded config with description: {config.description}")


def test_config_with_invalid_description():
    """Test handling of invalid description type."""
    print("\nTest: Config with invalid description type")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / ".graphiti.json"
        config_file.write_text(json.dumps({"group_id": "test-project", "description": 123}))

        config = find_project_config(tmpdir)
        assert config is not None
        assert config.group_id == "test-project"
        assert config.description is None  # Should be reset to None
        print("  ✓ Invalid description type handled correctly")


def test_config_with_invalid_group_id_type():
    """Test handling of invalid group_id type."""
    print("\nTest: Config with invalid group_id type")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_file = tmpdir / ".graphiti.json"
        config_file.write_text(json.dumps({"group_id": 12345}))

        config = find_project_config(tmpdir)
        assert config is None
        print("  ✓ Correctly returned None when group_id is not a string")


def test_nested_configs_uses_closest():
    """Test that nested configs use the closest one."""
    print("\nTest: Nested config files - uses closest")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create parent config
        parent_config = tmpdir / ".graphiti.json"
        parent_config.write_text(json.dumps({"group_id": "parent-project"}))

        # Create subdirectory with its own config
        subdir = tmpdir / "subdir"
        subdir.mkdir()
        child_config = subdir / ".graphiti.json"
        child_config.write_text(json.dumps({"group_id": "child-project"}))

        # Search from a nested directory
        nested = subdir / "nested"
        nested.mkdir()

        config = find_project_config(nested)
        assert config is not None
        assert config.group_id == "child-project"  # Should use child, not parent
        print(f"  ✓ Used closest config: group_id={config.group_id}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Project Configuration Tests")
    print("=" * 60)

    tests = [
        test_find_project_config_in_current_dir,
        test_find_project_config_in_parent_dir,
        test_no_config_found,
        test_invalid_config_json,
        test_missing_group_id,
        test_config_with_description,
        test_config_with_invalid_description,
        test_config_with_invalid_group_id_type,
        test_nested_configs_uses_closest,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
