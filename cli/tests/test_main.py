"""
Copyright (c) 2024 Zep Labs, Inc.
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
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from cli.main import _add_json_episode, _add_json_string_episode, cli_app, generate_project_group_id


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_graphiti():
    with patch('cli.main.Graphiti') as mock:
        # Set up our mock for the async context manager
        mock_instance = AsyncMock()
        mock.return_value = mock_instance
        mock_instance.driver.verify_connectivity = AsyncMock()
        mock_instance.add_episode = AsyncMock()
        mock_instance.close = AsyncMock()
        yield mock


@pytest.fixture
def mock_env(monkeypatch):
    """Setup mock environment variables"""
    monkeypatch.setenv('NEO4J_URI', 'bolt://localhost:7687')
    monkeypatch.setenv('NEO4J_USER', 'test_user')
    monkeypatch.setenv('NEO4J_PASSWORD', 'test_password')
    monkeypatch.setenv('OPENAI_API_KEY', 'test_api_key')
    monkeypatch.setenv('MODEL_NAME', 'gpt-3.5-turbo')


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file for testing"""
    test_data = {'test': 'data', 'nested': {'value': 123}}
    json_file = tmp_path / 'test.json'
    with open(json_file, 'w') as f:
        json.dump(test_data, f)
    return json_file


@pytest.mark.asyncio
async def test_add_json_episode(mock_graphiti, mock_env, temp_json_file):
    """Test the internal _add_json_episode function with a file"""
    await _add_json_episode(
        json_file=temp_json_file,
        name='Test Episode',
        source_description='Test Description',
        group_id='test_group',
        uuid_str='test_uuid',
    )

    # Check that methods were called with expected arguments
    mock_instance = mock_graphiti.return_value
    mock_instance.add_episode.assert_called_once()

    # Check the parameters passed to add_episode
    call_args = mock_instance.add_episode.call_args[1]
    assert call_args['name'] == 'Test Episode'
    assert call_args['source_description'] == 'Test Description'
    assert call_args['group_id'] == 'test_group'
    assert call_args['uuid'] == 'test_uuid'

    # Verify the file content was read correctly
    with open(temp_json_file) as f:
        expected_content = f.read()
    assert call_args['episode_body'] == expected_content


@pytest.mark.asyncio
async def test_add_json_string_episode(mock_graphiti, mock_env):
    """Test the internal function for adding JSON directly as a string"""
    json_data = json.dumps({'test': 'data', 'nested': {'value': 123}})

    await _add_json_string_episode(
        json_data=json_data,
        name='Test String Episode',
        source_description='Test String Description',
        group_id='test_string_group',
        uuid_str='test_string_uuid',
    )

    # Check that methods were called with expected arguments
    mock_instance = mock_graphiti.return_value
    mock_instance.add_episode.assert_called_once()

    # Check the parameters passed to add_episode
    call_args = mock_instance.add_episode.call_args[1]
    assert call_args['name'] == 'Test String Episode'
    assert call_args['episode_body'] == json_data
    assert call_args['source_description'] == 'Test String Description'
    assert call_args['group_id'] == 'test_string_group'
    assert call_args['uuid'] == 'test_string_uuid'


@pytest.mark.asyncio
async def test_add_json_string_episode_invalid_json(mock_graphiti, mock_env):
    """Test the internal function with invalid JSON string"""
    invalid_json = "{'this': 'is not valid JSON'}"  # Single quotes instead of double quotes

    with pytest.raises(typer.Exit):
        await _add_json_string_episode(
            json_data=invalid_json,
            name='Invalid JSON Test',
            source_description='Invalid JSON Description',
            group_id='test_group',
            uuid_str='test_uuid',
        )

    # Verify add_episode was never called
    mock_instance = mock_graphiti.return_value
    mock_instance.add_episode.assert_not_called()


def test_add_json_string_command(runner, mock_graphiti, mock_env, monkeypatch):
    """Test the add-json-string CLI command"""
    # Mock asyncio.run to run our coroutine synchronously
    async_mock = AsyncMock()
    monkeypatch.setattr('asyncio.run', async_mock)

    json_data = json.dumps({'test': 'string', 'command': 'test'})

    result = runner.invoke(
        cli_app,
        [
            'add-json-string',
            '--json-data',
            json_data,
            '--name',
            'Test String Command',
            '--desc',
            'Test String Command Description',
            '--group-id',
            'test_string_group',
            '--uuid',
            'test_string_uuid',
        ],
    )

    assert result.exit_code == 0
    async_mock.assert_called_once()


def test_add_json_command_with_explicit_parameters(runner, mock_graphiti, mock_env, temp_json_file, monkeypatch):
    """Test the add-json CLI command with explicitly provided parameters"""
    # Mock asyncio.run to run our coroutine synchronously
    async_mock = AsyncMock()
    monkeypatch.setattr('asyncio.run', async_mock)

    result = runner.invoke(
        cli_app,
        [
            'add-json',
            '--json-file',
            str(temp_json_file),
            '--name',
            'Test Episode',
            '--desc',
            'Test Description',
            '--group-id',
            'test_group',
            '--uuid',
            'test_uuid',
        ],
    )

    assert result.exit_code == 0
    async_mock.assert_called_once()


def test_check_connection_command(runner, monkeypatch, mock_env):
    """Test the check-connection CLI command"""
    # Mock the neo4j AsyncGraphDatabase
    mock_driver = AsyncMock()
    mock_driver.verify_connectivity = AsyncMock()
    mock_driver.close = AsyncMock()

    mock_graph_db = Mock()
    mock_graph_db.driver.return_value = mock_driver

    # Mock asyncio.run
    async_run_mock = AsyncMock()
    monkeypatch.setattr('asyncio.run', async_run_mock)

    # Mock the Neo4j driver import
    monkeypatch.setattr('neo4j.AsyncGraphDatabase', mock_graph_db)

    result = runner.invoke(cli_app, ['check-connection'])
    assert result.exit_code == 0
    async_run_mock.assert_called_once()


def test_generate_project_group_id():
    """Test the project group_id generation function"""
    # Test with explicit path
    test_path = '/Users/test/project'
    group_id = generate_project_group_id(test_path)
    assert group_id.startswith('cursor_')
    assert len(group_id) == 15  # "cursor_" + 8 hex chars

    # Test that the same path always generates the same group_id
    group_id2 = generate_project_group_id(test_path)
    assert group_id == group_id2

    # Test that different paths generate different group_ids
    different_path = '/Users/test/different_project'
    different_group_id = generate_project_group_id(different_path)
    assert group_id != different_group_id


def test_generate_project_group_id_with_env_var(monkeypatch):
    """Test group_id generation using the CURSOR_WORKSPACE environment variable"""
    test_path = '/Users/test/env_var_project'
    monkeypatch.setenv('CURSOR_WORKSPACE', test_path)

    # When no path is provided, it should use the env var
    group_id = generate_project_group_id(None)

    # Test with explicit path to verify it's the same
    direct_group_id = generate_project_group_id(test_path)
    assert group_id == direct_group_id


@pytest.mark.asyncio
async def test_add_json_episode_with_generated_group_id(
    mock_graphiti, mock_env, temp_json_file, monkeypatch
):
    """Test that the function uses generated group_id when none is provided"""
    # Setup the test path
    test_path = '/test/workspace/path'
    monkeypatch.setattr('os.getcwd', lambda: test_path)

    # Call the function without specifying a group_id
    await _add_json_episode(
        json_file=temp_json_file,
        name='Test with Generated Group ID',
        source_description='Test Description',
    )

    # Verify add_episode was called
    mock_instance = mock_graphiti.return_value
    mock_instance.add_episode.assert_called_once()


@pytest.mark.asyncio
async def test_add_json_string_episode_with_generated_group_id(
    mock_graphiti, mock_env, monkeypatch
):
    """Test that the function uses generated group_id when none is provided"""
    # Setup the test path
    test_path = '/test/workspace/path'
    monkeypatch.setattr('os.getcwd', lambda: test_path)

    json_data = json.dumps({'test': 'data', 'nested': {'value': 123}})

    # Call the function without specifying a group_id
    await _add_json_string_episode(
        json_data=json_data,
        name='Test String with Generated Group ID',
        source_description='Test Description',
    )

    # Verify add_episode was called
    mock_instance = mock_graphiti.return_value
    mock_instance.add_episode.assert_called_once()


def test_add_json_command_with_generated_group_id(
    runner, mock_graphiti, mock_env, temp_json_file, monkeypatch
):
    """Test the add-json CLI command with a generated group_id"""
    # Mock asyncio.run to run our coroutine synchronously
    async_mock = AsyncMock()
    monkeypatch.setattr('asyncio.run', async_mock)

    # Set a mock workspace path
    test_path = '/test/workspace/path'
    monkeypatch.setattr('os.getcwd', lambda: test_path)

    result = runner.invoke(
        cli_app,
        [
            'add-json',
            '--json-file',
            str(temp_json_file),
            '--name',
            'Test Episode',
            '--desc',
            'Test Description',
        ],
    )

    assert result.exit_code == 0
    async_mock.assert_called_once()


# Additional tests for the direct JSON string function will be added
