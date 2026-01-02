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
# cli/main.py
import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv

# Attempt to load from .env file in the current directory OR the project root
# This helps when running via alias from another directory
load_dotenv()  # Loads .env from CWD if present
project_root_env = (
    Path(__file__).parent.parent / '.env'
)  # Check ../.env relative to script location
if project_root_env.is_file():
    load_dotenv(
        dotenv_path=project_root_env, override=True
    )  # Load root .env, override CWD .env if values exist

# Ensure graphiti_core is importable (assumes it's installed in the env)
try:
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import LLMClient, LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.nodes import EpisodeType
except ImportError as err:
    print(f'Error importing graphiti-core: {err}')
    print(
        "Please ensure graphiti-core is installed in the environment (e.g., via 'uv sync' in the project root)."
    )
    raise typer.Exit(code=1) from err


# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Environment Variable Loading ---
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'demodemo')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')  # Defaulting to 3.5-turbo

cli_app = typer.Typer(
    help='A CLI tool to interact with Graphiti Core, primarily for adding JSON episodes.'
)

# Define app as an alias for cli_app to support package entry points
app = cli_app


# --- Typer Option Constants ---
# add_json options
JSON_FILE_OPTION = typer.Option(
    ...,
    '--json-file',
    '-f',
    help='Path to the JSON file to ingest.',
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
)
NAME_OPTION = typer.Option(..., '--name', '-n', help='Name for the episode.')
SOURCE_DESC_OPTION = typer.Option(None, '--desc', '-d', help='Description of the data source.')
GROUP_ID_OPTION = typer.Option(
    None,
    '--group-id',
    '-g',
    help='Optional group ID for the graph. If not provided, generates one based on workspace path.',
)
UUID_OPTION = typer.Option(None, '--uuid', help='Optional UUID for the episode.')
WORKSPACE_PATH_OPTION = typer.Option(
    None,
    '--workspace',
    '-w',
    help='Workspace path for generating consistent group_id. If not provided, uses CURSOR_WORKSPACE env var or cwd.',
)

# add_json_string options
JSON_DATA_OPTION = typer.Option(
    ..., '--json-data', '-d', help='JSON string to ingest (must be valid JSON).'
)
SOURCE_DESC_STRING_OPTION = typer.Option(None, '--desc', '-s', help='Description of the data source.')

# search_nodes options
QUERY_OPTION = typer.Option(..., '--query', '-q', help='Search query string')
MAX_NODES_OPTION = typer.Option(10, '--max', '-m', help='Maximum number of nodes to return')
CENTER_NODE_UUID_OPTION = typer.Option(
    None, '--center', '-c', help='Optional UUID of a node to center the search around'
)
ENTITY_OPTION = typer.Option(
    '',
    '--entity',
    '-e',
    help="Optional entity type to filter results (e.g., 'Preference', 'Procedure')",
)

# search_facts options
MAX_FACTS_OPTION = typer.Option(10, '--max', '-m', help='Maximum number of facts to return')

# get_episodes options
LAST_N_OPTION = typer.Option(
    10, '--last', '-n', help='Number of most recent episodes to retrieve'
)

# delete options
CONFIRM_OPTION = typer.Option(False, '--confirm', help='Confirmation flag is required for deletion')
SKIP_PREVIEW_OPTION = typer.Option(False, '--skip-preview', help='Skip the preview step')

# clear_graph options
FORCE_OPTION = typer.Option(False, '--force', help='Force flag as secondary confirmation')

# General options used in multiple commands
UUID_EDGE_OPTION = typer.Option(..., '--uuid', '-u', help='UUID of the entity edge to retrieve/delete')
UUID_EPISODE_OPTION = typer.Option(..., '--uuid', '-u', help='UUID of the episode to delete')
CONFIRM_FLAG_OPTION = typer.Option(False, '--confirm', help='Initial confirmation flag')


def generate_project_group_id(workspace_path: str = None) -> str:
    """
    Generate a consistent group_id based on the workspace path.

    This mimics the approach used by the MCP server to ensure consistent group_ids
    across both the CLI and the MCP server. If no workspace path is provided,
    it attempts to use the CURSOR_WORKSPACE environment variable, or falls back to
    the current working directory.

    Args:
        workspace_path: Optional path to workspace, defaults to CURSOR_WORKSPACE or cwd

    Returns:
        A consistent group_id in the format "cursor_<md5_hash[:8]>"
    """
    # Determine workspace path - first try provided path, then env var, then cwd
    effective_path = workspace_path
    if not effective_path:
        effective_path = os.getenv('CURSOR_WORKSPACE')
    if not effective_path:
        effective_path = os.getcwd()

    # Generate MD5 hash of the path and take first 8 chars
    path_hash = hashlib.md5(effective_path.encode()).hexdigest()[:8]
    return f'cursor_{path_hash}'


async def _add_json_episode(
    json_file: Path,
    name: str,
    source_description: str | None = None,
    group_id: str | None = None,
    uuid_str: str | None = None,
):
    """Internal async function to handle Graphiti client initialization and call."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None  # Initialize to None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )
        # Test connection (optional but recommended)
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        if not json_file.is_file():
            logger.error(f'JSON file not found: {json_file.resolve()}')
            raise typer.Exit(code=1)

        logger.info(f'Reading JSON data from: {json_file.resolve()}')
        try:
            # Read the file content directly as a string
            with open(json_file, encoding='utf-8') as f:
                episode_body_str = f.read()
            # Optional: Validate if it's valid JSON (Graphiti core might do this too)
            json.loads(episode_body_str)
            logger.info(f'Successfully read and validated JSON from {json_file.name}')
        except FileNotFoundError as err:
            logger.error(f'Could not read JSON file: {json_file.resolve()}')
            raise typer.Exit(code=1) from err
        except json.JSONDecodeError as err:
            logger.error(f'Invalid JSON content in file {json_file.resolve()}: {err}')
            raise typer.Exit(code=1) from err
        except Exception as err:
            logger.error(f'Error reading file {json_file.resolve()}: {err}')
            raise typer.Exit(code=1) from err

        effective_group_id = group_id  # Graphiti client handles None as default
        logger.info(
            f"Adding episode '{name}' with source 'json'. Group ID: {effective_group_id or 'default'}"
        )
        await graphiti_client.add_episode(
            name=name,
            episode_body=episode_body_str,  # Pass the raw JSON string
            source=EpisodeType.json,
            source_description=source_description or '',
            group_id=effective_group_id,
            uuid=uuid_str,
            reference_time=datetime.now(timezone.utc),  # Add the current time as reference_time
        )
        logger.info(f"Successfully added episode '{name}'.")

    except ImportError as err:
        # Catch potential Neo4j driver import issues if not installed correctly
        logger.error(f'Import error, potentially Neo4j driver: {err}', exc_info=True)
        print("Ensure Neo4j dependencies are installed correctly via 'uv sync'.")
        raise typer.Exit(code=1) from err
    except Exception as err:
        logger.error(
            f'An error occurred during Graphiti operation: {err}', exc_info=True
        )  # Log traceback
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


async def _add_json_string_episode(
    json_data: str,
    name: str,
    source_description: str | None = None,
    group_id: str | None = None,
    uuid_str: str | None = None,
):
    """Internal async function to handle Graphiti client initialization and add a JSON string."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None  # Initialize to None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )
        # Test connection (optional but recommended)
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # Validate the JSON string
        try:
            # Validate if it's valid JSON (Graphiti core might do this too)
            json.loads(json_data)
            logger.info('Successfully validated JSON data')
        except json.JSONDecodeError as err:
            logger.error(f'Invalid JSON content: {err}')
            raise typer.Exit(code=1) from err

        effective_group_id = group_id  # Graphiti client handles None as default
        logger.info(
            f"Adding episode '{name}' with source 'json'. Group ID: {effective_group_id or 'default'}"
        )
        await graphiti_client.add_episode(
            name=name,
            episode_body=json_data,  # Pass the raw JSON string
            source=EpisodeType.json,
            source_description=source_description or '',
            group_id=effective_group_id,
            uuid=uuid_str,
            reference_time=datetime.now(timezone.utc),  # Add the current time as reference_time
        )
        logger.info(f"Successfully added episode '{name}'.")

    except ImportError as err:
        # Catch potential Neo4j driver import issues if not installed correctly
        logger.error(f'Import error, potentially Neo4j driver: {err}', exc_info=True)
        print("Ensure Neo4j dependencies are installed correctly via 'uv sync'.")
        raise typer.Exit(code=1) from err
    except Exception as err:
        logger.error(
            f'An error occurred during Graphiti operation: {err}', exc_info=True
        )  # Log traceback
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


@cli_app.command()
def add_json(
    json_file: Path = JSON_FILE_OPTION,
    name: str = NAME_OPTION,
    source_description: str | None = SOURCE_DESC_OPTION,
    group_id: str | None = GROUP_ID_OPTION,
    uuid_str: str | None = UUID_OPTION,
    workspace_path: str | None = WORKSPACE_PATH_OPTION,
):
    """
    Adds a JSON file content as an episode to Graphiti using graphiti-core.
    Reads connection details and API keys from environment variables or a .env file.
    """
    # Typer's resolve_path=True handles making it absolute based on CWD
    logger.info('Running add-json command...')
    # Generate group_id if not provided
    effective_group_id = group_id
    if not effective_group_id:
        effective_group_id = generate_project_group_id(workspace_path)
        logger.info(f'Using generated group_id: {effective_group_id}')

    asyncio.run(
        _add_json_episode(json_file, name, source_description, effective_group_id, uuid_str)
    )


@cli_app.command()
def add_json_string(
    json_data: str = JSON_DATA_OPTION,
    name: str = NAME_OPTION,
    source_description: str | None = SOURCE_DESC_STRING_OPTION,
    group_id: str | None = GROUP_ID_OPTION,
    uuid_str: str | None = UUID_OPTION,
    workspace_path: str | None = WORKSPACE_PATH_OPTION,
):
    """
    Adds a JSON string directly as an episode to Graphiti using graphiti-core.
    The JSON data must be a valid JSON string.
    Reads connection details and API keys from environment variables or a .env file.
    """
    logger.info('Running add-json-string command...')
    # Generate group_id if not provided
    effective_group_id = group_id
    if not effective_group_id:
        effective_group_id = generate_project_group_id(workspace_path)
        logger.info(f'Using generated group_id: {effective_group_id}')

    asyncio.run(
        _add_json_string_episode(json_data, name, source_description, effective_group_id, uuid_str)
    )


# Add a simple check command
@cli_app.command()
def check_connection():
    """Checks the connection to Neo4j using credentials from environment variables."""

    async def _check():
        logger.info('Checking Neo4j connection...')
        if not NEO4J_USER or not NEO4J_PASSWORD:
            logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
            raise typer.Exit(code=1)
        driver = None
        try:
            # Import here to avoid error if neo4j driver isn't installed yet
            from neo4j import AsyncGraphDatabase

            logger.info(f'Attempting connection to {NEO4J_URI}...')
            driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            await driver.verify_connectivity()
            logger.info(f'Successfully connected to Neo4j at {NEO4J_URI}')
        except ImportError as e:
            logger.error(f'Failed to import Neo4j driver: {e}')
            print("Ensure Neo4j dependencies are installed correctly via 'uv sync'.")
        except Exception as e:
            logger.error(f'Failed to connect to Neo4j: {e}')
        finally:
            if driver:
                await driver.close()
                logger.info('Neo4j driver connection closed.')

    logger.info('Running check-connection command...')
    asyncio.run(_check())


@cli_app.command()
def search_nodes(
    query: str = QUERY_OPTION,
    group_id: str | None = GROUP_ID_OPTION,
    max_nodes: int = MAX_NODES_OPTION,
    center_node_uuid: str | None = CENTER_NODE_UUID_OPTION,
    entity: str = ENTITY_OPTION,
    workspace_path: str | None = WORKSPACE_PATH_OPTION,
):
    """
    Searches the knowledge graph for relevant nodes matching the query.
    Returns node summaries that contain information about all of a node's relationships.

    Results can be filtered by entity type (e.g., "Preference", "Procedure").
    If a center node UUID is provided, results will be ranked by proximity to that node.
    """
    logger.info('Running search-nodes command...')

    # Generate group_id if not provided
    effective_group_id = group_id
    if not effective_group_id:
        effective_group_id = generate_project_group_id(workspace_path)
        logger.info(f'Using generated group_id: {effective_group_id}')

    group_ids = [effective_group_id]  # Convert to list for the API

    asyncio.run(_search_nodes(query, group_ids, max_nodes, center_node_uuid, entity))


async def _search_nodes(
    query: str,
    group_ids: list[str],
    max_nodes: int,
    center_node_uuid: str | None = None,
    entity: str = '',
):
    """Internal async function to handle node search using Graphiti client."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )

        # Test connection
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # Import inside the function to avoid issues with missing modules
        from graphiti_core.search.search_config_recipes import (
            NODE_HYBRID_SEARCH_NODE_DISTANCE,
            NODE_HYBRID_SEARCH_RRF,
        )
        from graphiti_core.search.search_filters import SearchFilters

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
            logger.info(f'Using node distance-based search centered on node: {center_node_uuid}')
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            logger.info('Using standard reciprocal rank fusion search')

        search_config.limit = max_nodes

        # Set up filters if entity is specified
        filters = SearchFilters()
        if entity:
            filters.node_labels = [entity]
            logger.info(f'Filtering results to entity type: {entity}')

        # Perform the search
        logger.info(f"Searching for nodes with query: '{query}' in group_ids: {group_ids}")
        search_results = await graphiti_client.search_(
            query=query,
            config=search_config,
            group_ids=group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            logger.info('No nodes found matching the query.')
            print('No nodes found matching the query.')
            return

        # Format and display the results
        print(f"\nFound {len(search_results.nodes)} nodes matching query: '{query}'\n")

        for i, node in enumerate(search_results.nodes, 1):
            print(f'Node {i}: {node.name}')
            print(f'UUID: {node.uuid}')
            print(f'Group ID: {node.group_id}')

            if hasattr(node, 'labels') and node.labels:
                print(f'Labels: {", ".join(node.labels)}')

            if hasattr(node, 'summary') and node.summary:
                print(f'Summary: {node.summary}')

            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                # Format attributes more nicely
                for key, value in node.attributes.items():
                    if key not in ('uuid', 'name', 'group_id', 'created_at'):
                        print(f'  - {key}: {value}')

            # Add a separator between nodes
            print('-' * 50)

    except Exception as err:
        logger.error(f'An error occurred during node search: {err}', exc_info=True)
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


@cli_app.command()
def search_facts(
    query: str = QUERY_OPTION,
    group_id: str | None = GROUP_ID_OPTION,
    max_facts: int = MAX_FACTS_OPTION,
    center_node_uuid: str | None = CENTER_NODE_UUID_OPTION,
    workspace_path: str | None = WORKSPACE_PATH_OPTION,
):
    """
    Searches the knowledge graph for relevant facts (relationships) matching the query.
    Facts are represented as triplets in the form of "Subject - Relationship - Object".

    If a center node UUID is provided, results will be ranked by proximity to that node.
    """
    logger.info('Running search-facts command...')

    # Generate group_id if not provided
    effective_group_id = group_id
    if not effective_group_id:
        effective_group_id = generate_project_group_id(workspace_path)
        logger.info(f'Using generated group_id: {effective_group_id}')

    group_ids = [effective_group_id]  # Convert to list for the API

    asyncio.run(_search_facts(query, group_ids, max_facts, center_node_uuid))


async def _search_facts(
    query: str,
    group_ids: list[str],
    max_facts: int,
    center_node_uuid: str | None = None,
):
    """Internal async function to handle fact search using Graphiti client."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )

        # Test connection
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # Perform the search
        logger.info(f"Searching for facts with query: '{query}' in group_ids: {group_ids}")
        relevant_edges = await graphiti_client.search(
            group_ids=group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            logger.info('No facts found matching the query.')
            print('No facts found matching the query.')
            return

        # Format and display the results
        print(f"\nFound {len(relevant_edges)} facts matching query: '{query}'\n")

        for i, edge in enumerate(relevant_edges, 1):
            print(f'Fact {i}: {edge.fact}')
            print(f'UUID: {edge.uuid}')
            print(f'Group ID: {edge.group_id}')

            if hasattr(edge, 'source_node_uuid') and edge.source_node_uuid:
                print(f'Source Node UUID: {edge.source_node_uuid}')

            if hasattr(edge, 'target_node_uuid') and edge.target_node_uuid:
                print(f'Target Node UUID: {edge.target_node_uuid}')

            if hasattr(edge, 'name') and edge.name:
                print(f'Relationship Type: {edge.name}')

            if hasattr(edge, 'valid_at') and edge.valid_at:
                print(f'Valid At: {edge.valid_at.isoformat()}')

            if hasattr(edge, 'invalid_at') and edge.invalid_at:
                print(f'Invalid At: {edge.invalid_at.isoformat()}')
            else:
                print('Invalid At: Present (Currently Valid)')

            # Add a separator between facts
            print('-' * 50)

    except Exception as err:
        logger.error(f'An error occurred during fact search: {err}', exc_info=True)
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


@cli_app.command()
def get_entity_edge(
    uuid: str = UUID_EDGE_OPTION,
):
    """
    Retrieves detailed information about a specific entity edge (relationship) from the knowledge graph.
    The edge is identified by its UUID, which can be obtained from search results.
    """
    logger.info(f'Running get-entity-edge command for UUID: {uuid}')
    asyncio.run(_get_entity_edge(uuid))


async def _get_entity_edge(uuid: str):
    """Internal async function to get entity edge details using Graphiti client."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )

        # Test connection
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # Get the entity edge
        logger.info(f'Retrieving entity edge with UUID: {uuid}')
        edge = await graphiti_client.get_entity_edge(uuid)

        if not edge:
            logger.info(f'No entity edge found with UUID: {uuid}')
            print(f'No entity edge found with UUID: {uuid}')
            return

        # Format and display the edge details
        print('\nEntity Edge Details:\n')
        print(f'Fact: {edge.fact}')
        print(f'UUID: {edge.uuid}')
        print(f'Group ID: {edge.group_id}')
        print(f'Source Node UUID: {edge.source_node_uuid}')
        print(f'Target Node UUID: {edge.target_node_uuid}')
        print(f'Relationship Type: {edge.name}')
        print(f'Created At: {edge.created_at.isoformat()}')

        if hasattr(edge, 'valid_at') and edge.valid_at:
            print(f'Valid At: {edge.valid_at.isoformat()}')

        if hasattr(edge, 'invalid_at') and edge.invalid_at:
            print(f'Invalid At: {edge.invalid_at.isoformat()}')
        else:
            print('Invalid At: Present (Currently Valid)')

        if hasattr(edge, 'episodes') and edge.episodes:
            print(f'Associated Episodes: {", ".join(edge.episodes)}')

        # Get source and target nodes for additional context
        try:
            source_node = await graphiti_client.get_entity_node(edge.source_node_uuid)
            target_node = await graphiti_client.get_entity_node(edge.target_node_uuid)

            print('\nSource Node:')
            print(f'  Name: {source_node.name}')
            print(f'  UUID: {source_node.uuid}')

            print('\nTarget Node:')
            print(f'  Name: {target_node.name}')
            print(f'  UUID: {target_node.uuid}')
        except Exception as e:
            logger.warning(f'Could not retrieve associated nodes: {e}')

    except Exception as err:
        logger.error(f'An error occurred while retrieving entity edge: {err}', exc_info=True)
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


@cli_app.command()
def get_episodes(
    group_id: str | None = GROUP_ID_OPTION,
    last_n: int = LAST_N_OPTION,
    workspace_path: str | None = WORKSPACE_PATH_OPTION,
):
    """
    Retrieves the most recent episodes for a specific group from the knowledge graph.
    Episodes represent the raw data added to the graph, before it's processed into entities and relationships.
    """
    logger.info('Running get-episodes command...')

    # Generate group_id if not provided
    effective_group_id = group_id
    if not effective_group_id:
        effective_group_id = generate_project_group_id(workspace_path)
        logger.info(f'Using generated group_id: {effective_group_id}')

    asyncio.run(_get_episodes(effective_group_id, last_n))


async def _get_episodes(group_id: str, last_n: int):
    """Internal async function to get recent episodes using Graphiti client."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )

        # Test connection
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # First, check for available labels to find the correct episode node label
        logger.info('Identifying available node labels in the database...')
        available_labels = []
        async with graphiti_client.driver.session() as session:
            label_query = 'CALL db.labels()'
            label_result = await session.run(label_query)
            available_labels = [record['label'] async for record in label_result]

        logger.info(f'Available labels: {", ".join(available_labels)}')

        # Look for episode-related labels (Episodic is the most likely based on GraphitiCore)
        episode_label = None
        for label in available_labels:
            if label.lower() in ['episode', 'episodic']:
                episode_label = label
                logger.info(f'Found episode label: {episode_label}')
                break

        if not episode_label:
            logger.warning('Could not identify the episode label in the database')
            print('No episode nodes found in the database. Make sure data has been ingested.')
            return

        # Get the episodes directly using Neo4j with the correct label
        logger.info(f'Retrieving the {last_n} most recent episodes for group ID: {group_id}')

        episodes = []
        async with graphiti_client.driver.session() as session:
            # Check what properties are available on the episode nodes
            property_query = f"""
            MATCH (e:{episode_label})
            WHERE e.group_id = $group_id
            RETURN e LIMIT 1
            """
            prop_result = await session.run(property_query, group_id=group_id)
            sample_episode = None
            sample_record = [record async for record in prop_result]
            if sample_record and len(sample_record) > 0:
                sample_episode = sample_record[0]['e']
                logger.info(f'Sample episode properties: {list(sample_episode.keys())}')

            # Build a dynamic query based on available properties
            select_fields = ['e.uuid as uuid', 'e.name as name', 'e.group_id as group_id']
            optional_fields = [
                'e.source as source',
                'e.source_description as source_description',
                'e.created_at as created_at',
                'e.episode_body as episode_body',
                'e.body as body',  # Alternative name
                'e.content as content',  # Alternative name detected in the sample
            ]

            # Only include fields that exist in the database
            if sample_episode:
                properties = sample_episode.keys()
                select_fields = ['e.uuid as uuid', 'e.group_id as group_id']
                if 'name' in properties:
                    select_fields.append('e.name as name')
                for field in optional_fields:
                    field_name = field.split(' as ')[0].replace('e.', '')
                    if field_name in properties:
                        select_fields.append(field)

            # Construct and execute the query
            query = f"""
            MATCH (e:{episode_label})
            WHERE e.group_id = $group_id
            RETURN {', '.join(select_fields)}
            ORDER BY e.created_at DESC
            LIMIT $last_n
            """
            logger.info(f'Using query: {query}')
            result = await session.run(query, group_id=group_id, last_n=last_n)
            episodes = [record async for record in result]

        if not episodes:
            logger.info(f'No episodes found for group ID: {group_id}')
            print(f'No episodes found for group ID: {group_id}')
            return

        # Format and display the episodes
        print(f'\nFound {len(episodes)} episodes for group ID: {group_id}\n')

        for i, episode in enumerate(episodes, 1):
            # Convert Neo4j Record to dictionary for easier access
            episode_dict = dict(episode)

            print(f'Episode {i}:')

            # Display UUID (always available)
            print(f'UUID: {episode_dict["uuid"]}')

            # Display name if available
            if 'name' in episode_dict and episode_dict['name']:
                print(f'Name: {episode_dict["name"]}')

            # Display source if available
            if 'source' in episode_dict and episode_dict['source']:
                print(f'Source: {episode_dict["source"]}')

            # Display source description if available
            if 'source_description' in episode_dict and episode_dict['source_description']:
                print(f'Source Description: {episode_dict["source_description"]}')

            # Display created_at if available
            if 'created_at' in episode_dict and episode_dict['created_at']:
                created_at = episode_dict['created_at']
                # Handle Neo4j DateTime objects
                if hasattr(created_at, 'to_native'):
                    created_at = created_at.to_native().isoformat()
                elif isinstance(created_at, int | float):
                    from datetime import datetime

                    created_at = datetime.fromtimestamp(created_at / 1000.0).isoformat()
                print(f'Created At: {created_at}')

            # Display content preview if available
            if 'content' in episode_dict and episode_dict['content']:
                content = episode_dict['content']
                print(f'Content: {content}')

            # Add a separator between episodes
            print('-' * 50)

    except Exception as err:
        logger.error(f'An error occurred while retrieving episodes: {err}', exc_info=True)
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


@cli_app.command()
def delete_entity_edge(
    uuid: str = UUID_EDGE_OPTION,
    confirm: bool = CONFIRM_OPTION,
    skip_preview: bool = SKIP_PREVIEW_OPTION,
):
    """
    Deletes an entity edge (relationship) from the knowledge graph.

    CAUTION: This operation cannot be undone. The edge will be permanently removed.

    For safety, this command:
    1. First shows a preview of the edge to be deleted (unless --skip-preview is used)
    2. Requires explicit confirmation with the --confirm flag
    3. Asks for a final confirmation prompt
    """
    logger.info(f'Running delete-entity-edge command for UUID: {uuid}')

    # Check for confirmation flag
    if not confirm:
        logger.warning('Deletion aborted: --confirm flag is required for safety')
        print('\n⚠️  WARNING: To delete an entity edge, you must use the --confirm flag')
        print('    Example: delete-entity-edge --uuid <uuid> --confirm\n')
        raise typer.Exit(code=1)

    # Run the deletion process
    asyncio.run(_delete_entity_edge(uuid, skip_preview))


async def _delete_entity_edge(uuid: str, skip_preview: bool = False):
    """Internal async function to handle entity edge deletion using Graphiti client."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )

        # Test connection
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # First get the entity edge to show the user what will be deleted if not skipping preview
        if not skip_preview:
            try:
                entity_edge = await graphiti_client.get_entity_edge(uuid)
                if entity_edge:
                    logger.info(f'Retrieved entity edge {uuid}: {entity_edge}')
                    print(f'\nPreparing to delete entity edge: {uuid}')
                    print(f'  - Type: {entity_edge.get("type", "Unknown")}')
                    print(f'  - From: {entity_edge.get("from_node", {}).get("uuid", "N/A")}')
                    print(f'  - To: {entity_edge.get("to_node", {}).get("uuid", "N/A")}')
                else:
                    logger.warning(f'Entity edge with UUID {uuid} not found!')
                    print(f'\n⚠️ Entity edge {uuid} not found!')
                    return
            except Exception as e:
                logger.warning(f'Could not retrieve entity edge: {e}')

        # Now delete the entity edge
        logger.info(f'Deleting entity edge with UUID: {uuid}')
        result = await graphiti_client.delete_entity_edge(uuid)
        if result:
            logger.info(f'Successfully deleted entity edge: {uuid}')
            print(f'\n✓ Successfully deleted entity edge: {uuid}')
        else:
            logger.warning(f'Could not delete entity edge {uuid} - not found or already deleted')
            print(f'\n⚠️ Could not delete entity edge {uuid} - not found or already deleted')

    except Exception as err:
        logger.error(f'An error occurred while deleting entity edge: {err}', exc_info=True)
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


@cli_app.command()
def delete_episode(
    uuid: str = UUID_EPISODE_OPTION,
    confirm: bool = CONFIRM_OPTION,
    skip_preview: bool = SKIP_PREVIEW_OPTION,
):
    """
    Deletes an episode from the knowledge graph.

    CAUTION: This operation cannot be undone. The episode and potentially related entities
    and relationships will be permanently removed.

    For safety, this command:
    1. First shows a preview of the episode to be deleted (unless --skip-preview is used)
    2. Requires explicit confirmation with the --confirm flag
    3. Asks for a final confirmation prompt
    """
    logger.info(f'Running delete-episode command for UUID: {uuid}')

    # Check for confirmation flag
    if not confirm:
        logger.warning('Deletion aborted: --confirm flag is required for safety')
        print('\n⚠️  WARNING: To delete an episode, you must use the --confirm flag')
        print('    Example: delete-episode --uuid <uuid> --confirm\n')
        raise typer.Exit(code=1)

    # Run the deletion process
    asyncio.run(_delete_episode(uuid, skip_preview))


async def _delete_episode(uuid: str, skip_preview: bool = False):
    """Internal async function to handle episode deletion with safeguards."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )

        # Test connection
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # Preview the episode to be deleted
        if not skip_preview:
            logger.info(f'Retrieving episode with UUID: {uuid} for preview')

            # NOTE: We don't have a direct get_episode method in Graphiti,
            # so we'll use Neo4j directly to get episode details
            try:
                episodes_with_uuid = []
                async with graphiti_client.driver.session() as session:
                    result = await session.run(
                        """
                        MATCH (e:Episode {uuid: $uuid})
                        RETURN e.name as name, e.uuid as uuid, e.source as source, 
                               e.source_description as source_description, e.created_at as created_at,
                               e.group_id as group_id
                        """,
                        uuid=uuid,
                    )
                    episodes_with_uuid = [record async for record in result]

                if not episodes_with_uuid:
                    logger.warning(f'No episode found with UUID: {uuid}')
                    print(f'No episode found with UUID: {uuid}')
                    return

                # Show preview of what will be deleted
                episode = episodes_with_uuid[0]
                print('\n🔍 PREVIEW OF EPISODE TO BE DELETED:\n')
                print(f'Name: {episode["name"]}')
                print(f'UUID: {episode["uuid"]}')
                print(f'Source: {episode["source"]}')
                print(f'Group ID: {episode["group_id"]}')

                if episode['source_description']:
                    print(f'Source Description: {episode["source_description"]}')

                if episode['created_at']:
                    created_at = episode['created_at']
                    if isinstance(created_at, int | float):
                        from datetime import datetime

                        created_at = datetime.fromtimestamp(created_at / 1000.0).isoformat()
                    print(f'Created At: {created_at}')

                # Count associated entities and relationships
                try:
                    result = await session.run(
                        """
                        MATCH (e:Episode {uuid: $uuid})
                        OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity)
                        WITH e, count(DISTINCT n) as entityCount
                        OPTIONAL MATCH (e)-[:MENTIONS]->(rel:EntityEdge)
                        RETURN entityCount, count(DISTINCT rel) as relationshipCount
                        """,
                        uuid=uuid,
                    )
                    stats = [record async for record in result]

                    if stats and len(stats) > 0:
                        entity_count = stats[0]['entityCount']
                        relationship_count = stats[0]['relationshipCount']
                        print('\nThis episode is connected to:')
                        print(f'  - {entity_count} entities')
                        print(f'  - {relationship_count} relationships')

                        if entity_count > 0 or relationship_count > 0:
                            print(
                                '\n⚠️  WARNING: Deleting this episode may affect related entities and relationships!'
                            )
                except Exception as e:
                    logger.warning(f'Could not count associated items: {e}')

                print('\n⚠️  WARNING: This operation cannot be undone! ⚠️\n')
            except Exception as e:
                logger.warning(f'Could not preview episode: {e}')
                print('\n⚠️  Could not preview the episode to be deleted. Proceed with caution.')

        # Final confirmation prompt
        confirmation = (
            input('\nAre you sure you want to delete this episode? (y/N): ').strip().lower()
        )
        if confirmation != 'y':
            logger.info('Deletion cancelled by user')
            print('Deletion cancelled.')
            return

        # Delete the episode
        logger.info(f'Deleting episode with UUID: {uuid}')
        await graphiti_client.delete_episode(uuid)
        logger.info(f'Successfully deleted episode with UUID: {uuid}')
        print(f'\n✓ Successfully deleted episode: {uuid}')

    except Exception as err:
        logger.error(f'An error occurred while deleting episode: {err}', exc_info=True)
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


@cli_app.command()
def clear_graph(
    confirm: bool = CONFIRM_FLAG_OPTION,
    force: bool = FORCE_OPTION,
):
    """
    Clears all data from the knowledge graph and rebuilds indices.

    ⚠️ EXTREME CAUTION: This operation will permanently delete ALL data in the graph.
    This includes ALL episodes, entities, relationships, and other data.

    For maximum safety, this command requires:
    1. The --confirm flag
    2. The --force flag
    3. Typing a special confirmation code
    4. Waiting through a 5-second countdown
    """
    logger.info('Running clear-graph command...')

    # Check for both confirmation flags
    if not confirm or not force:
        logger.warning('Operation aborted: Both --confirm and --force flags are required')
        print('\n⚠️  CRITICAL WARNING: To clear the entire graph, you must use BOTH flags:')
        print('    clear-graph --confirm --force\n')
        print('This requirement exists to prevent catastrophic data loss.')
        raise typer.Exit(code=1)

    # Run the graph clearing process
    asyncio.run(_clear_graph())


async def _clear_graph():
    """Internal async function to handle graph clearing with extensive safeguards."""
    if not NEO4J_USER or not NEO4J_PASSWORD:
        logger.error('NEO4J_USER and NEO4J_PASSWORD must be set in environment or .env file')
        raise typer.Exit(code=1)

    llm_client: LLMClient | None = None
    if OPENAI_API_KEY:
        logger.info(f'Using OpenAI model: {MODEL_NAME}')
        llm_config = LLMConfig(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        if OPENAI_BASE_URL:
            llm_config.base_url = OPENAI_BASE_URL
        llm_client = OpenAIClient(config=llm_config)
    else:
        logger.warning(
            'OPENAI_API_KEY not found. LLM features like embedding/search might be limited.'
        )

    graphiti_client: Graphiti | None = None
    try:
        logger.info(f'Connecting to Neo4j at {NEO4J_URI} as user {NEO4J_USER}')
        graphiti_client = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=llm_client,
        )

        # Test connection
        await graphiti_client.driver.verify_connectivity()
        logger.info('Neo4j connection successful.')

        # Count items in the graph
        async with graphiti_client.driver.session() as session:
            episode_result = await session.run('MATCH (e:Episode) RETURN count(e) as count')
            episode_records = [record async for record in episode_result]
            episode_count = episode_records[0]['count'] if episode_records else 0

            entity_result = await session.run('MATCH (n:Entity) RETURN count(n) as count')
            entity_records = [record async for record in entity_result]
            entity_count = entity_records[0]['count'] if entity_records else 0

            edge_result = await session.run('MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count')
            edge_records = [record async for record in edge_result]
            edge_count = edge_records[0]['count'] if edge_records else 0

        # Display critical warning
        print('\n' + '!' * 80)
        print('⚠️  CRITICAL WARNING: YOU ARE ABOUT TO DELETE ALL DATA IN THE KNOWLEDGE GRAPH ⚠️')
        print('!' * 80)
        print('\nThis will permanently delete:')
        print(f' - {episode_count} episodes')
        print(f' - {entity_count} entities')
        print(f' - {edge_count} relationships')
        print('\nTHIS OPERATION CANNOT BE UNDONE!')
        print('\n' + '!' * 80 + '\n')

        # Require special confirmation code
        confirmation_code = 'CONFIRM-CLEAR-ALL'
        user_code = input(f"To proceed, type '{confirmation_code}' exactly: ").strip()

        if user_code != confirmation_code:
            logger.info('Operation cancelled: incorrect confirmation code')
            print('Operation cancelled: incorrect confirmation code')
            return

        # Final countdown
        print('\nFinal countdown before clearing graph:')
        for i in range(5, 0, -1):
            print(f'{i}... ', end='', flush=True)
            await asyncio.sleep(1)
        print('0!')

        # Clear the graph
        logger.info('Clearing all data from the graph')
        print('\nClearing graph...')
        await graphiti_client.clear_graph()
        logger.info('Successfully cleared all data from the graph')
        print('✓ Successfully cleared all data from the graph and rebuilt indices.')

    except Exception as err:
        logger.error(f'An error occurred while clearing the graph: {err}', exc_info=True)
        raise typer.Exit(code=1) from err
    finally:
        if graphiti_client:
            await graphiti_client.close()
            logger.info('Neo4j connection closed.')


if __name__ == '__main__':
    cli_app()
