"""
Shared test fixtures and utilities for Graphiti MCP integration tests.
"""

import asyncio
import contextlib
import json
import os
import random
import shlex
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from dotenv import dotenv_values
from faker import Faker
from http_mcp_test_client import RawHttpMCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

fake = Faker()
mcp_server_dir = Path(__file__).resolve().parents[1]
nas_env_path = mcp_server_dir / '.env.nas'
nas_env = {k: v for k, v in dotenv_values(nas_env_path).items() if isinstance(v, str)} if nas_env_path.exists() else {}


class TestDataGenerator:
    """Generate realistic test data for various scenarios."""

    @staticmethod
    def generate_company_profile() -> dict[str, Any]:
        """Generate a realistic company profile."""
        return {
            'company': {
                'name': fake.company(),
                'founded': random.randint(1990, 2023),
                'industry': random.choice(['Tech', 'Finance', 'Healthcare', 'Retail']),
                'employees': random.randint(10, 10000),
                'revenue': f'${random.randint(1, 1000)}M',
                'headquarters': fake.city(),
            },
            'products': [
                {
                    'id': fake.uuid4()[:8],
                    'name': fake.catch_phrase(),
                    'category': random.choice(['Software', 'Hardware', 'Service']),
                    'price': random.randint(10, 10000),
                }
                for _ in range(random.randint(1, 5))
            ],
            'leadership': {
                'ceo': fake.name(),
                'cto': fake.name(),
                'cfo': fake.name(),
            },
        }

    @staticmethod
    def generate_conversation(turns: int = 3) -> str:
        """Generate a realistic conversation."""
        topics = [
            'product features',
            'pricing',
            'technical support',
            'integration',
            'documentation',
            'performance',
        ]

        conversation = []
        for _ in range(turns):
            topic = random.choice(topics)
            user_msg = f'user: {fake.sentence()} about {topic}?'
            assistant_msg = f'assistant: {fake.paragraph(nb_sentences=2)}'
            conversation.extend([user_msg, assistant_msg])

        return '\n'.join(conversation)

    @staticmethod
    def generate_technical_document() -> str:
        """Generate technical documentation content."""
        sections = [
            f'# {fake.catch_phrase()}\n\n{fake.paragraph()}',
            f'## Architecture\n{fake.paragraph()}',
            f'## Implementation\n{fake.paragraph()}',
            f'## Performance\n- Latency: {random.randint(1, 100)}ms\n- Throughput: {random.randint(100, 10000)} req/s',
            f'## Dependencies\n- {fake.word()}\n- {fake.word()}\n- {fake.word()}',
        ]
        return '\n\n'.join(sections)

    @staticmethod
    def generate_news_article() -> str:
        """Generate a news article."""
        company = fake.company()
        return f"""
        {company} Announces {fake.catch_phrase()}

        {fake.city()}, {fake.date()} - {company} today announced {fake.paragraph()}.

        "This is a significant milestone," said {fake.name()}, CEO of {company}.
        "{fake.sentence()}"

        The announcement comes after {fake.paragraph()}.

        Industry analysts predict {fake.paragraph()}.
        """

    @staticmethod
    def generate_user_profile() -> dict[str, Any]:
        """Generate a user profile."""
        return {
            'user_id': fake.uuid4(),
            'name': fake.name(),
            'email': fake.email(),
            'joined': fake.date_time_this_year().isoformat(),
            'preferences': {
                'theme': random.choice(['light', 'dark', 'auto']),
                'notifications': random.choice([True, False]),
                'language': random.choice(['en', 'es', 'fr', 'de']),
            },
            'activity': {
                'last_login': fake.date_time_this_month().isoformat(),
                'total_sessions': random.randint(1, 1000),
                'average_duration': f'{random.randint(1, 60)} minutes',
            },
        }


class MockLLMProvider:
    """Mock LLM provider for testing without actual API calls."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay  # Simulate LLM latency

    async def generate(self, prompt: str) -> str:
        """Simulate LLM generation with delay."""
        await asyncio.sleep(self.delay)

        # Return deterministic responses based on prompt patterns
        if 'extract entities' in prompt.lower():
            return json.dumps(
                {
                    'entities': [
                        {'name': 'TestEntity1', 'type': 'PERSON'},
                        {'name': 'TestEntity2', 'type': 'ORGANIZATION'},
                    ]
                }
            )
        elif 'summarize' in prompt.lower():
            return 'This is a test summary of the provided content.'
        else:
            return 'Mock LLM response'


class HttpSessionAdapter:
    def __init__(self, client: RawHttpMCPClient):
        self._client = client

    async def list_tools(self):
        payload = await self._client.list_tools()
        tools = payload.get('result', {}).get('tools', [])
        return SimpleNamespace(
            tools=[SimpleNamespace(name=tool['name']) for tool in tools if isinstance(tool, dict)]
        )

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]):
        payload = await self._client.call_tool(tool_name, arguments)
        result = payload.get('result', {})
        content = result.get('content') or []
        text = ''
        if content and isinstance(content[0], dict):
            text = content[0].get('text', '')
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    async def close(self):
        return None


@asynccontextmanager
async def graphiti_test_client(
    group_id: str | None = None,
    database: str = 'falkordb',
    use_mock_llm: bool = False,
    config_overrides: dict[str, Any] | None = None,
):
    """
    Context manager for creating test clients with various configurations.

    Args:
        group_id: Test group identifier
        database: Database backend (neo4j, falkordb)
        use_mock_llm: Whether to use mock LLM for faster tests
        config_overrides: Additional config overrides
    """
    test_group_id = group_id or f'test_{int(time.time())}_{random.randint(1000, 9999)}'
    effective_database = database
    if (
        database == 'falkordb'
        and 'FALKORDB_URI' not in os.environ
        and nas_env.get('NEO4J_URI')
    ):
        effective_database = 'neo4j'

    env = {**os.environ, **nas_env}
    env['DATABASE_PROVIDER'] = effective_database
    env.setdefault('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', 'test_key'))
    env.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
    env.setdefault('UV_CACHE_DIR', '/tmp/graphiti-uv-cache')
    if effective_database == 'neo4j' and nas_env:
        env['CONFIG_PATH'] = str(mcp_server_dir / 'config' / 'config-docker-neo4j-external.yaml')

    # Database-specific configuration
    if effective_database == 'neo4j':
        env.update(
            {
                'NEO4J_URI': os.environ.get('NEO4J_URI', env.get('NEO4J_URI', 'bolt://localhost:7687')),
                'NEO4J_USER': os.environ.get('NEO4J_USER', env.get('NEO4J_USER', 'neo4j')),
                'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', env.get('NEO4J_PASSWORD', 'graphiti')),
            }
        )
    elif effective_database == 'falkordb':
        env['FALKORDB_URI'] = os.environ.get('FALKORDB_URI', env.get('FALKORDB_URI', 'redis://localhost:6379'))

    # Apply config overrides
    if config_overrides:
        env.update(config_overrides)

    # Add mock LLM flag if needed
    if use_mock_llm:
        env['USE_MOCK_LLM'] = 'true'

    main_py = mcp_server_dir / 'main.py'
    command = (
        f'cd {shlex.quote(str(mcp_server_dir))} && '
        f'uv run {shlex.quote(str(main_py))} --transport stdio'
    )
    if env.get('CONFIG_PATH'):
        command += f" --config {shlex.quote(env['CONFIG_PATH'])}"
    use_http_transport = effective_database == 'neo4j' and bool(nas_env)
    if use_http_transport:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            http_port = sock.getsockname()[1]

        http_command = command.replace('--transport stdio', f'--transport http --port {http_port}')
        process = await asyncio.create_subprocess_exec(
            'bash',
            '-c',
            http_command,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        base_url = f'http://127.0.0.1:{http_port}'
        async with httpx.AsyncClient(timeout=2.0) as health_client:
            ready = False
            for _ in range(60):
                with contextlib.suppress(Exception):
                    response = await health_client.get(f'{base_url}/health')
                    if response.status_code == 200:
                        ready = True
                        break
                await asyncio.sleep(1)

        if not ready:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
            await process.wait()
            raise RuntimeError(
                f'HTTP test server did not become ready on {base_url} (returncode={process.returncode})'
            )

        async with RawHttpMCPClient(base_url) as client:
            await client.initialize()
            session = HttpSessionAdapter(client)
            try:
                yield session, test_group_id
            finally:
                with contextlib.suppress(Exception):
                    await session.call_tool('clear_graph', {'group_ids': [test_group_id]})
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=5)
        return

    server_params = StdioServerParameters(
        command='bash',
        args=[
            '-c',
            command,
        ],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        session = ClientSession(read, write)
        await session.initialize()

        try:
            yield session, test_group_id
        finally:
            # Cleanup: Clear test data
            with contextlib.suppress(Exception):
                await session.call_tool('clear_graph', {'group_ids': [test_group_id]})

            await session.close()


class PerformanceBenchmark:
    """Track and analyze performance benchmarks."""

    def __init__(self):
        self.measurements: dict[str, list[float]] = {}

    def record(self, operation: str, duration: float):
        """Record a performance measurement."""
        if operation not in self.measurements:
            self.measurements[operation] = []
        self.measurements[operation].append(duration)

    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.measurements or not self.measurements[operation]:
            return {}

        durations = self.measurements[operation]
        return {
            'count': len(durations),
            'mean': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'median': sorted(durations)[len(durations) // 2],
        }

    def report(self) -> str:
        """Generate a performance report."""
        lines = ['Performance Benchmark Report', '=' * 40]

        for operation in sorted(self.measurements.keys()):
            stats = self.get_stats(operation)
            lines.append(f'\n{operation}:')
            lines.append(f'  Samples: {stats["count"]}')
            lines.append(f'  Mean: {stats["mean"]:.3f}s')
            lines.append(f'  Median: {stats["median"]:.3f}s')
            lines.append(f'  Min: {stats["min"]:.3f}s')
            lines.append(f'  Max: {stats["max"]:.3f}s')

        return '\n'.join(lines)


# Pytest fixtures
@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


@pytest.fixture
def performance_benchmark():
    """Provide performance benchmark tracker."""
    return PerformanceBenchmark()


@pytest.fixture
async def mock_graphiti_client():
    """Provide a Graphiti client with mocked LLM."""
    async with graphiti_test_client(use_mock_llm=True) as (session, group_id):
        yield session, group_id


@pytest.fixture
async def graphiti_client():
    """Provide a real Graphiti client."""
    async with graphiti_test_client(use_mock_llm=False) as (session, group_id):
        yield session, group_id


# Test data fixtures
@pytest.fixture
def sample_memories():
    """Provide sample memory data for testing."""
    return [
        {
            'name': 'Company Overview',
            'episode_body': TestDataGenerator.generate_company_profile(),
            'source': 'json',
            'source_description': 'company database',
        },
        {
            'name': 'Product Launch',
            'episode_body': TestDataGenerator.generate_news_article(),
            'source': 'text',
            'source_description': 'press release',
        },
        {
            'name': 'Customer Support',
            'episode_body': TestDataGenerator.generate_conversation(),
            'source': 'message',
            'source_description': 'support chat',
        },
        {
            'name': 'Technical Specs',
            'episode_body': TestDataGenerator.generate_technical_document(),
            'source': 'text',
            'source_description': 'documentation',
        },
    ]


@pytest.fixture
def large_dataset():
    """Generate a large dataset for stress testing."""
    return [
        {
            'name': f'Document {i}',
            'episode_body': TestDataGenerator.generate_technical_document(),
            'source': 'text',
            'source_description': 'bulk import',
        }
        for i in range(50)
    ]
