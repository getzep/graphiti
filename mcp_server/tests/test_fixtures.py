"""
Shared test fixtures and utilities for Graphiti MCP integration tests.
"""

import asyncio
import contextlib
import json
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest
from faker import Faker
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

fake = Faker()


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

    env = {
        'DATABASE_PROVIDER': database,
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'test_key' if use_mock_llm else None),
    }

    # Database-specific configuration
    if database == 'neo4j':
        env.update(
            {
                'NEO4J_URI': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                'NEO4J_USER': os.environ.get('NEO4J_USER', 'neo4j'),
                'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', 'graphiti'),
            }
        )
    elif database == 'falkordb':
        env['FALKORDB_URI'] = os.environ.get('FALKORDB_URI', 'redis://localhost:6379')

    # Apply config overrides
    if config_overrides:
        env.update(config_overrides)

    # Add mock LLM flag if needed
    if use_mock_llm:
        env['USE_MOCK_LLM'] = 'true'

    server_params = StdioServerParameters(
        command='uv', args=['run', 'main.py', '--transport', 'stdio'], env=env
    )

    async with stdio_client(server_params) as (read, write):
        session = ClientSession(read, write)
        await session.initialize()

        try:
            yield session, test_group_id
        finally:
            # Cleanup: Clear test data
            with contextlib.suppress(Exception):
                await session.call_tool('clear_graph', {'group_id': test_group_id})

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
