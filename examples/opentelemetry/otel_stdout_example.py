"""
Copyright 2025, Zep Software, Inc.

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

import asyncio
import json
import logging
from datetime import datetime, timezone
from logging import INFO

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.nodes import EpisodeType

logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def setup_otel_stdout_tracing():
    """Configure OpenTelemetry to export traces to stdout."""
    resource = Resource(attributes={'service.name': 'graphiti-example'})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


async def main():
    otel_tracer = setup_otel_stdout_tracing()

    print('OpenTelemetry stdout tracing enabled\n')

    kuzu_driver = KuzuDriver()
    graphiti = Graphiti(
        graph_driver=kuzu_driver, tracer=otel_tracer, trace_span_prefix='graphiti.example'
    )

    try:
        await graphiti.build_indices_and_constraints()
        print('Graph indices and constraints built\n')

        episodes = [
            {
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'biographical information',
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'term dates',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                },
                'type': EpisodeType.json,
                'description': 'structured data',
            },
        ]

        print('Adding episodes...\n')
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Episode {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Episode {i} ({episode["type"].value})')

        print("\nSearching for: 'Who was the California Attorney General?'\n")
        results = await graphiti.search('Who was the California Attorney General?')

        print('Search Results:')
        for idx, result in enumerate(results[:3]):
            print(f'\nResult {idx + 1}:')
            print(f'  Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'  Valid from: {result.valid_at}')

        print("\nSearching for: 'What positions has Gavin Newsom held?'\n")
        results = await graphiti.search('What positions has Gavin Newsom held?')

        print('Search Results:')
        for idx, result in enumerate(results[:3]):
            print(f'\nResult {idx + 1}:')
            print(f'  Fact: {result.fact}')

        print('\nExample complete')

    finally:
        await graphiti.close()


if __name__ == '__main__':
    asyncio.run(main())
