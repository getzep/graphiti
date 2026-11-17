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

# Ingest the episodes of a JSONL file with serial or concurrent ingestion.
#
# Each line of the file is one JSON object:
#
#     {"name": "...", "content": "...", "reference_time": "2025-01-31T09:00:00Z",
#      "source_description": "..."}
#
# `name` and `source_description` are optional. The script ingests the lines in file
# order into a new graph on embedded FalkorDB, and prints the time and the graph
# counts as JSON.
#
#     python ingest_jsonl.py episodes.jsonl --window 0      # serial ingestion
#     python ingest_jsonl.py episodes.jsonl --window 4      # concurrent ingestion
#
# The script needs Python 3.12 or later, `graphiti-core[falkordblite,tracing]`, and
# the environment variable `OPENAI_API_KEY`.

import argparse
import asyncio
import json
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from redislite.async_falkordb_client import AsyncFalkorDB

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import GroupsEdgesNotFoundError, GroupsNodesNotFoundError
from graphiti_core.nodes import EntityNode, EpisodeType

GROUP_ID = 'concurrent_ingestion_example'

COMMIT_COUNTERS = (
    'repair.c3.node_merged',
    'repair.c4a.edge_removed',
    'repair.c5.edge_merged',
    'repair.c6.edge_ended',
    'repair.llm_calls',
    'operation.dropped',
    'apply.attempts',
)


def read_episodes(path: Path, limit: int | None) -> list[dict]:
    with path.open() as file:
        episodes = [json.loads(line) for line in file if line.strip()]
    return episodes[:limit] if limit else episodes


def make_graphiti(database_path: str, window: int, exporter: InMemorySpanExporter) -> Graphiti:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    concurrent_ingestion = None
    if window > 0:
        from graphiti_core.ingestion import ConcurrentIngestionConfig

        concurrent_ingestion = ConcurrentIngestionConfig(window=window)
    driver = FalkorDriver(falkor_db=AsyncFalkorDB(dbfilename=database_path), database=GROUP_ID)
    return Graphiti(
        graph_driver=driver,
        tracer=provider.get_tracer('concurrent_ingestion_example'),
        concurrent_ingestion=concurrent_ingestion,
    )


async def add_episode(graphiti: Graphiti, index: int, episode: dict):
    return await graphiti.add_episode(
        name=episode.get('name') or f'episode-{index}',
        episode_body=episode['content'],
        source_description=episode.get('source_description', 'JSONL file'),
        reference_time=datetime.fromisoformat(episode['reference_time'].replace('Z', '+00:00')),
        source=EpisodeType.text,
        group_id=GROUP_ID,
    )


async def graph_counts(graphiti: Graphiti) -> dict:
    try:
        nodes = await EntityNode.get_by_group_ids(graphiti.driver, [GROUP_ID])
    except GroupsNodesNotFoundError:
        nodes = []
    try:
        edges = await EntityEdge.get_by_group_ids(graphiti.driver, [GROUP_ID])
    except GroupsEdgesNotFoundError:
        edges = []
    names = Counter(' '.join(node.name.lower().split()) for node in nodes)
    return {
        'entity_node_count': len(nodes),
        'entity_edge_count': len(edges),
        'nodes_with_a_shared_name': sum(count for count in names.values() if count > 1),
        'ended_edge_count': sum(
            1 for edge in edges if edge.invalid_at is not None or edge.expired_at is not None
        ),
    }


def span_report(exporter: InMemorySpanExporter, episode_count: int) -> dict:
    """Read the sequence numbers and the repair counts from the spans of the run."""
    spans = exporter.get_finished_spans()
    sequence = sorted(
        span.attributes['ingest.seq']
        for span in spans
        if span.name == 'graphiti.add_episode' and 'ingest.seq' in (span.attributes or {})
    )
    counters: Counter = Counter()
    for span in spans:
        if span.name == 'graphiti.ingest.commit':
            for key in COMMIT_COUNTERS:
                counters[key] += (span.attributes or {}).get(key, 0)
    return {
        'sequence_is_complete': sequence == list(range(1, episode_count + 1)),
        'commit_counters': {key: counters[key] for key in COMMIT_COUNTERS},
    }


async def run(path: Path, window: int, limit: int | None) -> dict:
    episodes = read_episodes(path, limit)
    exporter = InMemorySpanExporter()
    with tempfile.TemporaryDirectory() as directory:
        graphiti = make_graphiti(str(Path(directory) / 'graph.db'), window, exporter)
        try:
            await graphiti.build_indices_and_constraints()
            started = time.monotonic()
            if window > 0:
                # The calls start in file order, and the admission gives them their
                # sequence numbers in that order. Each call returns after its commit
                # and its enrichment, so the time includes all the work of the run.
                tasks = [
                    asyncio.create_task(add_episode(graphiti, index, episode))
                    for index, episode in enumerate(episodes)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                results = []
                for index, episode in enumerate(episodes):
                    try:
                        results.append(await add_episode(graphiti, index, episode))
                    except Exception as error:
                        results.append(error)
            seconds = time.monotonic() - started
            failures = [repr(result) for result in results if isinstance(result, BaseException)]
            report = {
                'mode': 'concurrent' if window > 0 else 'serial',
                'window': window,
                'episode_count': len(episodes),
                'failed_episode_count': len(failures),
                'failures': failures[:10],
                'seconds': round(seconds, 2),
                'episodes_per_minute': round(60 * len(episodes) / seconds, 2) if seconds else None,
                **await graph_counts(graphiti),
            }
            if window > 0:
                report.update(span_report(exporter, len(episodes)))
            return report
        finally:
            await graphiti.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Ingest the episodes of a JSONL file with serial or concurrent ingestion.'
    )
    parser.add_argument('path', type=Path, help='The JSONL file with one episode on each line.')
    parser.add_argument(
        '--window',
        type=int,
        default=4,
        help='The window of concurrent ingestion. 0 selects serial ingestion.',
    )
    parser.add_argument('--limit', type=int, default=None, help='Ingest only the first N lines.')
    arguments = parser.parse_args()
    print(json.dumps(asyncio.run(run(arguments.path, arguments.window, arguments.limit)), indent=2))


if __name__ == '__main__':
    main()
