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

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

import graphiti_core.graphiti as graphiti_module
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode

GROUP = 'bulk-validation'


class Person(BaseModel):
    """A person."""


def make_graphiti() -> Graphiti:
    driver = MagicMock()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = MagicMock(driver=driver)
    graphiti.tracer = MagicMock()
    graphiti.max_coroutines = None
    graphiti.concurrent_ingestion = None
    graphiti._resolve_request_scope = lambda group_id: (
        group_id or GROUP,
        driver,
        graphiti.clients,
    )
    return graphiti


def raw_episode() -> RawEpisode:
    return RawEpisode(
        name='episode',
        content='Ada met Charles.',
        source_description='test',
        source=EpisodeType.text,
        reference_time=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_add_episode_bulk_rejects_unknown_excluded_entity_type(monkeypatch):
    graph_write = AsyncMock()
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', graph_write)
    graphiti = make_graphiti()

    with pytest.raises(ValueError, match='Invalid excluded entity types'):
        await graphiti.add_episode_bulk(
            [raw_episode()],
            group_id=GROUP,
            entity_types={'Person': Person},
            excluded_entity_types=['Persn'],
        )

    graph_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_episode_bulk_accepts_known_excluded_entity_type(monkeypatch):
    class GraphWriteReached(Exception):
        pass

    monkeypatch.setattr(
        graphiti_module,
        'add_nodes_and_edges_bulk',
        AsyncMock(side_effect=GraphWriteReached),
    )
    graphiti = make_graphiti()

    with pytest.raises(GraphWriteReached):
        await graphiti.add_episode_bulk(
            [raw_episode()],
            group_id=GROUP,
            entity_types={'Person': Person},
            excluded_entity_types=['Person', 'Entity'],
        )
