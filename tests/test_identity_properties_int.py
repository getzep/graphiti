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

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import resolve_extracted_nodes


async def _driver_for_group(graph_driver, group_id: str):
    if graph_driver.provider == GraphProvider.FALKORDB:
        driver = graph_driver.clone(database=group_id)
        init_task = getattr(driver, '_init_task', None)
        if init_task is not None:
            await init_task
        return driver
    return graph_driver


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_by_attribute_values_scopes_by_group_and_label(graph_driver):
    person_group = f'identity-person-{uuid4().hex}'
    other_group = f'identity-other-{uuid4().hex}'
    nodes = [
        EntityNode(
            name='Alex',
            group_id=person_group,
            labels=['Entity', 'Person'],
            name_embedding=[0.1],
            attributes={'email': 'alex@example.com'},
        ),
        EntityNode(
            name='Alex contact record',
            group_id=person_group,
            labels=['Entity', 'Company'],
            name_embedding=[0.1],
            attributes={'email': 'alex@example.com'},
        ),
        EntityNode(
            name='Alex from another group',
            group_id=other_group,
            labels=['Entity', 'Person'],
            name_embedding=[0.1],
            attributes={'email': 'alex@example.com'},
        ),
    ]
    person_driver = await _driver_for_group(graph_driver, person_group)
    other_driver = await _driver_for_group(graph_driver, other_group)

    try:
        for node in nodes:
            driver = person_driver if node.group_id == person_group else other_driver
            await node.save(driver)

        matches = await EntityNode.get_by_attribute_values(
            person_driver,
            person_group,
            {'email': 'alex@example.com'},
            labels=['Person'],
        )

        assert [node.uuid for node in matches] == [nodes[0].uuid]
    finally:
        for node in nodes:
            driver = person_driver if node.group_id == person_group else other_driver
            await node.delete(driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resolve_nodes_matches_identity_property_without_llm(
    graph_driver,
    mock_embedder,
    monkeypatch,
):
    group_id = f'identity-resolve-{uuid4().hex}'
    driver = await _driver_for_group(graph_driver, group_id)
    stored_node = EntityNode(
        name='Alex Smith',
        group_id=group_id,
        labels=['Entity', 'Person'],
        name_embedding=[0.1],
        attributes={'email': 'alex@example.com'},
    )
    await stored_node.save(driver)

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock()
    clients = GraphitiClients.model_construct(
        driver=driver,
        embedder=mock_embedder,
        cross_encoder=MagicMock(),
        llm_client=llm_client,
        max_coroutines=1,
    )
    episode = EpisodicNode(
        name='conversation',
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test',
        content='Alex uses alex@example.com.',
        valid_at=utc_now(),
    )
    extracted_node = EntityNode(
        name='Alex from a different source',
        group_id=group_id,
        labels=['Entity', 'Person'],
        attributes={'email': 'alex@example.com'},
    )

    async def no_semantic_candidates(_clients, nodes):
        return [[] for _ in nodes]

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        no_semantic_candidates,
    )

    try:
        resolved, uuid_map, duplicates = await resolve_extracted_nodes(
            clients,
            [extracted_node],
            episode=episode,
            previous_episodes=[],
            identity_properties={'Person': ['email']},
        )

        assert resolved[0].uuid == stored_node.uuid
        assert uuid_map[extracted_node.uuid] == stored_node.uuid
        assert duplicates == [(extracted_node, stored_node)]
        llm_client.generate_response.assert_not_awaited()
    finally:
        await stored_node.delete(driver)
