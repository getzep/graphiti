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

import asyncio
import logging
import os
from datetime import datetime
from time import time
from typing import Callable

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

from core.edges import EntityEdge, EpisodicEdge
from core.llm_client import LLMClient, LLMConfig, OpenAIClient
from core.nodes import EntityNode, EpisodicNode
from core.search.search import SearchConfig, hybrid_search
from core.search.search_utils import (
    get_relevant_edges,
    get_relevant_nodes,
)
from core.utils import (
    build_episodic_edges,
    retrieve_episodes,
)
from core.utils.bulk_utils import (
    BulkEpisode,
    dedupe_edges_bulk,
    dedupe_nodes_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from core.utils.maintenance.edge_operations import (
    dedupe_extracted_edges,
    extract_edges,
)
from core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    build_indices_and_constraints,
)
from core.utils.maintenance.node_operations import dedupe_extracted_nodes, extract_nodes
from core.utils.maintenance.temporal_operations import (
    extract_edge_dates,
    invalidate_edges,
    prepare_edges_for_invalidation,
)

logger = logging.getLogger(__name__)

load_dotenv()


class Graphiti:
    def __init__(self, uri: str, user: str, password: str, llm_client: LLMClient | None = None):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = 'neo4j'
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient(
                LLMConfig(
                    api_key=os.getenv('OPENAI_API_KEY', default=''),
                    model='gpt-4o-mini',
                    base_url='https://api.openai.com/v1',
                )
            )

    def close(self):
        self.driver.close()

    async def build_indices_and_constraints(self):
        await build_indices_and_constraints(self.driver)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
    ) -> list[EpisodicNode]:
        """Retrieve the last n episodic nodes from the graph"""
        return await retrieve_episodes(self.driver, reference_time, last_n)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        success_callback: Callable | None = None,
        error_callback: Callable | None = None,
    ):
        """Process an episode and update the graph"""
        try:
            start = time()

            nodes: list[EntityNode] = []
            entity_edges: list[EntityEdge] = []
            episodic_edges: list[EpisodicEdge] = []
            embedder = self.llm_client.get_embedder()
            now = datetime.now()

            previous_episodes = await self.retrieve_episodes(reference_time)
            episode = EpisodicNode(
                name=name,
                labels=[],
                source='messages',
                content=episode_body,
                source_description=source_description,
                created_at=now,
                valid_at=reference_time,
            )

            extracted_nodes = await extract_nodes(self.llm_client, episode, previous_episodes)
            logger.info(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

            # Calculate Embeddings

            await asyncio.gather(
                *[node.generate_name_embedding(embedder) for node in extracted_nodes]
            )
            existing_nodes = await get_relevant_nodes(extracted_nodes, self.driver)
            logger.info(f'Extracted nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')
            touched_nodes, _, brand_new_nodes = await dedupe_extracted_nodes(
                self.llm_client, extracted_nodes, existing_nodes
            )
            logger.info(f'Adjusted touched nodes: {[(n.name, n.uuid) for n in touched_nodes]}')
            nodes.extend(touched_nodes)

            extracted_edges = await extract_edges(
                self.llm_client, episode, touched_nodes, previous_episodes
            )

            await asyncio.gather(*[edge.generate_embedding(embedder) for edge in extracted_edges])

            existing_edges = await get_relevant_edges(extracted_edges, self.driver)
            logger.info(f'Existing edges: {[(e.name, e.uuid) for e in existing_edges]}')
            logger.info(f'Extracted edges: {[(e.name, e.uuid) for e in extracted_edges]}')

            # deduped_edges = await dedupe_extracted_edges_v2(
            #     self.llm_client,
            #     extract_node_and_edge_triplets(extracted_edges, nodes),
            #     extract_node_and_edge_triplets(existing_edges, nodes),
            # )

            deduped_edges = await dedupe_extracted_edges(
                self.llm_client,
                extracted_edges,
                existing_edges,
            )

            edge_touched_node_uuids = [n.uuid for n in brand_new_nodes]
            for edge in deduped_edges:
                edge_touched_node_uuids.append(edge.source_node_uuid)
                edge_touched_node_uuids.append(edge.target_node_uuid)

            (
                old_edges_with_nodes_pending_invalidation,
                new_edges_with_nodes,
            ) = prepare_edges_for_invalidation(
                existing_edges=existing_edges, new_edges=deduped_edges, nodes=nodes
            )

            invalidated_edges = await invalidate_edges(
                self.llm_client,
                old_edges_with_nodes_pending_invalidation,
                new_edges_with_nodes,
                episode,
                previous_episodes,
            )

            for edge in invalidated_edges:
                for existing_edge in existing_edges:
                    if existing_edge.uuid == edge.uuid:
                        existing_edge.expired_at = edge.expired_at
                for deduped_edge in deduped_edges:
                    if deduped_edge.uuid == edge.uuid:
                        deduped_edge.expired_at = edge.expired_at
                edge_touched_node_uuids.append(edge.source_node_uuid)
                edge_touched_node_uuids.append(edge.target_node_uuid)

            edges_to_save = existing_edges + deduped_edges

            for edge_to_extract_dates_from in edges_to_save:
                valid_at, invalid_at, _ = await extract_edge_dates(
                    self.llm_client,
                    edge_to_extract_dates_from,
                    episode.valid_at,
                    episode,
                    previous_episodes,
                )
                edge_to_extract_dates_from.valid_at = valid_at
                edge_to_extract_dates_from.invalid_at = invalid_at
            entity_edges.extend(edges_to_save)

            edge_touched_node_uuids = list(set(edge_touched_node_uuids))
            involved_nodes = [node for node in nodes if node.uuid in edge_touched_node_uuids]

            logger.info(f'Edge touched nodes: {[(n.name, n.uuid) for n in involved_nodes]}')

            logger.info(f'Invalidated edges: {[(e.name, e.uuid) for e in invalidated_edges]}')

            logger.info(f'Deduped edges: {[(e.name, e.uuid) for e in deduped_edges]}')

            episodic_edges.extend(
                build_episodic_edges(
                    # There may be an overlap between new_nodes and affected_nodes, so we're deduplicating them
                    involved_nodes,
                    episode,
                    now,
                )
            )
            # Important to append the episode to the nodes at the end so that self referencing episodic edges are not built
            logger.info(f'Built episodic edges: {episodic_edges}')

            # Future optimization would be using batch operations to save nodes and edges
            await episode.save(self.driver)
            await asyncio.gather(*[node.save(self.driver) for node in nodes])
            await asyncio.gather(*[edge.save(self.driver) for edge in episodic_edges])
            await asyncio.gather(*[edge.save(self.driver) for edge in entity_edges])

            end = time()
            logger.info(f'Completed add_episode in {(end-start) * 1000} ms')
            # for node in nodes:
            #     if isinstance(node, EntityNode):
            #         await node.update_summary(self.driver)
            if success_callback:
                await success_callback(episode)
        except Exception as e:
            if error_callback:
                await error_callback(episode, e)
            else:
                raise e

    async def add_episode_bulk(
        self,
        bulk_episodes: list[BulkEpisode],
    ):
        try:
            start = time()
            embedder = self.llm_client.get_embedder()
            now = datetime.now()

            episodes = [
                EpisodicNode(
                    name=episode.name,
                    labels=[],
                    source='messages',
                    content=episode.content,
                    source_description=episode.source_description,
                    created_at=now,
                    valid_at=episode.reference_time,
                )
                for episode in bulk_episodes
            ]

            # Save all the episodes
            await asyncio.gather(*[episode.save(self.driver) for episode in episodes])

            # Get previous episode context for each episode
            episode_pairs = await retrieve_previous_episodes_bulk(self.driver, episodes)

            # Extract all nodes and edges
            (
                extracted_nodes,
                extracted_edges,
                episodic_edges,
            ) = await extract_nodes_and_edges_bulk(self.llm_client, episode_pairs)

            # Generate embeddings
            await asyncio.gather(
                *[node.generate_name_embedding(embedder) for node in extracted_nodes],
                *[edge.generate_embedding(embedder) for edge in extracted_edges],
            )

            # Dedupe extracted nodes
            nodes, uuid_map = await dedupe_nodes_bulk(self.driver, self.llm_client, extracted_nodes)

            # save nodes to KG
            await asyncio.gather(*[node.save(self.driver) for node in nodes])

            # re-map edge pointers so that they don't point to discard dupe nodes
            extracted_edges_with_resolved_pointers: list[EntityEdge] = resolve_edge_pointers(
                extracted_edges, uuid_map
            )
            episodic_edges_with_resolved_pointers: list[EpisodicEdge] = resolve_edge_pointers(
                episodic_edges, uuid_map
            )

            # save episodic edges to KG
            await asyncio.gather(
                *[edge.save(self.driver) for edge in episodic_edges_with_resolved_pointers]
            )

            # Dedupe extracted edges
            edges = await dedupe_edges_bulk(
                self.driver, self.llm_client, extracted_edges_with_resolved_pointers
            )
            logger.info(f'extracted edge length: {len(edges)}')

            # invalidate edges

            # save edges to KG
            await asyncio.gather(*[edge.save(self.driver) for edge in edges])

            end = time()
            logger.info(f'Completed add_episode_bulk in {(end-start) * 1000} ms')

        except Exception as e:
            raise e

    async def search(self, query: str, num_results=10):
        search_config = SearchConfig(num_episodes=0, num_results=num_results)
        edges = (
            await hybrid_search(
                self.driver,
                self.llm_client.get_embedder(),
                query,
                datetime.now(),
                search_config,
            )
        ).edges

        facts = [edge.fact for edge in edges]

        return facts

    async def _search(self, query: str, timestamp: datetime, config: SearchConfig):
        return await hybrid_search(
            self.driver, self.llm_client.get_embedder(), query, timestamp, config
        )
