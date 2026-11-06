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

from __future__ import annotations

import inspect
from typing import Any

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface

# Maps each ``GraphOperationsInterface`` method name to the FalkorDB driver
# attribute holding the concrete operations instance and the method to call on
# it. The interface methods take ``(_cls, driver, ...)`` while the operations
# classes take ``(executor, ...)``; the adapter below bridges the two.
_METHOD_MAP: dict[str, tuple[str, str]] = {
    # EntityNode
    'node_save': ('entity_node_ops', 'save'),
    'node_delete': ('entity_node_ops', 'delete'),
    'node_save_bulk': ('entity_node_ops', 'save_bulk'),
    'node_delete_by_group_id': ('entity_node_ops', 'delete_by_group_id'),
    'node_delete_by_uuids': ('entity_node_ops', 'delete_by_uuids'),
    'node_get_by_uuid': ('entity_node_ops', 'get_by_uuid'),
    'node_get_by_uuids': ('entity_node_ops', 'get_by_uuids'),
    'node_get_by_group_ids': ('entity_node_ops', 'get_by_group_ids'),
    'node_load_embeddings': ('entity_node_ops', 'load_embeddings'),
    'node_load_embeddings_bulk': ('entity_node_ops', 'load_embeddings_bulk'),
    # EpisodicNode
    'episodic_node_save': ('episode_node_ops', 'save'),
    'episodic_node_delete': ('episode_node_ops', 'delete'),
    'episodic_node_save_bulk': ('episode_node_ops', 'save_bulk'),
    'episodic_node_delete_by_group_id': ('episode_node_ops', 'delete_by_group_id'),
    'episodic_node_delete_by_uuids': ('episode_node_ops', 'delete_by_uuids'),
    'episodic_node_get_by_uuid': ('episode_node_ops', 'get_by_uuid'),
    'episodic_node_get_by_uuids': ('episode_node_ops', 'get_by_uuids'),
    'episodic_node_get_by_group_ids': ('episode_node_ops', 'get_by_group_ids'),
    'episodic_node_get_by_entity_node_uuid': ('episode_node_ops', 'get_by_entity_node_uuid'),
    'retrieve_episodes': ('episode_node_ops', 'retrieve_episodes'),
    # CommunityNode
    'community_node_save': ('community_node_ops', 'save'),
    'community_node_delete': ('community_node_ops', 'delete'),
    'community_node_save_bulk': ('community_node_ops', 'save_bulk'),
    'community_node_delete_by_group_id': ('community_node_ops', 'delete_by_group_id'),
    'community_node_delete_by_uuids': ('community_node_ops', 'delete_by_uuids'),
    'community_node_get_by_uuid': ('community_node_ops', 'get_by_uuid'),
    'community_node_get_by_uuids': ('community_node_ops', 'get_by_uuids'),
    'community_node_get_by_group_ids': ('community_node_ops', 'get_by_group_ids'),
    'community_node_load_name_embedding': ('community_node_ops', 'load_name_embedding'),
    # SagaNode
    'saga_node_save': ('saga_node_ops', 'save'),
    'saga_node_delete': ('saga_node_ops', 'delete'),
    'saga_node_save_bulk': ('saga_node_ops', 'save_bulk'),
    'saga_node_delete_by_group_id': ('saga_node_ops', 'delete_by_group_id'),
    'saga_node_delete_by_uuids': ('saga_node_ops', 'delete_by_uuids'),
    'saga_node_get_by_uuid': ('saga_node_ops', 'get_by_uuid'),
    'saga_node_get_by_uuids': ('saga_node_ops', 'get_by_uuids'),
    'saga_node_get_by_group_ids': ('saga_node_ops', 'get_by_group_ids'),
    # EntityEdge
    'edge_save': ('entity_edge_ops', 'save'),
    'edge_delete': ('entity_edge_ops', 'delete'),
    'edge_save_bulk': ('entity_edge_ops', 'save_bulk'),
    'edge_delete_by_uuids': ('entity_edge_ops', 'delete_by_uuids'),
    'edge_get_by_uuid': ('entity_edge_ops', 'get_by_uuid'),
    'edge_get_by_uuids': ('entity_edge_ops', 'get_by_uuids'),
    'edge_get_by_group_ids': ('entity_edge_ops', 'get_by_group_ids'),
    'edge_get_between_nodes': ('entity_edge_ops', 'get_between_nodes'),
    'edge_get_by_node_uuid': ('entity_edge_ops', 'get_by_node_uuid'),
    'edge_load_embeddings': ('entity_edge_ops', 'load_embeddings'),
    'edge_load_embeddings_bulk': ('entity_edge_ops', 'load_embeddings_bulk'),
    # EpisodicEdge
    'episodic_edge_save': ('episodic_edge_ops', 'save'),
    'episodic_edge_delete': ('episodic_edge_ops', 'delete'),
    'episodic_edge_save_bulk': ('episodic_edge_ops', 'save_bulk'),
    'episodic_edge_delete_by_uuids': ('episodic_edge_ops', 'delete_by_uuids'),
    'episodic_edge_get_by_uuid': ('episodic_edge_ops', 'get_by_uuid'),
    'episodic_edge_get_by_uuids': ('episodic_edge_ops', 'get_by_uuids'),
    'episodic_edge_get_by_group_ids': ('episodic_edge_ops', 'get_by_group_ids'),
    # CommunityEdge
    'community_edge_save': ('community_edge_ops', 'save'),
    'community_edge_delete': ('community_edge_ops', 'delete'),
    'community_edge_delete_by_uuids': ('community_edge_ops', 'delete_by_uuids'),
    'community_edge_get_by_uuid': ('community_edge_ops', 'get_by_uuid'),
    'community_edge_get_by_uuids': ('community_edge_ops', 'get_by_uuids'),
    'community_edge_get_by_group_ids': ('community_edge_ops', 'get_by_group_ids'),
    # HasEpisodeEdge
    'has_episode_edge_save': ('has_episode_edge_ops', 'save'),
    'has_episode_edge_delete': ('has_episode_edge_ops', 'delete'),
    'has_episode_edge_save_bulk': ('has_episode_edge_ops', 'save_bulk'),
    'has_episode_edge_delete_by_uuids': ('has_episode_edge_ops', 'delete_by_uuids'),
    'has_episode_edge_get_by_uuid': ('has_episode_edge_ops', 'get_by_uuid'),
    'has_episode_edge_get_by_uuids': ('has_episode_edge_ops', 'get_by_uuids'),
    'has_episode_edge_get_by_group_ids': ('has_episode_edge_ops', 'get_by_group_ids'),
    # NextEpisodeEdge
    'next_episode_edge_save': ('next_episode_edge_ops', 'save'),
    'next_episode_edge_delete': ('next_episode_edge_ops', 'delete'),
    'next_episode_edge_save_bulk': ('next_episode_edge_ops', 'save_bulk'),
    'next_episode_edge_delete_by_uuids': ('next_episode_edge_ops', 'delete_by_uuids'),
    'next_episode_edge_get_by_uuid': ('next_episode_edge_ops', 'get_by_uuid'),
    'next_episode_edge_get_by_uuids': ('next_episode_edge_ops', 'get_by_uuids'),
    'next_episode_edge_get_by_group_ids': ('next_episode_edge_ops', 'get_by_group_ids'),
    # Graph maintenance / search
    'clear_data': ('graph_ops', 'clear_data'),
    'get_community_clusters': ('graph_ops', 'get_community_clusters'),
    'determine_entity_community': ('graph_ops', 'determine_entity_community'),
    'get_mentioned_nodes': ('graph_ops', 'get_mentioned_nodes'),
    'get_communities_by_nodes': ('graph_ops', 'get_communities_by_nodes'),
}

# Interface parameter name -> operations parameter name.
_PARAM_ALIASES: dict[str, str] = {
    'driver': 'executor',
    'transaction': 'tx',
}


def _not_implemented_hook(name: str) -> Any:
    async def _not_implemented(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(name)

    return _not_implemented


def _make_adapter(interface_method: Any, ops_method: Any) -> Any:
    """Build an async callable that forwards interface args to an ops method."""
    # ``interface_method`` is fetched off the class, so it is an unbound
    # function whose signature starts with ``self``. Callers invoke the
    # interface as ``interface.method(_cls, driver, ...)`` and never pass
    # ``self``, so drop it before binding to keep the arguments aligned.
    interface_signature = inspect.signature(interface_method)
    interface_params = list(interface_signature.parameters.values())
    if interface_params and interface_params[0].name == 'self':
        interface_signature = interface_signature.replace(parameters=interface_params[1:])
    ops_params = inspect.signature(ops_method).parameters

    async def adapter(*args: Any, **kwargs: Any) -> Any:
        bound = interface_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call_kwargs: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            # ``_cls`` is the classmethod's ``cls`` and is not forwarded to
            # the ops classes.
            if name == '_cls':
                continue
            mapped = _PARAM_ALIASES.get(name, name)
            if mapped in ops_params:
                call_kwargs[mapped] = value
        return await ops_method(**call_kwargs)

    return adapter


class FalkorGraphOperationsInterface:
    """Facade exposing the FalkorDB operations classes via ``GraphOperationsInterface``.

    ``GraphDriver.graph_operations_interface`` is the dispatch point used by the
    node/edge classmethods. FalkorDB stores each ``group_id`` in its own physical
    graph, so those classmethods must reach the Falkor operations classes (which
    clone the driver per group) rather than the raw default-database fallback.
    """

    def __init__(self, driver: Any):
        self._driver = driver

    def __getattr__(self, name: str) -> Any:
        target = _METHOD_MAP.get(name)
        if target is None:
            # Absent from the map: raise on call so callers' NotImplementedError
            # fallbacks (query / graph_ops) still run.
            return _not_implemented_hook(name)

        ops_attr, ops_method_name = target
        ops = getattr(self._driver, ops_attr)
        ops_method = getattr(ops, ops_method_name, None)
        if ops_method is None:
            # Not implemented for FalkorDB: raise so callers fall back to raw Cypher.
            return _not_implemented_hook(name)

        interface_method = getattr(GraphOperationsInterface, name)
        return _make_adapter(interface_method, ops_method)
