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

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.helpers import semaphore_gather
from graphiti_core.search.search_config import (
    EdgeReranker,
    NodeReranker,
    SearchResults,
)

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])


async def _rerank_cross_group(
    merged: SearchResults, call_args: dict[str, Any], clients: Any, limit: int
) -> SearchResults:
    """
    Re-rank SearchResults after cross-group merge to ensure global relevance ordering.

    Per-group rerankers (RRF, MMR) produce similar top scores (e.g., 1.0) across groups,
    making weak #1 results indistinguishable from strong #1 results. This function
    re-applies cross-encoder reranking globally when available.

    Parameters
    ----------
    merged : SearchResults
        The merged results from all groups
    call_args : dict
        The bound function arguments (query, config, etc.)
    clients : GraphitiClients
        The clients object with cross_encoder
    limit : int
        The final result limit

    Returns
    -------
    SearchResults
        Re-ranked results with globally comparable scores
    """
    config = call_args.get('config')
    query = call_args.get('query')

    if not config or not query:
        return merged

    cross_encoder = clients.cross_encoder if hasattr(clients, 'cross_encoder') else None
    if not cross_encoder:
        return merged

    # Re-rank edges: use cross-encoder for cross_encoder config OR as secondary
    # reranker for RRF/other configs to distinguish top results with similar scores
    if config.edge_config and merged.edges:
        edge_needs_rerank = (
            config.edge_config.reranker == EdgeReranker.cross_encoder
            or config.edge_config.reranker == EdgeReranker.rrf
        )
        if edge_needs_rerank:
            edge_facts = [edge.fact for edge in merged.edges[: 2 * limit]]
            reranked_facts = await cross_encoder.rank(query, edge_facts)
            fact_to_edge = {edge.fact: edge for edge in merged.edges[: 2 * limit]}

            merged.edges = [fact_to_edge[fact] for fact, _score in reranked_facts[:limit]]
            merged.edge_reranker_scores = [score for _fact, score in reranked_facts[:limit]]

    # Re-rank nodes: use cross-encoder for cross_encoder config OR as secondary
    # reranker for RRF/other configs to distinguish top results with similar scores
    if config.node_config and merged.nodes:
        node_needs_rerank = (
            config.node_config.reranker == NodeReranker.cross_encoder
            or config.node_config.reranker == NodeReranker.rrf
        )
        if node_needs_rerank:
            node_names = [node.name for node in merged.nodes[: 2 * limit]]
            reranked_names = await cross_encoder.rank(query, node_names)
            name_to_node = {node.name: node for node in merged.nodes[: 2 * limit]}

            merged.nodes = [name_to_node[name] for name, _score in reranked_names[:limit]]
            merged.node_reranker_scores = [score for _name, score in reranked_names[:limit]]

    return merged


def handle_multiple_group_ids(func: F) -> F:
    """
    Decorator for FalkorDB methods that need to handle multiple group_ids.
    Runs the function for each group_id separately and merges results.

    Also routes a *single* group_id to the matching FalkorDB graph via a
    call-scoped driver clone. Without this, add_episode re-binds the shared
    driver for writes while search/retrieve with one group_id query the
    driver's default database and silently return empty results (#1659).
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        group_ids_func_pos = get_parameter_position(func, 'group_ids')
        group_ids_pos = (
            group_ids_func_pos - 1 if group_ids_func_pos is not None else None
        )  # Adjust for zero-based index
        group_ids = kwargs.get('group_ids')

        # If not in kwargs and position exists, get from args
        if group_ids is None and group_ids_pos is not None and len(args) > group_ids_pos:
            group_ids = args[group_ids_pos]

        is_falkor = (
            hasattr(self, 'clients')
            and hasattr(self.clients, 'driver')
            and self.clients.driver.provider == GraphProvider.FALKORDB
        )

        # FalkorDB: one group_id still needs the graph named after that id.
        # Clone is call-scoped so we never reassign self.driver / self.clients.driver.
        if is_falkor and group_ids and len(group_ids) == 1:
            gid = group_ids[0]
            driver = self.clients.driver
            if gid != getattr(driver, '_database', None):
                return await func(
                    self,
                    *args,
                    **{**kwargs, 'driver': driver.clone(database=gid)},
                )
            return await func(self, *args, **kwargs)

        # FalkorDB with multiple group_ids: run per graph and merge
        if is_falkor and group_ids and len(group_ids) > 1:
            # Execute for each group_id concurrently
            driver = self.clients.driver

            async def execute_for_group(gid: str):
                # Bind the call by name, per task, from the original args.
                # Popping group_ids from its original positional slot shifts
                # every later positional argument down, so a caller that
                # passed driver positionally collided with the keyword
                # injected here (TypeError: got multiple values for argument
                # 'driver') (#1758). Re-binding from the signature normalizes
                # every argument to its declared name before the per-group
                # rewrite; a fresh BoundArguments per task also avoids
                # sharing one mutated-in-place object across concurrent
                # tasks, which is safe only as long as nothing awaits
                # between the rewrite and the read.
                bound = inspect.signature(func).bind(self, *args, **kwargs)
                bound.arguments['group_ids'] = [gid]
                bound.arguments['driver'] = driver.clone(database=gid)
                return await func(*bound.args, **bound.kwargs)

            results = await semaphore_gather(
                *[execute_for_group(gid) for gid in group_ids],
                max_coroutines=getattr(self, 'max_coroutines', None),
            )

            # Merge results based on type
            if isinstance(results[0], SearchResults):
                merge_limit = None
                call_bound = inspect.signature(func).bind(self, *args, **kwargs)
                call_bound.apply_defaults()

                # Extract limit from config (search_) or num_results (search)
                config = call_bound.arguments.get('config')
                if config is not None and hasattr(config, 'limit'):
                    merge_limit = config.limit
                elif 'num_results' in call_bound.arguments:
                    merge_limit = call_bound.arguments['num_results']

                merged = SearchResults.merge(results, limit=merge_limit)

                # Re-rank across groups after merge to ensure global relevance ordering
                # when per-group rerankers produce similar top scores (#2014)
                if merge_limit and len(group_ids) > 1:
                    merged = await _rerank_cross_group(
                        merged, call_bound.arguments, self.clients, merge_limit
                    )

                return merged
            elif isinstance(results[0], list):
                return [item for result in results for item in result]
            elif isinstance(results[0], tuple):
                # Handle tuple outputs (like build_communities returning (nodes, edges))
                merged_tuple = []
                for i in range(len(results[0])):
                    component_results = [result[i] for result in results]
                    if isinstance(component_results[0], list):
                        merged_tuple.append(
                            [item for component in component_results for item in component]
                        )
                    else:
                        merged_tuple.append(component_results)
                return tuple(merged_tuple)
            else:
                return results

        # Normal execution
        return await func(self, *args, **kwargs)

    return wrapper  # type: ignore


def get_parameter_position(func: Callable, param_name: str) -> int | None:
    """
    Returns the positional index of a parameter in the function signature.
    If the parameter is not found, returns None.
    """
    sig = inspect.signature(func)
    for idx, (name, _param) in enumerate(sig.parameters.items()):
        if name == param_name:
            return idx
    return None
