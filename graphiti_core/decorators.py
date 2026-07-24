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
from graphiti_core.search.search_config import SearchResults

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])


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
                cloned = driver.clone(database=gid)
                await cloned.ensure_database_initialized()
                return await func(
                    self,
                    *args,
                    **{**kwargs, 'driver': cloned},
                )
            return await func(self, *args, **kwargs)

        # FalkorDB with multiple group_ids: run per graph and merge
        if is_falkor and group_ids and len(group_ids) > 1:
            # Execute for each group_id concurrently
            driver = self.clients.driver

            async def execute_for_group(gid: str):
                # Remove group_ids from args if it was passed positionally
                filtered_args = list(args)
                if group_ids_pos is not None and len(args) > group_ids_pos:
                    filtered_args.pop(group_ids_pos)

                cloned = driver.clone(database=gid)
                await cloned.ensure_database_initialized()

                return await func(
                    self,
                    *filtered_args,
                    **{**kwargs, 'group_ids': [gid], 'driver': cloned},
                )

            results = await semaphore_gather(
                *[execute_for_group(gid) for gid in group_ids],
                max_coroutines=getattr(self, 'max_coroutines', None),
            )

            # Merge results based on type
            if isinstance(results[0], SearchResults):
                return SearchResults.merge(results)
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


def handle_single_group_id(func: F) -> F:
    """
    Decorator for FalkorDB write methods that need to scope to a single group_id.
    Injects a cloned driver targeted at the group's graph instead of mutating self.driver.
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        group_id = kwargs.get('group_id')
        if group_id is None:
            pos = get_parameter_position(func, 'group_id')
            if pos is not None and len(args) > pos - 1:
                group_id = args[pos - 1]  # adjust for self
        if (
            group_id
            and hasattr(self, 'clients')
            and hasattr(self.clients, 'driver')
            and self.clients.driver.provider == GraphProvider.FALKORDB
            and group_id != self.clients.driver._database
        ):
            cloned = self.clients.driver.clone(database=group_id)
            await cloned.ensure_database_initialized()
            kwargs['driver'] = cloned
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
