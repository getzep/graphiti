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
from typing import Any, Awaitable, Callable, TypeVar

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.helpers import semaphore_gather
from graphiti_core.search.search_config import SearchResults

F = TypeVar('F', bound=Callable[..., Awaitable[Any]])


def handle_multiple_group_ids(func: F) -> F:
    """
    Decorator for FalkorDB methods that need to handle multiple group_ids.
    Runs the function for each group_id separately and merges results.
    """
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        group_ids = kwargs.get('group_ids')
        
        # Only handle FalkorDB with multiple group_ids
        if (hasattr(self, 'clients') and hasattr(self.clients, 'driver') and 
            self.clients.driver.provider == GraphProvider.FALKORDB and 
            group_ids and len(group_ids) > 1):
            
            # Execute for each group_id concurrently
            driver = self.clients.driver
            
            async def execute_for_group(gid: str):
                return await func(
                    self,
                    *args,
                    **{**kwargs, "group_ids": [gid], "driver": driver.clone(database=gid)},
                )
            
            results = await semaphore_gather(
                *[execute_for_group(gid) for gid in group_ids],
                max_coroutines=getattr(self, 'max_coroutines', None)
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
                        merged_tuple.append([item for component in component_results for item in component])
                    else:
                        merged_tuple.append(component_results)
                return tuple(merged_tuple)
            else:
                return results
        
        # Normal execution
        return await func(self, *args, **kwargs)
    
    return wrapper  # type: ignore
