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

from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import SearchResults


def format_edge_date_range(edge: EntityEdge) -> str:
    # return f"{datetime(edge.valid_at).strftime('%Y-%m-%d %H:%M:%S') if edge.valid_at else 'date unknown'} - {(edge.invalid_at.strftime('%Y-%m-%d %H:%M:%S') if edge.invalid_at else 'present')}"
    return f'{edge.valid_at if edge.valid_at else "date unknown"} - {(edge.invalid_at if edge.invalid_at else "present")}'


def search_results_to_context_string(search_results: SearchResults) -> str:
    """Reformats a set of SearchResults into a single string to pass directly to an LLM as context"""
    context_string = """FACTS and ENTITIES represent relevant context to the current conversation.
                        COMMUNITIES represent a cluster of closely related entities.

                        # These are the most relevant facts and their valid date ranges
                        # format: FACT (Date range: from - to)
                    """
    context_string += '<FACTS>\n'
    for edge in search_results.edges:
        context_string += f'- {edge.fact} ({format_edge_date_range(edge)})\n'
    context_string += '</FACTS>\n'
    context_string += '<ENTITIES>\n'
    for node in search_results.nodes:
        context_string += f'- {node.name}: {node.summary}\n'
    context_string += '</ENTITIES>\n'
    context_string += '<COMMUNITIES>\n'
    for community in search_results.communities:
        context_string += f'- {community.name}: {community.summary}\n'
    context_string += '</COMMUNITIES>\n'

    return context_string
