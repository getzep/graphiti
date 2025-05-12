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

import json

from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import SearchResults


def format_edge_date_range(edge: EntityEdge) -> str:
    # return f"{datetime(edge.valid_at).strftime('%Y-%m-%d %H:%M:%S') if edge.valid_at else 'date unknown'} - {(edge.invalid_at.strftime('%Y-%m-%d %H:%M:%S') if edge.invalid_at else 'present')}"
    return f'{edge.valid_at if edge.valid_at else "date unknown"} - {(edge.invalid_at if edge.invalid_at else "present")}'


def search_results_to_context_string(search_results: SearchResults) -> str:
    """Reformats a set of SearchResults into a single string to pass directly to an LLM as context"""
    fact_json = [
        {
            'fact': edge.fact,
            'valid_at': str(edge.valid_at),
            'invalid_at': str(edge.invalid_at or 'Present'),
        }
        for edge in search_results.edges
    ]
    entity_json = [
        {'entity_name': node.name, 'summary': node.summary} for node in search_results.nodes
    ]
    episode_json = [
        {
            'source_description': episode.source_description,
            'content': episode.content,
        }
        for episode in search_results.episodes
    ]
    community_json = [
        {'community_name': community.name, 'summary': community.summary}
        for community in search_results.communities
    ]

    context_string = f"""
    FACTS and ENTITIES represent relevant context to the current conversation.
    COMMUNITIES represent a cluster of closely related entities.

    These are the most relevant facts and their valid and invalid dates. Facts are considered valid
    between their valid_at and invalid_at dates. Facts with an invalid_at date of "Present" are considered valid.
    <FACTS>
    {json.dumps(fact_json, indent=12)}
    </FACTS>
    <ENTITIES>
    {json.dumps(entity_json, indent=12)}
    </ENTITIES>
    <EPISODES>
    {json.dumps(episode_json, indent=12)}
    </EPISODES>
    <COMMUNITIES>
    {json.dumps(community_json, indent=12)}
    </COMMUNITIES>
"""

    return context_string
