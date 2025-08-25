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

from typing import Any

from graphiti_core.driver.driver import GraphProvider


def get_episode_node_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.NEPTUNE:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content, 
                entity_edges: join([x IN coalesce($entity_edges, []) | toString(x) ], '|'), created_at: $created_at, valid_at: $valid_at}
                RETURN n.uuid AS uuid
            """
        case GraphProvider.FALKORDB:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET n:$($group_label)
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
                RETURN n.uuid AS uuid
            """


def get_episode_node_save_bulk_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.NEPTUNE:
            return """
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, 
                    source: episode.source, content: episode.content, 
                entity_edges: join([x IN coalesce(episode.entity_edges, []) | toString(x) ], '|'), created_at: episode.created_at, valid_at: episode.valid_at}
                RETURN n.uuid AS uuid
            """
        case GraphProvider.FALKORDB:
            return """
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, source: episode.source, content: episode.content, 
                entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at}
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n:$(episode.group_label)
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, source: episode.source, content: episode.content, 
                entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at}
                RETURN n.uuid AS uuid
            """


EPISODIC_NODE_RETURN = """
    e.content AS content,
    e.created_at AS created_at,
    e.valid_at AS valid_at,
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.source_description AS source_description,
    e.source AS source,
    e.entity_edges AS entity_edges
"""

EPISODIC_NODE_RETURN_NEPTUNE = """
    e.content AS content,
    e.created_at AS created_at,
    e.valid_at AS valid_at,
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.source_description AS source_description,
    e.source AS source,
    split(e.entity_edges, ",") AS entity_edges
"""


def get_entity_node_save_query(provider: GraphProvider, labels: str) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                SET n:{labels}
                SET n = $entity_data
                RETURN n.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            label_subquery = ''
            for label in labels.split(':'):
                label_subquery += f' SET n:{label}\n'
            return f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                {label_subquery}
                SET n = removeKeyFromMap(removeKeyFromMap($entity_data, "labels"), "name_embedding")
                SET n.name_embedding = join([x IN coalesce($entity_data.name_embedding, []) | toString(x) ], ",")
                RETURN n.uuid AS uuid
            """
        case _:
            return f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                SET n:{labels}
                SET n = $entity_data
                WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)
                RETURN n.uuid AS uuid
            """


def get_entity_node_save_bulk_query(provider: GraphProvider, nodes: list[dict]) -> str | Any:
    match provider:
        case GraphProvider.FALKORDB:
            queries = []
            for node in nodes:
                for label in node['labels']:
                    queries.append(
                        (
                            f"""
                            UNWIND $nodes AS node
                            MERGE (n:Entity {{uuid: node.uuid}})
                            SET n:{label}
                            SET n = node
                            WITH n, node
                            SET n.name_embedding = vecf32(node.name_embedding)
                            RETURN n.uuid AS uuid
                            """,
                            {'nodes': [node]},
                        )
                    )
            return queries
        case GraphProvider.NEPTUNE:
            queries = []
            for node in nodes:
                labels = ''
                for label in node['labels']:
                    labels += f' SET n:{label}\n'
                queries.append(
                    f"""
                        UNWIND $nodes AS node
                        MERGE (n:Entity {{uuid: node.uuid}})
                        {labels}
                        SET n = removeKeyFromMap(removeKeyFromMap(node, "labels"), "name_embedding")
                        SET n.name_embedding = join([x IN coalesce(node.name_embedding, []) | toString(x) ], ",")
                        RETURN n.uuid AS uuid
                    """
                )
            return queries
        case _:  # Neo4j
            return """
                UNWIND $nodes AS node
                MERGE (n:Entity {uuid: node.uuid})
                SET n:$(node.labels)
                SET n = node
                WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)
                RETURN n.uuid AS uuid
            """


ENTITY_NODE_RETURN = """
    n.uuid AS uuid,
    n.name AS name,
    n.group_id AS group_id,
    n.created_at AS created_at,
    n.summary AS summary,
    labels(n) AS labels,
    properties(n) AS attributes
"""


def get_community_node_save_query(provider: GraphProvider) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return """
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at, name_embedding: vecf32($name_embedding)}
                RETURN n.uuid AS uuid
            """
        case GraphProvider.NEPTUNE:
            return """
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}        
                SET n.name_embedding = join([x IN coalesce($name_embedding, []) | toString(x) ], ",")
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            return """
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
                WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $name_embedding)
                RETURN n.uuid AS uuid
            """


COMMUNITY_NODE_RETURN = """
    n.uuid AS uuid,
    n.name AS name,
    n.name_embedding AS name_embedding,
    n.group_id AS group_id,
    n.summary AS summary,
    n.created_at AS created_at
"""

COMMUNITY_NODE_RETURN_NEPTUNE = """
    n.uuid AS uuid,
    n.name AS name,
    [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding,
    n.group_id AS group_id,
    n.summary AS summary,
    n.created_at AS created_at
"""
