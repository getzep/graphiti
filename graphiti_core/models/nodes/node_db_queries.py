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
        case GraphProvider.KUZU:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
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
        case GraphProvider.KUZU:
            return """
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
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
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, source: episode.source, content: episode.content, 
                entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at}
                RETURN n.uuid AS uuid
            """


EPISODIC_NODE_RETURN = """
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.source AS source,
    e.source_description AS source_description,
    e.content AS content,
    e.valid_at AS valid_at,
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


def get_entity_node_save_query(provider: GraphProvider, labels: str, has_aoss: bool = False) -> str:
    match provider:
        case GraphProvider.FALKORDB:
            return f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                SET n:{labels}
                SET n = $entity_data
                RETURN n.uuid AS uuid
            """
        case GraphProvider.KUZU:
            return """
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                WITH n
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
            save_embedding_query = (
                'WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)'
                if not has_aoss
                else ''
            )
            return (
                f"""
                MERGE (n:Entity {{uuid: $entity_data.uuid}})
                SET n:{labels}
                SET n = $entity_data
                """
                + save_embedding_query
                + """
                RETURN n.uuid AS uuid
            """
            )


def get_entity_node_save_bulk_query(
    provider: GraphProvider, nodes: list[dict], has_aoss: bool = False
) -> str | Any:
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
        case GraphProvider.KUZU:
            return """
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                RETURN n.uuid AS uuid
            """
        case _:  # Neo4j
            save_embedding_query = (
                'WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)'
                if not has_aoss
                else ''
            )
            return (
                """
                    UNWIND $nodes AS node
                    MERGE (n:Entity {uuid: node.uuid})
                    SET n:$(node.labels)
                    SET n = node
                    """
                + save_embedding_query
                + """
                RETURN n.uuid AS uuid
            """
            )


def get_entity_node_return_query(provider: GraphProvider) -> str:
    # `name_embedding` is not returned by default and must be loaded manually using `load_name_embedding()`.
    if provider == GraphProvider.KUZU:
        return """
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.labels AS labels,
            n.created_at AS created_at,
            n.summary AS summary,
            n.attributes AS attributes
        """

    return """
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
        case GraphProvider.KUZU:
            return """
                MERGE (n:Community {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary
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
    c.uuid AS uuid,
    c.name AS name,
    c.group_id AS group_id,
    c.created_at AS created_at,
    c.name_embedding AS name_embedding,
    c.summary AS summary
"""

COMMUNITY_NODE_RETURN_NEPTUNE = """
    n.uuid AS uuid,
    n.name AS name,
    [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding,
    n.group_id AS group_id,
    n.summary AS summary,
    n.created_at AS created_at
"""
