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

EPISODIC_NODE_SAVE = """
    MERGE (n:Episodic {uuid: $uuid})
    SET
        n.name = $name,
        n.group_id = $group_id,
        n.source_description = $source_description,
        n.source = $source,
        n.content = $content,
        n.entity_edges = $entity_edges,
        n.created_at = $created_at,
        n.valid_at = $valid_at
    RETURN n.uuid AS uuid
"""

def get_episodic_node_save_bulk_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            UNWIND CAST($episodes AS STRUCT(uuid STRING, name STRING, group_id STRING, source_description STRING, source STRING, content STRING, entity_edges STRING[], created_at TIMESTAMP, valid_at TIMESTAMP, labels STRING[])[]) AS episode
            MERGE (n:Episodic {uuid: episode.uuid})
            SET
                n.name = episode.name,
                n.group_id = episode.group_id,
                n.source_description = episode.source_description,
                n.source = episode.source,
                n.content = episode.content,
                n.entity_edges = episode.entity_edges,
                n.created_at = episode.created_at,
                n.valid_at = episode.valid_at
            RETURN n.uuid AS uuid
        """

    return """
        UNWIND $episodes AS episode
        MERGE (n:Episodic {uuid: episode.uuid})
        SET
            n.name = episode.name,
            n.group_id = episode.group_id,
            n.source_description = episode.source_description,
            n.source = episode.source,
            n.content = episode.content,
            n.entity_edges = episode.entity_edges,
            n.created_at = episode.created_at,
            n.valid_at = episode.valid_at
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


def get_entity_node_save_query(provider: GraphProvider, labels: str) -> str:
    if provider == GraphProvider.FALKORDB:
        return f"""
            MERGE (n:Entity {{uuid: $entity_data.uuid}})
            SET n:{labels}
            SET n = $entity_data
            RETURN n.uuid AS uuid
        """

    if provider == GraphProvider.KUZU:
        return """
            MERGE (n:Entity {uuid: $uuid})
            SET
                n.labels = $labels,
                n.name = $name,
                n.name_embedding = $name_embedding,
                n.group_id = $group_id,
                n.summary = $summary,
                n.created_at = $created_at,
                n.attributes = $attributes
            WITH n
            RETURN n.uuid AS uuid
        """

    return f"""
        MERGE (n:Entity {{uuid: $entity_data.uuid}})
        SET n:{labels}
        SET n = $entity_data
        WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)
        RETURN n.uuid AS uuid
    """


def get_entity_node_save_bulk_query(provider: GraphProvider, nodes) -> str | Any:
    if provider == GraphProvider.FALKORDB:
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

    if provider == GraphProvider.KUZU:
        return """
            UNWIND CAST($nodes AS STRUCT(uuid STRING, labels STRING[], name STRING, name_embedding FLOAT[], group_id STRING, summary STRING, created_at TIMESTAMP, attributes STRING)[]) AS node
            MERGE (n:Entity {uuid: node.uuid})
            SET
                n.labels =node.labels,
                n.name = node.name,
                n.name_embedding = node.name_embedding,
                n.group_id = node.group_id,
                n.summary = node.summary,
                n.created_at = node.created_at,
                n.attributes = node.attributes
            RETURN n.uuid AS uuid
        """

    return """
        UNWIND $nodes AS node
        MERGE (n:Entity {uuid: node.uuid})
        SET n:$(node.labels)
        SET n = node
        WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)
        RETURN n.uuid AS uuid
    """


def get_entity_node_return_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.created_at AS created_at,
            n.summary AS summary,
            n.labels AS labels,
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
    if provider == GraphProvider.FALKORDB or provider == GraphProvider.KUZU:
        return """
            MERGE (n:Community {uuid: $uuid})
            SET
                n.name = $name,
                n.group_id = $group_id,
                n.summary = $summary,
                n.created_at = $created_at,
                n.name_embedding = $name_embedding
            RETURN n.uuid AS uuid
        """

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
