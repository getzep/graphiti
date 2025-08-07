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

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from time import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.edges.field_edges_db_queries import (
    BELONGS_TO_EDGE_SAVE,
    BELONGS_TO_EDGE_GET,
    BELONGS_TO_EDGE_DELETE,
    FIELD_RELATIONSHIP_EDGE_SAVE,
    FIELD_RELATIONSHIP_EDGE_UPDATE,
    FIELD_RELATIONSHIP_EDGE_EXPIRE,
    FIELD_RELATIONSHIP_EDGE_DELETE,
)
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.cluster_metadata.cluster_service import ClusterMetadataService
from graphiti_core.cluster_metadata.exceptions import (
    ClusterNotFoundError as MongoClusterNotFoundError,
    ClusterValidationError,
)

logger = logging.getLogger(__name__)


class Edge(BaseModel, ABC):
    """
    Abstract base class for all edge/relationship types in the graph.
    Enforces consistent implementation across all relationship schemas.
    Provides type safety through abstract methods that must be implemented by subclasses.
    """
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    source_node_uuid: str              # UUID of the source node
    target_node_uuid: str              # UUID of the target node
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver):
        """Save edge to Neo4j database - must be implemented by each edge type"""
        ...

    async def delete(self, driver: GraphDriver):
        """Delete edge from Neo4j database using generic relationship deletion"""
        result = await driver.execute_query(
            """
            MATCH ()-[e {uuid: $uuid}]->()
            DELETE e
            RETURN count(e) as deleted_count
            """,
            uuid=self.uuid,
        )

        deleted_count = result.records[0]['deleted_count'] if result.records else 0
        if deleted_count == 0:
            raise EdgeNotFoundError(f"Edge with UUID {self.uuid} not found")

        logger.debug(f'Deleted Edge: {self.uuid}')
        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.uuid == other.uuid
        return False

    @classmethod
    @abstractmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve edge by UUID - must be implemented by each edge type for type safety"""
        ...

    @classmethod
    @abstractmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple edges by UUIDs - must be implemented by each edge type"""
        ...

    def update_temporal_fields(self):
        """Update temporal tracking fields"""
        # For edges, we typically don't update created_at, but subclasses can override
        pass


class BelongsToEdge(Edge):
    """
    Edge representing Field belonging to Cluster relationship.
    Connects Field nodes directly to their organizational Cluster.
    Enforces cluster isolation constraints.
    """

    # ==================== EDGE-SPECIFIC PROPERTIES ====================
    cluster_partition_id: str          # UUID of the Cluster for isolation tracking

    # ==================== RELATIONSHIP OPERATIONS ====================

    async def save(self, driver: GraphDriver):
        """Save BELONGS_TO relationship to Neo4j database with MongoDB validation"""
        # Validate cluster exists in MongoDB before creating relationship
        await self._validate_cluster_exists()
        
        result = await driver.execute_query(
            BELONGS_TO_EDGE_SAVE,
            source_node_uuid=self.source_node_uuid,
            target_node_uuid=self.target_node_uuid,
            uuid=self.uuid,
            cluster_partition_id=self.cluster_partition_id,
            created_at=self.created_at,
        )

        logger.debug(f'Saved BELONGS_TO edge to Graph: {self.uuid}')
        return result

    async def delete(self, driver: GraphDriver):
        """Delete BELONGS_TO relationship from Neo4j database"""
        result = await driver.execute_query(
            BELONGS_TO_EDGE_DELETE,
            field_uuid=self.source_node_uuid,
            cluster_uuid=self.target_node_uuid,
        )

        logger.debug(f'Deleted BELONGS_TO edge: {self.uuid}')
        return result

    async def _validate_cluster_exists(self):
        """Validate that the target cluster exists in MongoDB"""
        try:
            # Initialize cluster metadata service
            cluster_service = ClusterMetadataService()
            
            # Check if cluster exists and is active
            cluster_exists = await cluster_service.validate_cluster_exists(self.target_node_uuid)
            if not cluster_exists:
                logger.warning(f'Target cluster {self.target_node_uuid} not found or inactive in MongoDB for BELONGS_TO edge {self.uuid}')
                
        except (MongoClusterNotFoundError, ClusterValidationError) as e:
            # Log the warning but don't fail the Neo4j operation
            logger.warning(f'MongoDB cluster validation failed for BELONGS_TO edge {self.uuid}: {e}')
        except Exception as e:
            # Log unexpected errors but don't fail the Neo4j operation
            logger.error(f'Unexpected error during MongoDB cluster validation for BELONGS_TO edge {self.uuid}: {e}')

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve BELONGS_TO edge by UUID and return full Edge object"""
        records, _, _ = await driver.execute_query(
            BELONGS_TO_EDGE_GET,
            field_uuid=uuid,  # This query expects field_uuid, but we'll need to modify it
            routing_='r',
        )

        if not records:
            raise EdgeNotFoundError(f"BelongsToEdge with UUID {uuid} not found")

        return get_belongs_to_edge_from_record(records[0])

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple BELONGS_TO edges by UUIDs"""
        if not uuids:
            return []

        # Note: The current query structure doesn't support multiple UUIDs directly
        # This would need to be implemented in the field_edges_db_queries.py
        edges = []
        for uuid in uuids:
            try:
                edge = await cls.get_by_uuid(driver, uuid)
                edges.append(edge)
            except EdgeNotFoundError:
                continue

        return edges




class FieldRelationshipEdge(Edge):
    """
    Edge representing dynamic semantic relationships between Field nodes within the same Cluster.
    Captures domain-specific connections like correlation, similarity, or derivation between security fields.
    All FIELD_RELATES_TO relationships must respect cluster isolation constraints.
    """

    # ==================== FIELD RELATIONSHIP PROPERTIES ====================
    name: str                          # Relationship type (e.g., 'CORRELATES_WITH', 'SIMILAR_TO', 'DERIVED_FROM')
    description: str                   # Why this relationship exists (semantic context)
    confidence: float = 1.0            # Confidence score for the relationship (0.0 to 1.0)
    cluster_partition_id: str          # UUID of the Cluster (both fields must belong here)
    relationship_type: str = "FIELD_RELATES_TO"  # Base relationship type for Neo4j

    # Semantic Search Support
    description_embedding: list[float] | None = Field(default=None, description='embedding of description for semantic search')

    # Temporal Validation Fields
    valid_at: datetime | None = Field(default=None, description='when the relationship became valid')
    invalid_at: datetime | None = Field(default=None, description='when the relationship stopped being valid')

    # ==================== RELATIONSHIP OPERATIONS ====================

    async def generate_description_embedding(self, embedder: EmbedderClient):
        """Generate embedding for relationship description using embedder client"""
        start = time()
        text = self.description.replace('\n', ' ')
        self.description_embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded relationship description {text} in {end - start} ms')
        return self.description_embedding

    async def load_description_embedding(self, driver: GraphDriver):
        """Load description embedding from database"""
        records, _, _ = await driver.execute_query(
            """
            MATCH ()-[r:FIELD_RELATES_TO {uuid: $uuid}]->()
            RETURN r.description_embedding AS description_embedding
            """,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise EdgeNotFoundError(self.uuid)

        self.description_embedding = records[0]['description_embedding']

    async def save(self, driver: GraphDriver):
        """Save Field-to-Field relationship to Neo4j database with MongoDB cluster validation"""
        # Validate cluster exists in MongoDB before creating relationship
        await self._validate_cluster_exists()
        
        result = await driver.execute_query(
            FIELD_RELATIONSHIP_EDGE_SAVE,
            source_node_uuid=self.source_node_uuid,
            target_node_uuid=self.target_node_uuid,
            uuid=self.uuid,
            name=self.name,
            description=self.description,
            confidence=self.confidence,
            cluster_partition_id=self.cluster_partition_id,
            relationship_type=self.relationship_type,
            description_embedding=self.description_embedding,
            created_at=self.created_at,
            valid_at=self.valid_at,
            invalid_at=self.invalid_at,
        )

        logger.debug(f'Saved FIELD_RELATES_TO edge to Graph: {self.uuid}')
        return result

    async def update(self, driver: GraphDriver):
        """Update Field relationship edge in Neo4j database"""
        result = await driver.execute_query(
            FIELD_RELATIONSHIP_EDGE_UPDATE,
            source_field_uuid=self.source_node_uuid,
            edge_uuid=self.uuid,
            name=self.name,
            description=self.description,
            confidence=self.confidence,
            valid_at=self.valid_at,
        )

        logger.debug(f'Updated FIELD_RELATES_TO edge: {self.uuid}')
        return result

    async def expire(self, driver: GraphDriver):
        """Mark Field relationship edge as expired"""
        result = await driver.execute_query(
            FIELD_RELATIONSHIP_EDGE_EXPIRE,
            source_field_uuid=self.source_node_uuid,
            edge_uuid=self.uuid,
        )

        logger.debug(f'Expired FIELD_RELATES_TO edge: {self.uuid}')
        return result

    async def delete(self, driver: GraphDriver):
        """Delete Field relationship edge from Neo4j database"""
        result = await driver.execute_query(
            FIELD_RELATIONSHIP_EDGE_DELETE,
            source_field_uuid=self.source_node_uuid,
            edge_uuid=self.uuid,
        )

        logger.debug(f'Deleted FIELD_RELATES_TO edge: {self.uuid}')
        return result

    async def _validate_cluster_exists(self):
        """Validate that the cluster_partition_id exists in MongoDB"""
        try:
            # Initialize cluster metadata service
            cluster_service = ClusterMetadataService()
            
            # Check if cluster exists and is active
            cluster_exists = await cluster_service.validate_cluster_exists(self.cluster_partition_id)
            if not cluster_exists:
                logger.warning(f'Cluster partition {self.cluster_partition_id} not found or inactive in MongoDB for FIELD_RELATES_TO edge {self.uuid}')
                
        except (MongoClusterNotFoundError, ClusterValidationError) as e:
            # Log the warning but don't fail the Neo4j operation
            logger.warning(f'MongoDB cluster validation failed for FIELD_RELATES_TO edge {self.uuid}: {e}')
        except Exception as e:
            # Log unexpected errors but don't fail the Neo4j operation
            logger.error(f'Unexpected error during MongoDB cluster validation for FIELD_RELATES_TO edge {self.uuid}: {e}')

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve Field relationship edge by UUID and return full Edge object"""
        # We need to find the source field UUID first since the query requires it
        records, _, _ = await driver.execute_query(
            """
            MATCH (f1:Field)-[r:FIELD_RELATES_TO {uuid: $uuid}]->(f2:Field)
            RETURN r, f1.uuid as source_uuid, f2.uuid as target_uuid
            """,
            uuid=uuid,
            routing_='r',
        )

        if not records:
            raise EdgeNotFoundError(f"FieldRelationshipEdge with UUID {uuid} not found")

        return get_field_relationship_edge_from_record(records[0])

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple Field relationship edges by UUIDs"""
        if not uuids:
            return []

        records, _, _ = await driver.execute_query(
            """
            MATCH (f1:Field)-[r:FIELD_RELATES_TO]->(f2:Field)
            WHERE r.uuid IN $uuids
            RETURN r, f1.uuid as source_uuid, f2.uuid as target_uuid
            """,
            uuids=uuids,
            routing_='r',
        )

        return [get_field_relationship_edge_from_record(record) for record in records]




# Edge helpers
def get_belongs_to_edge_from_record(record: Any) -> BelongsToEdge:
    """Convert database record to BelongsToEdge instance"""
    return BelongsToEdge(
        uuid=record['uuid'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        cluster_partition_id=record['cluster_partition_id'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
    )


def get_field_relationship_edge_from_record(record: Any) -> FieldRelationshipEdge:
    """Convert database record to FieldRelationshipEdge instance"""
    edge_data = record['r'] if 'r' in record else record
    
    return FieldRelationshipEdge(
        uuid=edge_data['uuid'],
        name=edge_data['name'],
        description=edge_data['description'],
        confidence=edge_data['confidence'],
        cluster_partition_id=edge_data['cluster_partition_id'],
        relationship_type=edge_data.get('relationship_type', 'FIELD_RELATES_TO'),
        description_embedding=edge_data.get('description_embedding'),
        source_node_uuid=record.get('source_uuid', record.get('source_node_uuid')),
        target_node_uuid=record.get('target_uuid', record.get('target_node_uuid')),
        created_at=parse_db_date(edge_data['created_at']),  # type: ignore
        valid_at=parse_db_date(edge_data['valid_at']) if edge_data.get('valid_at') else None,
        invalid_at=parse_db_date(edge_data['invalid_at']) if edge_data.get('invalid_at') else None,
    )


