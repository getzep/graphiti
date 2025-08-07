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
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.nodes.field_db_queries import (
    FIELD_NODE_SAVE,
    FIELD_NODE_GET_BY_UUID,
    FIELD_NODE_GET_BY_UUIDS,
    FIELD_NODE_UPDATE,
    FIELD_NODE_DELETE,
    CLUSTER_NODE_SAVE,
    CLUSTER_NODE_GET_BY_UUID,
    CLUSTER_NODE_GET_BY_UUIDS,
    CLUSTER_NODE_UPDATE,
    CLUSTER_NODE_DELETE,
)
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.cluster_metadata.cluster_service import ClusterMetadataService
from graphiti_core.cluster_metadata.cluster_fields_service import ClusterFieldsService
from graphiti_core.cluster_metadata.exceptions import (
    ClusterNotFoundError as MongoClusterNotFoundError,
    ClusterValidationError,
    DuplicateClusterError,
    InvalidOrganizationError,
    FieldAlreadyExistsError,
    FieldValidationError,
    FieldNotFoundError as MongoFieldNotFoundError
)
from graphiti_core.cluster_metadata.models import ClusterCreateRequest, FieldCreateRequest

logger = logging.getLogger(__name__)


class Node(BaseModel, ABC):
    """
    Abstract base class for all node types in the security field graph.
    Enforces consistent implementation across all node schemas.
    Provides type safety through abstract methods that must be implemented by subclasses.
    """
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())
    validated_at: datetime = Field(default_factory=lambda: utc_now())             # Validation timestamp (default: utc_now())
    invalidated_at: datetime | None = Field(default=None)                         # Invalidation timestamp (default: None)
    last_updated: datetime = Field(default_factory=lambda: utc_now())             # Last update timestamp (default: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver):
        """Save node to Neo4j database - must be implemented by each node type"""
        ...

    async def delete(self, driver: GraphDriver):
        """Delete node and all its relationships from Neo4j database"""
        result = await driver.execute_query(
            """
            MATCH (n:Field|Cluster {uuid: $uuid})
            DETACH DELETE n
            """,
            uuid=self.uuid,
        )

        logger.debug(f'Deleted Node: {self.uuid}')
        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

    @classmethod
    @abstractmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Get node by UUID - must be implemented by child classes for type safety"""
        ...

    @classmethod
    @abstractmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Get multiple nodes by UUIDs - must be implemented by child classes"""
        ...

    def update_temporal_fields(self):
        """Update last_updated timestamp and optionally validated_at"""
        self.last_updated = utc_now()
        self.validated_at = utc_now()


class FieldNode(Node):
    """
    Specialized node for security audit fields.
    Relationships are used for clustering and hierarchy.
    Inherits temporal validation fields from Node base class.
    """

    # ==================== FIELD-SPECIFIC PROPERTIES ====================

    # Core Field Properties
    description: str                   # Field description for AI context and embedding
    examples: list[str] = Field(default_factory=list)  # Example values for validation
    data_type: str                     # Field data type (string, integer, etc.)
    count: int = 0                     # Field occurrences in events
    distinct_count: int = 0            # Distinct value count in events

    # Cluster Tracking (Relationship-based)
    primary_cluster_id: str          # Primary cluster UUID

    # Semantic Search Support
    embedding: list[float] | None = None  # Vector for semantic search on description

    def __init__(self, **data):
        # Set default label for Field nodes
        if 'labels' not in data:
            data['labels'] = ['Field']
        super().__init__(**data)

    # ==================== FIELD-SPECIFIC METHODS ====================

    async def generate_embedding(self, embedder: EmbedderClient):
        """Generate embedding for field description using embedder client"""
        start = time()
        text = self.description.replace('\n', ' ')
        self.embedding = await embedder.create(input_data=[text])
        end = time()
        logger.debug(f'embedded {text} in {end - start} ms')
        return self.embedding

    async def load_embedding(self, driver: GraphDriver):
        """Load embedding from database"""
        records, _, _ = await driver.execute_query(
            """
            MATCH (f:Field {uuid: $uuid})
            RETURN f.embedding AS embedding
            """,
            uuid=self.uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(self.uuid)

        self.embedding = records[0]['embedding']

    async def save(self, driver: GraphDriver):
        """Save Field node to Neo4j database with MongoDB synchronization"""
        # Update temporal fields before saving
        self.update_temporal_fields()

        result = await driver.execute_query(
            FIELD_NODE_SAVE,
            uuid=self.uuid,
            name=self.name,
            description=self.description,
            examples=self.examples,
            data_type=self.data_type,
            count=self.count,
            distinct_count=self.distinct_count,
            primary_cluster_id=self.primary_cluster_id,
            embedding=self.embedding,
            validated_at=self.validated_at,
            invalidated_at=self.invalidated_at,
            last_updated=self.last_updated,
            created_at=self.created_at,
        )

        # Sync field count with MongoDB cluster metadata
        await self._sync_with_mongodb()

        logger.debug(f'Saved Field Node to Graph: {self.uuid}')
        return result

    async def update(self, driver: GraphDriver):
        """Update Field node in Neo4j database"""
        # Update temporal fields before updating
        self.update_temporal_fields()

        result = await driver.execute_query(
            FIELD_NODE_UPDATE,
            uuid=self.uuid,
            name=self.name,
            description=self.description,
            examples=self.examples,
            data_type=self.data_type,
            count=self.count,
            distinct_count=self.distinct_count,
            embedding=self.embedding,
            validated_at=self.validated_at,
            last_updated=self.last_updated,
        )

        logger.debug(f'Updated Field Node: {self.uuid}')
        return result

    async def _sync_with_mongodb(self):
        """Sync field data with MongoDB cluster metadata"""
        try:
            # Initialize cluster metadata service
            cluster_service = ClusterMetadataService()
            
            # Validate that cluster exists and increment field count
            cluster_exists = await cluster_service.validate_cluster_exists(self.primary_cluster_id)
            if cluster_exists:
                await cluster_service.increment_field_count(self.primary_cluster_id)
                logger.debug(f'Synchronized field {self.uuid} with MongoDB cluster {self.primary_cluster_id}')
            else:
                logger.warning(f'Cluster {self.primary_cluster_id} not found in MongoDB for field {self.uuid}')
                
        except (MongoClusterNotFoundError, ClusterValidationError) as e:
            # Log the error but don't fail the Neo4j operation
            logger.warning(f'MongoDB sync failed for field {self.uuid}: {e}')
        except Exception as e:
            # Log unexpected errors but don't fail the Neo4j operation
            logger.error(f'Unexpected error during MongoDB sync for field {self.uuid}: {e}')

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve Field node by UUID"""
        records, _, _ = await driver.execute_query(
            FIELD_NODE_GET_BY_UUID,
            uuid=uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(uuid)

        return get_field_node_from_record(records[0])

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple Field nodes by UUIDs"""
        if not uuids:
            return []

        records, _, _ = await driver.execute_query(
            FIELD_NODE_GET_BY_UUIDS,
            uuids=uuids,
            routing_='r',
        )

        return [get_field_node_from_record(record) for record in records]




class ClusterNode(Node):
    """
    Primary cluster node for organizational isolation and macro-level grouping.
    Represents organizational/macro clusters like 'linux_audit_batelco'.
    Inherits temporal validation fields from Node base class.
    """

    # ==================== CLUSTER-SPECIFIC PROPERTIES ====================

    # Core Cluster Properties
    organization: str                  # Organization identifier (e.g., 'batelco', 'sico')
    macro_name: str                    # Refers to the macro name with this organization

    def __init__(self, **data):
        # Set default label for Cluster nodes
        if 'labels' not in data:
            data['labels'] = ['Cluster']
        super().__init__(**data)

    async def save(self, driver: GraphDriver):
        """Save Cluster node to Neo4j database with MongoDB synchronization"""
        # Update temporal fields before saving
        self.update_temporal_fields()

        result = await driver.execute_query(
            CLUSTER_NODE_SAVE,
            uuid=self.uuid,
            name=self.name,
            organization=self.organization,
            macro_name=self.macro_name,
            validated_at=self.validated_at,
            invalidated_at=self.invalidated_at,
            last_updated=self.last_updated,
            created_at=self.created_at,
        )

        # Sync cluster with MongoDB
        await self._sync_with_mongodb()

        logger.debug(f'Saved Cluster Node to Graph: {self.uuid}')
        return result

    async def update(self, driver: GraphDriver):
        """Update Cluster node in Neo4j database"""
        # Update temporal fields before updating
        self.update_temporal_fields()

        result = await driver.execute_query(
            CLUSTER_NODE_UPDATE,
            uuid=self.uuid,
            name=self.name,
            organization=self.organization,
            macro_name=self.macro_name,
            validated_at=self.validated_at,
            last_updated=self.last_updated,
        )

        logger.debug(f'Updated Cluster Node: {self.uuid}')
        return result

    async def _sync_with_mongodb(self):
        """Sync cluster data with MongoDB cluster metadata"""
        try:
            # Initialize cluster metadata service
            cluster_service = ClusterMetadataService()
            
            # Check if cluster already exists in MongoDB
            existing_cluster = await cluster_service.get_cluster(self.name)
            
            if not existing_cluster:
                # Create new cluster in MongoDB
                cluster_request = ClusterCreateRequest(
                    cluster_id=self.name,
                    cluster_uuid=self.uuid,
                    macro_name=self.macro_name,
                    organization=self.organization,
                    description=f"Cluster for {self.macro_name} in {self.organization}",
                    status="active",
                    total_fields=0,
                    created_by="neo4j_sync"
                )
                
                await cluster_service.create_cluster(cluster_request)

            
                logger.debug(f'Created cluster in MongoDB: {self.name}')
            else:
                logger.debug(f'Cluster {self.name} already exists in MongoDB')
                
        except (DuplicateClusterError, InvalidOrganizationError, ClusterValidationError) as e:
            # Log the error but don't fail the Neo4j operation
            logger.warning(f'MongoDB sync failed for cluster {self.uuid}: {e}')
        except Exception as e:
            # Log unexpected errors but don't fail the Neo4j operation
            logger.error(f'Unexpected error during MongoDB sync for cluster {self.uuid}: {e}')

    @classmethod
    async def get_by_uuid(cls, driver: GraphDriver, uuid: str):
        """Retrieve Cluster node by UUID"""
        records, _, _ = await driver.execute_query(
            CLUSTER_NODE_GET_BY_UUID,
            uuid=uuid,
            routing_='r',
        )

        if len(records) == 0:
            raise NodeNotFoundError(uuid)

        return get_cluster_node_from_record(records[0])

    @classmethod
    async def get_by_uuids(cls, driver: GraphDriver, uuids: list[str]):
        """Retrieve multiple Cluster nodes by UUIDs"""
        if not uuids:
            return []

        records, _, _ = await driver.execute_query(
            CLUSTER_NODE_GET_BY_UUIDS,
            uuids=uuids,
            routing_='r',
        )

        return [get_cluster_node_from_record(record) for record in records]




# Node helpers
def get_field_node_from_record(record: Any) -> FieldNode:
    """Convert database record to FieldNode instance"""
    return FieldNode(
        uuid=record['uuid'],
        name=record['name'],
        description=record['description'],
        examples=record['examples'],
        data_type=record['data_type'],
        count=record['count'],
        distinct_count=record['distinct_count'],
        primary_cluster_id=record['primary_cluster_id'],
        embedding=record['embedding'],
        labels=record.get('labels', ['Field']),
        created_at=parse_db_date(record['created_at']),  # type: ignore
        validated_at=parse_db_date(record['validated_at']),  # type: ignore
        invalidated_at=parse_db_date(record['invalidated_at']) if record['invalidated_at'] else None,
        last_updated=parse_db_date(record['last_updated']),  # type: ignore
    )


def get_cluster_node_from_record(record: Any) -> ClusterNode:
    """Convert database record to ClusterNode instance"""
    return ClusterNode(
        uuid=record['uuid'],
        name=record['name'],
        organization=record['organization'],
        macro_name=record['macro_name'],
        labels=['Cluster'],
        created_at=parse_db_date(record['created_at']),  # type: ignore
        validated_at=parse_db_date(record['validated_at']),  # type: ignore
        invalidated_at=parse_db_date(record['invalidated_at']) if record['invalidated_at'] else None,
        last_updated=parse_db_date(record['last_updated']),  # type: ignore
    )


