"""
Cluster fields graph security nodes refactor for graphiti_core structure.
This module defines the base class for all security nodes in the Graphiti project.
"""



from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphiti_core.driver import GraphDriver

from graphiti_core.utils.utc_helpers import utc_now
from graphiti_core.errors import NodeNotFoundError
import logging


logger = logging.getLogger(__name__)



class Node(BaseModel, ABC):
    """
    Enhanced base class for all nodes in the security field graph.
    Provides temporal validation, relationship management, and type safety.
    """
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='Name of the node')
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())
    validated_at: datetime = Field(default_factory=lambda: utc_now())
    invalidated_at: datetime | None = Field(default=None)
    last_updated: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: 'GraphDriver'):
        """Save node to Neo4j database - must be implemented by each node type"""
        ...

    async def delete(self, driver: 'GraphDriver'):
        """Delete node and all its relationships from Neo4j database"""
        result = await driver.execute_query(
            """
            MATCH (n {uuid: $uuid})
            WHERE n:Field OR n:Cluster
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """,
            uuid=self.uuid,
        )

        deleted_count = result.records[0]['deleted_count'] if result.records else 0
        if deleted_count == 0:
            logger.warning(f'Node with UUID {self.uuid} not found for deletion')
        else:
            logger.debug(f'Deleted Node: {self.uuid}')
        
        return result

    def __hash__(self):
        return hash(self.uuid)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        return False

