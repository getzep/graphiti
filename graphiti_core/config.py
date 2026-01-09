"""
Configuration classes for Graphiti.
"""

from pydantic import BaseModel, Field


class DeduplicationConfig(BaseModel):
    """Episode deduplication configuration."""

    enabled: bool = Field(default=True, description='Enable/disable episode deduplication')
    strategy: str = Field(
        default='exact',
        description="Deduplication strategy: 'exact', 'similarity', or 'hybrid'",
    )
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description='Minimum similarity score for similarity-based deduplication',
    )
    check_by_uuid_first: bool = Field(
        default=True,
        description='If uuid is provided, check existing episode before processing',
    )
