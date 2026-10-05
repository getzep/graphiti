"""
Configuration classes for Graphiti.
"""

from pydantic import BaseModel, Field


class DeduplicationConfig(BaseModel):
    """Episode deduplication configuration.

    The single mechanism is a fulltext near-match confirmed by exact
    content+name equality. The former strategy/threshold fields described
    behaviors that were never implemented and were removed.
    """

    enabled: bool = Field(default=False, description='Enable episode deduplication')
