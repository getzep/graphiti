"""
Storage-neutral ingestion trace interfaces for Graphiti.

Graphiti core should not depend on a specific database. Applications can attach an
implementation to GraphitiClients to persist traces from optional trace steps.
"""

from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class IngestionTraceRecord(BaseModel):
    trace_id: str
    workflow_run_id: str | None = None
    stage: str
    prompt_name: str
    account_uuid: str | None = None
    project_uuid: str | None = None
    graph_uuid: str | None = None
    episode_uuids: list[str] = Field(default_factory=list)
    model: str | None = None
    model_size: str | None = None
    validation_status: Literal['recorded', 'failed']
    attempt_count: int
    input_context: dict[str, Any] = Field(default_factory=dict)
    original_output: dict[str, Any]
    decision_explanations: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


@runtime_checkable
class IngestionTraceInterface(Protocol):
    async def write_trace(self, record: IngestionTraceRecord) -> None: ...


class NoOpIngestionTrace:
    async def write_trace(self, record: IngestionTraceRecord) -> None:
        pass
