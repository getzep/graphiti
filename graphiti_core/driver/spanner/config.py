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

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphiti_core.driver.spanner.schema import validate_embedding_dim, validate_graph_name
from graphiti_core.embedder.client import EMBEDDING_DIM


class SpannerConfig(BaseModel):
    """Connection and schema settings for :class:`SpannerDriver`.

    ``project`` falls back to ``SPANNER_PROJECT`` and then ``GOOGLE_CLOUD_PROJECT``;
    ``emulator_host`` falls back to ``SPANNER_EMULATOR_HOST``. ``embedding_dim`` sets
    ``vector_length`` on the embedding columns; ``None`` leaves them unbounded (exact KNN
    still works, ANN vector indexes do not).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project: str | None = None
    instance: str
    database: str
    graph_name: str = 'graphiti'
    embedding_dim: int | None = EMBEDDING_DIM
    emulator_host: str | None = None
    credentials: Any | None = None
    max_workers: int = Field(default=16, ge=1)
    mutation_chunk_rows: int = Field(default=500, ge=1)
    ddl_timeout_seconds: float = Field(default=900.0, gt=0)

    @model_validator(mode='after')
    def _apply_defaults(self) -> SpannerConfig:
        if self.project is None:
            self.project = os.getenv('SPANNER_PROJECT') or os.getenv('GOOGLE_CLOUD_PROJECT')
        if self.emulator_host is None:
            self.emulator_host = os.getenv('SPANNER_EMULATOR_HOST') or None
        validate_graph_name(self.graph_name)
        validate_embedding_dim(self.embedding_dim)
        return self
