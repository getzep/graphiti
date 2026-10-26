# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator

GRAPH_IDENTIFIER = re.compile(r'^[A-Za-z][A-Za-z0-9_$#]{0,127}$')


class Oracle26aiConfig(BaseModel):
    """Immutable Oracle connection and schema configuration."""

    model_config = ConfigDict(frozen=True)

    dsn: str
    user: str
    password: SecretStr
    config_dir: str | None = None
    wallet_location: str | None = None
    wallet_password: SecretStr | None = None
    embedding_dim: int = 1024
    graph_name: str = 'GRAPHITI_GRAPH'
    pool_min: int = 1
    pool_max: int = 8

    @field_validator('embedding_dim')
    @classmethod
    def require_phase_zero_dimension(cls, value: int) -> int:
        if value != 1024:
            raise ValueError('embedding_dim must be 1024 in Phase 0')
        return value

    @field_validator('graph_name')
    @classmethod
    def validate_graph_name(cls, value: str) -> str:
        if not GRAPH_IDENTIFIER.fullmatch(value):
            raise ValueError('graph_name must be a simple Oracle identifier')
        return value.upper()

    @model_validator(mode='after')
    def validate_pool_bounds(self) -> Oracle26aiConfig:
        if self.pool_min < 0:
            raise ValueError('pool_min must be non-negative')
        if self.pool_max < max(1, self.pool_min):
            raise ValueError('pool_max must be at least pool_min and greater than zero')
        return self
