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

from pydantic import BaseModel, ConfigDict

from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient


class GraphitiClients(BaseModel):
    driver: GraphDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient

    model_config = ConfigDict(arbitrary_types_allowed=True)
