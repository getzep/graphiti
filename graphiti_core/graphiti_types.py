from neo4j import AsyncDriver
from pydantic import BaseModel

from graphiti_core.cross_encoder import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient


class GraphitiClients(BaseModel):
    driver: AsyncDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient
