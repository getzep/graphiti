"""
Jina AI Embedder

Get your Jina AI API key for free: https://jina.ai/?sui=apikey
"""

from collections.abc import Iterable

import httpx
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = "jina-embeddings-v4"
DEFAULT_BASE_URL = "https://api.jina.ai"
DEFAULT_TASK = "text-matching"


class JinaAIEmbedderConfig(EmbedderConfig):
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    api_key: str | None = None
    base_url: str = Field(default=DEFAULT_BASE_URL)
    task: str = Field(default=DEFAULT_TASK)


class JinaAIEmbedder(EmbedderClient):
    def __init__(
        self,
        config: JinaAIEmbedderConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if config is None:
            config = JinaAIEmbedderConfig()
        self.config = config
        if client is None:
            self.client = httpx.AsyncClient(base_url=self.config.base_url, headers={"Accept": "application/json"})
        else:
            self.client = client

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            items = [{"text": input_data}]
        elif isinstance(input_data, list) and all(isinstance(i, str) for i in input_data):
            items = [{"text": text} for text in input_data]
        else:
            items = [{"text": str(input_data)}]

        payload = {
            "model": self.config.embedding_model,
            "task": self.config.task,
            "dimensions": self.config.embedding_dim,
            "input": items,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        response = await self.client.post("/v1/embeddings", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        embedding = data["data"][0]["embedding"]
        return [float(x) for x in embedding[: self.config.embedding_dim]]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        items = [{"text": text} for text in input_data_list]
        payload = {
            "model": self.config.embedding_model,
            "task": self.config.task,
            "dimensions": self.config.embedding_dim,
            "input": items,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        response = await self.client.post("/v1/embeddings", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [
            [float(x) for x in item["embedding"][: self.config.embedding_dim]]
            for item in data["data"]
        ]
