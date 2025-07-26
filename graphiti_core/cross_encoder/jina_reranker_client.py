"""
Jina AI Reranker Client
"""

from __future__ import annotations

import os
import httpx

from .client import CrossEncoderClient

DEFAULT_MODEL = "jina-reranker-m0"
DEFAULT_BASE_URL = "https://api.jina.ai"


class JinaAIRerankerConfig:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.api_key = api_key or os.getenv("JINAAI_API_KEY")
        self.model = model
        self.base_url = base_url


class JinaAIRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: JinaAIRerankerConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if config is None:
            config = JinaAIRerankerConfig()
        self.config = config
        if client is None:
            self.client = httpx.AsyncClient(base_url=self.config.base_url)
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        payload = {
            "model": self.config.model,
            "query": query,
            "top_n": len(passages),
            "documents": [{"text": p} for p in passages],
            "return_documents": False,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        response = await self.client.post("/v1/rerank", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = [
            (passages[item["index"]], float(item["relevance_score"]))
            for item in data.get("results", [])
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
