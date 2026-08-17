"""A small dependency-free reranker for local and OpenAI-compatible setups."""

from __future__ import annotations

import math
import re
import unicodedata

from .client import CrossEncoderClient


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize('NFKC', text).casefold()
    tokens = set(re.findall(r'\w+', normalized, flags=re.UNICODE))
    compact = re.sub(r'\s+', '', normalized)
    tokens.update(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
    return {token for token in tokens if token}


class LexicalRerankerClient(CrossEncoderClient):
    """Rank passages by normalized lexical and character n-gram overlap."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        query_tokens = _tokens(query)
        scored: list[tuple[str, float]] = []
        for passage in passages:
            passage_tokens = _tokens(passage)
            denominator = math.sqrt(max(1, len(query_tokens)) * max(1, len(passage_tokens)))
            score = len(query_tokens & passage_tokens) / denominator
            scored.append((passage, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)
