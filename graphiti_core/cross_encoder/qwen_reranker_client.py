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

import asyncio
import logging
import math
from dataclasses import dataclass

import httpx

from .client import CrossEncoderClient

logger = logging.getLogger(__name__)


@dataclass
class QwenRerankerConfig:
    """Qwen Reranker 配置"""

    base_url: str = 'http://localhost:7001'
    model: str = 'qwen3-reranker-0.6b'
    api_key: str | None = None
    timeout: float = 30.0
    max_concurrent: int = 10


class QwenRerankerClient(CrossEncoderClient):
    """
    Qwen3 Reranker 客户端

    使用 Qwen3-Reranker 模型对文档进行相关性排序。
    通过 vLLM chat completions API 调用，使用 logprobs 获取 "yes" token 的概率作为相关性分数。

    需要 vLLM 部署的 Qwen3-Reranker 模型。
    """

    # Qwen3-Reranker 的系统 prompt
    SYSTEM_PROMPT = 'Judge whether the Document is relevant to the Query. Note that the answer can only be "yes" or "no".'

    def __init__(self, config: QwenRerankerConfig | None = None):
        self.config = config or QwenRerankerConfig()
        headers = {}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        self._client = httpx.AsyncClient(timeout=self.config.timeout, headers=headers)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def _score_single(self, query: str, document: str) -> float:
        """计算单个文档的相关性分数"""

        # 构造 Qwen3-Reranker 格式的 user message
        user_content = f'<Query>{query}</Query>\n<Document>{document}</Document>'

        messages = [
            {'role': 'system', 'content': self.SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
        ]

        async with self._semaphore:
            try:
                response = await self._client.post(
                    f'{self.config.base_url}/v1/chat/completions',
                    json={
                        'model': self.config.model,
                        'messages': messages,
                        'max_tokens': 1,
                        'temperature': 0,
                        'logprobs': True,
                        'top_logprobs': 5,
                        # 禁用 Qwen3 的思考模式，强制直接输出 yes/no
                        'chat_template_kwargs': {'enable_thinking': False},
                    },
                )
                response.raise_for_status()
                data = response.json()

                # 从 logprobs 中提取分数
                choice = data['choices'][0]
                logprobs = choice.get('logprobs')

                if logprobs and logprobs.get('content'):
                    top_logprobs = logprobs['content'][0].get('top_logprobs', [])

                    # 查找 "yes" 和 "no" 的概率
                    yes_logprob = None
                    no_logprob = None

                    # top_logprobs 按概率降序排列，只取第一个匹配
                    # 避免 "yes", "Yes", "YES" 都匹配时被覆盖
                    for item in top_logprobs:
                        token = item['token'].lower().strip()
                        if token == 'yes' and yes_logprob is None:
                            yes_logprob = item['logprob']
                        elif token == 'no' and no_logprob is None:
                            no_logprob = item['logprob']

                    # 计算 yes 的概率
                    if yes_logprob is not None:
                        if no_logprob is not None:
                            # 使用 softmax 计算相对概率
                            yes_prob = math.exp(yes_logprob)
                            no_prob = math.exp(no_logprob)
                            score = yes_prob / (yes_prob + no_prob)
                        else:
                            score = math.exp(yes_logprob)
                        return min(max(score, 0.0), 1.0)
                    elif no_logprob is not None:
                        # 只有 no，返回 1 - no_prob
                        return min(max(1 - math.exp(no_logprob), 0.0), 1.0)

                # 回退：检查生成的文本
                content = choice.get('message', {}).get('content', '').lower().strip()
                if 'yes' in content:
                    return 0.7  # 默认高分
                elif 'no' in content:
                    return 0.3  # 默认低分

                return 0.5  # 无法判断时返回中间值

            except Exception as e:
                logger.error(f'Reranker scoring failed: {e}')
                return 0.5  # 出错时返回中间值

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """
        对 passages 进行相关性排序

        Args:
            query: 查询文本
            passages: 待排序的文档列表

        Returns:
            按相关性降序排列的 (passage, score) 列表
        """
        if not passages:
            return []

        # 并发计算所有文档的分数
        tasks = [self._score_single(query, passage) for passage in passages]
        scores = await asyncio.gather(*tasks)

        # 组合结果并排序
        results = list(zip(passages, scores, strict=True))
        results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f'Reranked {len(passages)} passages for query: {query[:50]}...')

        return results

    async def close(self):
        """关闭 HTTP 客户端"""
        await self._client.aclose()
