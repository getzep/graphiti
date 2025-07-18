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

import logging
from typing import List, Tuple
import os
import asyncio

import httpx

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.llm_client import RateLimitError

logger = logging.getLogger(__name__)

# SiliconFlow支持的重排序模型 (根据官方文档更新)
SUPPORTED_MODELS = [
    "Qwen/Qwen3-Reranker-8B",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-0.6B",
    "BAAI/bge-reranker-v2-m3",
    "Pro/BAAI/bge-reranker-v2-m3",
    "netease-youdao/bce-reranker-base_v1"
]

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-8B"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
MAX_OVERLAP_TOKENS = 80  # 官方文档指定的最大值


class SiliconFlowRerankerClient(CrossEncoderClient):
    """
    功能描述：SiliconFlow平台的重排序客户端，直接调用/v1/rerank端点

    该客户端实现了Graphiti的CrossEncoderClient接口，使用SiliconFlow平台提供的
    专用重排序服务来对文本段落进行相关性排序。完全符合官方API规范。

    属性说明：
        api_key (str): SiliconFlow API密钥
        base_url (str): SiliconFlow API基础URL
        model (str): 重排序模型名称
        timeout (int): 请求超时时间（秒）
        return_documents (bool): 是否在响应中返回文档文本
        max_chunks_per_doc (int): 每个文档最大分块数（仅特定模型支持）
        overlap_tokens (int): 重叠token数，最大80（仅特定模型支持）
        client (httpx.AsyncClient): HTTP异步客户端

    注意事项：
        - 严格按照SiliconFlow官方API规范实现
        - 支持最新的Qwen重排序模型系列
        - 包含完善的错误处理和重试逻辑
        - 兼容Graphiti项目的错误处理模式
        - max_chunks_per_doc和overlap_tokens仅在特定模型下生效

    示例：
        >>> client = SiliconFlowRerankerClient(
        ...     api_key="sk-siliconflow-xxxxx",
        ...     model="Qwen/Qwen3-Reranker-8B"
        ... )
        >>> async with client:
        ...     results = await client.rank("苹果手机", ["iPhone 15", "安卓手机", "苹果电脑"])
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        return_documents: bool = True,  # 默认返回文档文本，便于Graphiti使用
        max_chunks_per_doc: int = None,
        overlap_tokens: int = None,
    ):
        """
        功能描述：初始化SiliconFlow重排序客户端

        参数说明：
            api_key (str): SiliconFlow API密钥，如果为None则从环境变量SILICONFLOW_API_KEY获取
            base_url (str): API基础URL，默认为SiliconFlow官方端点
            model (str): 重排序模型名称，默认为Qwen/Qwen3-Reranker-8B
            timeout (int): 请求超时时间（秒），默认60秒
            return_documents (bool): 是否返回文档文本，默认True
            max_chunks_per_doc (int): 每个文档的最大分块数，仅特定模型支持
            overlap_tokens (int): 分块间重叠的token数，最大80，仅特定模型支持

        注意事项：
            - API密钥优先从参数获取，其次从环境变量SILICONFLOW_API_KEY获取
            - 模型必须在官方支持列表中
            - max_chunks_per_doc和overlap_tokens仅在以下模型中有效：
              BAAI/bge-reranker-v2-m3, Pro/BAAI/bge-reranker-v2-m3, netease-youdao/bce-reranker-base_v1
            - overlap_tokens最大值为80（官方限制）

        示例：
            >>> # 从环境变量获取API密钥
            >>> client = SiliconFlowRerankerClient()
            >>>
            >>> # 使用新的Qwen模型
            >>> client = SiliconFlowRerankerClient(
            ...     api_key="sk-siliconflow-xxxxx",
            ...     model="Qwen/Qwen3-Reranker-8B"
            ... )
        """
        # API密钥处理：优先使用参数，其次环境变量
        if api_key is None:
            api_key = os.getenv("SILICONFLOW_API_KEY")

        if not api_key:
            raise ValueError(
                "SiliconFlow API密钥未提供。请通过api_key参数提供，"
                "或设置环境变量SILICONFLOW_API_KEY"
            )

        # 验证模型是否在官方支持列表中
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"模型 '{model}' 不在官方支持列表中。"
                f"支持的模型: {SUPPORTED_MODELS}"
            )

        # 验证overlap_tokens限制
        if overlap_tokens is not None and overlap_tokens > MAX_OVERLAP_TOKENS:
            raise ValueError(
                f"overlap_tokens不能超过{MAX_OVERLAP_TOKENS}，当前值: {overlap_tokens}"
            )

        # 检查分块参数是否适用于当前模型
        chunking_supported_models = [
            "BAAI/bge-reranker-v2-m3",
            "Pro/BAAI/bge-reranker-v2-m3",
            "netease-youdao/bce-reranker-base_v1"
        ]

        if (max_chunks_per_doc is not None or overlap_tokens is not None) and \
           model not in chunking_supported_models:
            logger.warning(
                f"模型 '{model}' 不支持max_chunks_per_doc和overlap_tokens参数。"
                f"这些参数仅在以下模型中有效: {chunking_supported_models}"
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")  # 移除末尾斜杠
        self.model = model
        self.timeout = timeout
        self.return_documents = return_documents
        self.max_chunks_per_doc = max_chunks_per_doc
        self.overlap_tokens = overlap_tokens

        # 创建异步HTTP客户端
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Graphiti-SiliconFlow-Reranker/1.0"
            }
        )

    async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """
        功能描述：对文本段落进行重排序

        参数说明：
            query (str): 查询文本
            passages (List[str]): 待排序的文本段落列表

        返回值说明：
            List[Tuple[str, float]]: 排序后的(文本, 相关性分数)列表，按分数降序排列

        注意事项：
            - 严格按照SiliconFlow官方API规范实现
            - 分数为相关性分数，范围和含义由模型决定
            - 返回结果按相关性分数降序排列
            - 空输入会返回空列表
            - 包含重试机制，网络错误时会自动重试
            - 如果API调用最终失败，返回原始顺序且分数为0

        示例：
            >>> results = await client.rank("苹果手机", ["iPhone 15", "安卓手机", "苹果电脑"])
            >>> print(results)
            [("iPhone 15", 0.95), ("苹果电脑", 0.72), ("安卓手机", 0.23)]
        """
        # 边界情况处理
        if not passages:
            return []

        if not query or not query.strip():
            logger.warning("查询文本为空，返回原始顺序")
            return [(passage, 0.0) for passage in passages]

        if len(passages) == 1:
            return [(passages[0], 1.0)]

        # 执行重排序请求，包含重试机制
        retry_count = 0
        last_error = None

        while retry_count <= MAX_RETRIES:
            try:
                # 构造请求数据，根据模型类型使用不同参数策略
                request_data = {
                    "model": self.model,
                    "query": query.strip(),
                    "documents": passages,
                    "return_documents": self.return_documents
                }

                # 根据模型系列调整参数策略
                if self.model.startswith("Qwen/"):
                    # Qwen模型系列：使用最简参数，避免兼容性问题
                    logger.debug(f"使用Qwen模型 {self.model}，采用简化参数策略")
                    # 不添加top_n参数，让API返回默认数量
                else:
                    # 其他模型：可以使用完整参数
                    if len(passages) < 100:  # 如果文档不多，返回全部排序结果
                        request_data["top_n"] = len(passages)

                # 仅在支持的模型中添加分块参数
                chunking_supported_models = [
                    "BAAI/bge-reranker-v2-m3",
                    "Pro/BAAI/bge-reranker-v2-m3",
                    "netease-youdao/bce-reranker-base_v1"
                ]

                if self.model in chunking_supported_models:
                    if self.max_chunks_per_doc is not None:
                        request_data["max_chunks_per_doc"] = self.max_chunks_per_doc
                    if self.overlap_tokens is not None:
                        request_data["overlap_tokens"] = self.overlap_tokens

                # 调用SiliconFlow rerank API
                response = await self.client.post(
                    f"{self.base_url}/rerank",
                    json=request_data
                )

                # 检查HTTP状态码并按照官方错误码处理
                if response.status_code == 429:
                    # 速率限制错误，不重试
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("message", "Rate limit exceeded")
                    except:
                        error_msg = "Rate limit exceeded"
                    raise RateLimitError(f"SiliconFlow API速率限制: {error_msg}")

                if response.status_code == 401:
                    # 认证错误，不重试
                    try:
                        error_msg = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                    except:
                        error_msg = "Unauthorized"
                    raise Exception(f"API认证失败: {error_msg}")

                if response.status_code == 400:
                    # 请求参数错误，不重试
                    try:
                        error_data = response.json()
                        error_msg = f"Code: {error_data.get('code', 'N/A')}, Message: {error_data.get('message', 'Bad Request')}"
                    except:
                        error_msg = response.text
                    raise Exception(f"请求参数错误: {error_msg}")

                if response.status_code == 503:
                    # 服务过载，可以重试
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("message", "Service overloaded")
                    except:
                        error_msg = "Service overloaded"
                    raise Exception(f"服务过载: {error_msg}")

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(
                        f"SiliconFlow API错误 (状态码: {response.status_code}): {error_text}"
                    )
                    raise Exception(f"API请求失败 (状态码: {response.status_code}): {error_text}")

                # 解析JSON响应
                try:
                    result = response.json()
                except Exception as e:
                    raise Exception(f"无法解析API响应为JSON: {e}") from e

                # 按照官方响应格式解析结果
                if "results" not in result:
                    logger.error(f"API响应格式异常，缺少'results'字段: {result}")
                    raise Exception("API响应格式异常")

                ranked_results = []

                for item in result.get("results", []):
                    try:
                        # 按照官方格式解析文档和分数
                        text = ""
                        if "document" in item and isinstance(item["document"], dict):
                            text = item["document"].get("text", "")

                        # 如果没有文档文本，使用索引获取原始文档
                        if not text and "index" in item:
                            doc_index = item["index"]
                            if 0 <= doc_index < len(passages):
                                text = passages[doc_index]
                            else:
                                logger.warning(f"文档索引 {doc_index} 超出范围，跳过此结果")
                                continue

                        # 获取相关性分数
                        score = item.get("relevance_score", 0.0)

                        # 确保分数是数值类型
                        if not isinstance(score, (int, float)):
                            logger.warning(f"相关性分数类型异常: {type(score)}, 使用默认值0.0")
                            score = 0.0

                        if text:  # 只有当文本不为空时才添加结果
                            ranked_results.append((text, float(score)))

                    except Exception as e:
                        logger.warning(f"解析单个结果时出错: {e}, 跳过此结果")
                        continue

                # 确保结果按分数降序排列
                ranked_results.sort(key=lambda x: x[1], reverse=True)

                # 确保所有输入文档都在结果中（处理API可能只返回部分结果的情况）
                returned_texts = {text for text, _ in ranked_results}
                missing_passages = [p for p in passages if p not in returned_texts]

                if missing_passages:
                    logger.debug(f"添加 {len(missing_passages)} 个未返回的文档，分数设为0.0")
                    for missing_passage in missing_passages:
                        ranked_results.append((missing_passage, 0.0))

                    # 重新排序，确保0分的文档在最后
                    ranked_results.sort(key=lambda x: x[1], reverse=True)

                # 验证返回结果的完整性
                if len(ranked_results) == 0:
                    logger.warning("API返回了空结果，使用原始顺序")
                    return [(passage, 0.0) for passage in passages]

                # 记录tokens使用情况（如果有的话）
                if "tokens" in result:
                    tokens_info = result["tokens"]
                    logger.debug(
                        f"Token使用情况 - 输入: {tokens_info.get('input_tokens', 0)}, "
                        f"输出: {tokens_info.get('output_tokens', 0)}"
                    )

                logger.info(
                    f"重排序完成，处理了 {len(passages)} 个文档，"
                    f"返回了 {len(ranked_results)} 个结果"
                )

                return ranked_results

            except RateLimitError:
                # 速率限制错误不重试
                logger.error("API速率限制，停止重试")
                raise
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"请求超时 (尝试 {retry_count + 1}/{MAX_RETRIES + 1}): {e}")
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(f"HTTP错误 (尝试 {retry_count + 1}/{MAX_RETRIES + 1}): {e}")
            except Exception as e:
                last_error = e

                # 对于某些不应重试的错误，直接抛出
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["认证", "auth", "unauthorized", "参数错误", "bad request"]):
                    logger.error(f"不可重试的错误，停止重试: {e}")
                    raise

                logger.warning(f"重排序出现错误 (尝试 {retry_count + 1}/{MAX_RETRIES + 1}): {e}")

            retry_count += 1

            # 如果需要重试，等待一段时间
            if retry_count <= MAX_RETRIES:
                wait_time = min(2 ** retry_count, 30)  # 指数退避，最大30秒
                logger.info(f"等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)

        # 所有重试都失败了，记录错误并返回原始顺序
        logger.error(
            f"重排序最终失败，已重试 {MAX_RETRIES} 次。最后的错误: {last_error}"
        )
        return [(passage, 0.0) for passage in passages]

    async def close(self):
        """
        功能描述：关闭HTTP客户端连接

        注意事项：
            - 在应用程序结束时调用，释放资源
            - 建议在使用完毕后始终调用此方法
            - 在异步上下文管理器中会自动调用

        示例：
            >>> await client.close()
        """
        if self.client:
            await self.client.aclose()

    async def __aenter__(self):
        """
        功能描述：异步上下文管理器入口

        返回值说明：
            SiliconFlowRerankerClient: 当前实例

        示例：
            >>> async with SiliconFlowRerankerClient(api_key="...") as client:
            ...     results = await client.rank("query", ["doc1", "doc2"])
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        功能描述：异步上下文管理器出口

        参数说明：
            exc_type: 异常类型
            exc_val: 异常值
            exc_tb: 异常跟踪信息

        注意事项：
            - 自动关闭HTTP客户端连接
            - 即使出现异常也会正常清理资源
        """
        await self.close()

    def __repr__(self) -> str:
        """
        功能描述：返回对象的字符串表示

        返回值说明：
            str: 包含关键配置信息的字符串表示
        """
        return (
            f"SiliconFlowRerankerClient("
            f"model='{self.model}', "
            f"base_url='{self.base_url}', "
            f"timeout={self.timeout}, "
            f"return_documents={self.return_documents})"
        )