#!/usr/bin/env python3
"""
修复 GeminiEmbedderClient 的脚本
"""

import re

# 读取文件
with open('graphiti_mcp_server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义要替换的旧代码
old_code = '''# --- START of FINAL, CORRECT, COMPLETE GeminiEmbedderClient ---
from typing import Dict, List, Union

class GeminiEmbedderClient(EmbedderClient):
    """An embedder client for Google Gemini embedding models."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Google API Key cannot be empty for Embedder.")
        genai.configure(api_key=api_key)
        self.model_name = model
        logger.info(f"GeminiEmbedderClient initialized for model: {self.model_name}")

    @classmethod
    def create(cls, api_key: str, model: str, **_kwargs):
        """
        Factory method to create an instance of the client.
        This is the abstract method required by the EmbedderClient base class.
        """
        return cls(api_key=api_key, model=model)

    async def aembed(self, texts: List[str]) -> List[Dict[str, Union[str, List[float]]]]:
        """
        Generate embeddings for a list of texts using Gemini API.
        """
        if not texts:
            return []

        try:
            result = await genai.embed_content_async(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )

            embeddings_list = result['embedding']

            return [
                {'text': text, 'embedding': vector}
                for text, vector in zip(texts, embeddings_list)
            ]

        except Exception as e:
            logger.error(f"Error during Gemini embedding API call: {e}")
            return [{'text': text, 'embedding': []} for text in texts]

# --- END of FINAL, CORRECT, COMPLETE GeminiEmbedderClient ---'''

# 定义新代码
new_code = '''# --- START of FINAL, CORRECT, COMPLETE GeminiEmbedderClient ---
from typing import Dict, List, Union
from collections.abc import Iterable

class GeminiEmbedderClient(EmbedderClient):
    """An embedder client for Google Gemini embedding models."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("Google API Key cannot be empty for Embedder.")
        genai.configure(api_key=api_key)
        self.model_name = model
        logger.info(f"GeminiEmbedderClient initialized for model: {self.model_name}")

    async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> list[float]:
        """
        Generate embedding for a single input using Gemini API.
        Required by EmbedderClient base class.
        """
        # Handle different input types
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], str):
            text = input_data[0]  # Take first string from list
        else:
            raise ValueError(f"Unsupported input_data type: {type(input_data)}")

        try:
            result = await genai.embed_content_async(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error during Gemini embedding API call: {e}")
            return []

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using Gemini API.
        Required by EmbedderClient base class.
        """
        if not input_data_list:
            return []

        try:
            result = await genai.embed_content_async(
                model=self.model_name,
                content=input_data_list,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error during Gemini embedding batch API call: {e}")
            return [[] for _ in input_data_list]

# --- END of FINAL, CORRECT, COMPLETE GeminiEmbedderClient ---'''

# 替换
new_content = content.replace(old_code, new_code)

# 写回文件
with open('graphiti_mcp_server.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✓ GeminiEmbedderClient 已修复")
