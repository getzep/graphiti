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

import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import boto3
else:
    try:
        import boto3
    except ImportError:
        raise ImportError(
            'boto3 is required for AmazonBedrockEmbedder. '
            'Install it with: pip install graphiti-core[bedrock]'
        ) from None

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'amazon.titan-embed-text-v2:0'


class AmazonBedrockEmbedderConfig(EmbedderConfig):
    model: str = DEFAULT_EMBEDDING_MODEL
    region: str = 'us-east-1'


class AmazonBedrockEmbedder(EmbedderClient):
    def __init__(self, config: AmazonBedrockEmbedderConfig | None = None):
        self.config = config or AmazonBedrockEmbedderConfig()
        self.client = boto3.client('bedrock-runtime', region_name=self.config.region)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list):
            text = ' '.join(str(item) for item in input_data)
        else:
            text = str(input_data)

        body = json.dumps({'inputText': text})

        try:
            response = self.client.invoke_model(
                modelId=self.config.model,
                body=body,
                accept='application/json',
                contentType='application/json',
            )

            response_body = json.loads(response['body'].read().decode('utf-8'))
            return response_body['embedding']

        except Exception as e:
            logger.error(f'Bedrock embedding failed: {e}')
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input_data_list:
            embedding = await self.create(text)
            embeddings.append(embedding)
        return embeddings
