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
from typing import TYPE_CHECKING, Literal

from .client import CrossEncoderClient

if TYPE_CHECKING:
    import boto3
else:
    try:
        import boto3
    except ImportError:
        raise ImportError(
            'boto3 is required for AmazonBedrockRerankerClient. '
            'Install it with: pip install graphiti-core[bedrock]'
        ) from None

logger = logging.getLogger(__name__)

BedrockRerankModel = Literal['cohere.rerank-v3-5:0', 'amazon.rerank-v1:0']

DEFAULT_MODEL: BedrockRerankModel = 'cohere.rerank-v3-5:0'

# Model support by region as of December 2025
MODEL_REGIONS = {
    'amazon.rerank-v1:0': ['ap-northeast-1', 'ca-central-1', 'eu-central-1', 'us-west-2'],
    'cohere.rerank-v3-5:0': [
        'ap-northeast-1',
        'ca-central-1',
        'eu-central-1',
        'us-east-1',
        'us-west-2',
    ],
}


class AmazonBedrockRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        model: BedrockRerankModel = DEFAULT_MODEL,
        region: str = 'us-east-1',
        max_results: int = 100,
    ):
        # Validate region supports the model
        if region not in MODEL_REGIONS[model]:
            supported_regions = ', '.join(MODEL_REGIONS[model])
            raise ValueError(
                f'Model {model} is not supported in region {region}. Supported regions: {supported_regions}'
            )

        self.model = model
        self.region = region
        self.max_results = max_results
        self.client = boto3.client('bedrock-agent-runtime', region_name=region)

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []

        sources = [
            {
                'type': 'INLINE',
                'inlineDocumentSource': {
                    'type': 'TEXT',
                    'textDocument': {
                        'text': passage,
                    },
                },
            }
            for passage in passages
        ]

        model_arn = f'arn:aws:bedrock:{self.region}::foundation-model/{self.model}'

        try:
            # Use executor to run sync boto3 call in async context
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.rerank(
                    queries=[{'type': 'TEXT', 'textQuery': {'text': query}}],
                    sources=sources,
                    rerankingConfiguration={
                        'type': 'BEDROCK_RERANKING_MODEL',
                        'bedrockRerankingConfiguration': {
                            'numberOfResults': min(self.max_results, len(passages)),
                            'modelConfiguration': {
                                'modelArn': model_arn,
                            },
                        },
                    },
                ),
            )

            # Extract results and map back to original passages
            results = []
            for result in response.get('results', []):
                index = result['index']
                relevance_score = result['relevanceScore']
                passage = passages[index]
                results.append((passage, relevance_score))

            return results

        except Exception as e:
            logger.error(f'Error in Bedrock reranking: {e}')
            raise
