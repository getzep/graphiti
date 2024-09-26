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
import typing
from time import time

from graphiti_core.llm_client.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


async def generate_embedding(
    embedder: typing.Any, text: str, model: str = DEFAULT_EMBEDDING_MODEL
):
    start = time()

    text = text.replace('\n', ' ')
    embedding = (await embedder.create(input=[text], model=model)).data[0].embedding
    embedding = embedding[:EMBEDDING_DIM]

    end = time()
    logger.debug(f'embedded text of length {len(text)} in {end - start} ms')

    return embedding
