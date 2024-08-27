import logging
import typing
from time import time

from graphiti_core.llm_client.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


async def generate_embedding(
    embedder: typing.Any, text: str, model: str = 'text-embedding-3-small'
):
    start = time()

    text = text.replace('\n', ' ')
    embedding = (await embedder.create(input=[text], model=model)).data[0].embedding
    embedding = embedding[:EMBEDDING_DIM]

    end = time()
    logger.debug(f'embedded text of length {len(text)} in {end-start} ms')

    return embedding
