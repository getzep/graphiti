import logging
from time import time

from graphiti_core.llm_client.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)

async def generate_embedding(embedder, text, model='text-embedding-3-small'):
    start = time()

    text = text.replace('\n', ' ')
    embedding = (await embedder.create(input=[text], model=model)).data[0].embedding
    embedding = embedding[:EMBEDDING_DIM]

    end = time()
    logger.info(f'embedded {text} in {end-start} ms')

    return embedding
