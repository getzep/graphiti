"""Graphiti client construction and the async batch worker.

Everything here is configured for a fully local deployment: an OpenAI-compatible
LLM endpoint (vLLM), an OpenAI-compatible embeddings endpoint (TEI/vLLM), a local
graph database, and no reranker that phones home.

## One Graphiti instance per graph — why

graphiti_core 0.29.3 treats `group_id` as the *database name*. In `add_episode`:

    if group_id != self.driver._database:
        self.driver = self.driver.clone(database=group_id)
        self.clients.driver = self.driver

Two consequences that shape this module:

1. A graph's data lives in its own database, so a node/episode UUID is only
   resolvable inside that database. Zep's `GET graph/node/{uuid}` carries no
   graph_id, hence the uuid->graph index in `store.py`.
2. That assignment MUTATES the shared instance. Two concurrent `add_episode`
   calls for different graphs on one instance would clobber each other's driver
   and write into the wrong database. So we keep one instance per graph_id,
   already pointed at the right database — which also means `group_id ==
   driver._database` and the clone never fires.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType, EpisodicNode

from .ontology import build_edge_types, build_entity_types
from .store import Store

logger = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, '') else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ''):
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


class PassthroughCrossEncoder(CrossEncoderClient):
    """A reranker that does not rerank.

    Graphiti's default is OpenAIRerankerClient, which is unusable locally for two
    reasons: it needs an OPENAI_API_KEY at construction time, and it scores with
    ``logit_bias={'6432': 1, '7983': 1}`` — hard-coded OpenAI tokenizer IDs for
    True/False that mean nothing to a Qwen or Llama tokenizer, so its scores
    would be noise. The RRF search recipes never call a cross-encoder, so this
    exists to satisfy the constructor and to degrade gracefully if a
    cross-encoder recipe is ever selected.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        total = len(passages)
        return [(passage, 1.0 - (i / total if total else 0)) for i, passage in enumerate(passages)]


def _build_cross_encoder() -> CrossEncoderClient:
    kind = (_env('GRAPHITI_RERANKER', 'none') or 'none').strip().lower()
    if kind == 'bge':
        # Local cross-encoder; downloads BAAI/bge-reranker-v2-m3 (~2.2GB) once.
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

        return BGERerankerClient()
    if kind not in {'none', 'passthrough'}:
        logger.warning('Unknown GRAPHITI_RERANKER=%r; using passthrough', kind)
    return PassthroughCrossEncoder()


def build_driver(database: str | None = None):
    backend = (_env('GRAPHITI_DB_BACKEND', 'falkordb') or 'falkordb').strip().lower()
    if backend == 'falkordb':
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        return FalkorDriver(
            host=_env('FALKORDB_HOST', 'localhost') or 'localhost',
            port=int(_env('FALKORDB_PORT', '6379') or 6379),
            password=_env('FALKORDB_PASSWORD'),
            database=database or _env('FALKORDB_DATABASE', 'default_db') or 'default_db',
        )
    if backend == 'neo4j':
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        return Neo4jDriver(
            uri=_env('NEO4J_URI', 'bolt://localhost:7687') or 'bolt://localhost:7687',
            user=_env('NEO4J_USER', 'neo4j'),
            password=_env('NEO4J_PASSWORD', 'password'),
            database=database or _env('NEO4J_DATABASE', 'neo4j') or 'neo4j',
        )
    raise ValueError(f'Unsupported GRAPHITI_DB_BACKEND={backend!r} (use falkordb or neo4j)')


def build_graphiti(database: str | None = None) -> Graphiti:
    llm_client = OpenAIGenericClient(
        config=LLMConfig(
            api_key=_env('GRAPHITI_LLM_API_KEY', 'local') or 'local',
            base_url=_env('GRAPHITI_LLM_BASE_URL', 'http://localhost:8000/v1'),
            model=_env('GRAPHITI_LLM_MODEL', 'local-llm') or 'local-llm',
            small_model=_env('GRAPHITI_LLM_SMALL_MODEL', _env('GRAPHITI_LLM_MODEL', 'local-llm')),
            temperature=float(_env('GRAPHITI_LLM_TEMPERATURE', '0.0') or 0.0),
        ),
        max_tokens=int(_env('GRAPHITI_LLM_MAX_TOKENS', '16384') or 16384),
        # 'json_schema' uses the server's constrained decoding (vLLM xgrammar).
        # Switch to 'json_object' only for endpoints that reject json_schema.
        structured_output_mode=_env('GRAPHITI_STRUCTURED_OUTPUT_MODE', 'json_schema'),  # type: ignore[arg-type]
    )

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=_env('GRAPHITI_EMBEDDER_API_KEY', 'local') or 'local',
            base_url=_env('GRAPHITI_EMBEDDER_BASE_URL', 'http://localhost:8081/v1'),
            embedding_model=_env('GRAPHITI_EMBEDDER_MODEL', 'BAAI/bge-m3') or 'BAAI/bge-m3',
        )
    )

    return Graphiti(
        graph_driver=build_driver(database),
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=_build_cross_encoder(),
        max_coroutines=int(_env('GRAPHITI_MAX_COROUTINES', '4') or 4),
    )


class GraphitiPool:
    """One Graphiti per graph_id. See the module docstring for why."""

    def __init__(
        self,
        factory: Callable[[str], Graphiti] = build_graphiti,
        build_indices: bool = True,
    ):
        self._factory = factory
        self._build_indices = build_indices
        self._instances: dict[str, Graphiti] = {}
        self._lock = asyncio.Lock()

    async def get(self, graph_id: str) -> Graphiti:
        instance = self._instances.get(graph_id)
        if instance is not None:
            return instance
        async with self._lock:
            # Re-check: another coroutine may have built it while we waited.
            instance = self._instances.get(graph_id)
            if instance is not None:
                return instance
            logger.info('Opening graph %s', graph_id)
            instance = self._factory(graph_id)
            # Hard invariant: the driver's database must BE the graph_id. If it
            # is not, add_episode(group_id=graph_id) clones the driver to a
            # database named graph_id and writes there instead — so anything we
            # pre-saved through this instance lands somewhere else and the
            # ingest fails with NodeNotFoundError. Fail loudly here rather than
            # debug that later.
            actual = getattr(instance.driver, '_database', None)
            if actual is not None and actual != graph_id:
                raise ValueError(
                    f'Graphiti driver database {actual!r} must equal graph_id '
                    f'{graph_id!r}; graphiti maps group_id onto the database name'
                )
            if self._build_indices:
                # Indices and constraints are per-database, so every graph needs
                # its own pass. Idempotent.
                await instance.build_indices_and_constraints()
            self._instances[graph_id] = instance
            return instance

    async def forget(self, graph_id: str) -> None:
        async with self._lock:
            instance = self._instances.pop(graph_id, None)
        if instance is not None:
            await instance.close()

    async def close(self) -> None:
        async with self._lock:
            instances = list(self._instances.values())
            self._instances.clear()
        for instance in instances:
            try:
                await instance.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.warning('Error closing a Graphiti instance', exc_info=True)


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return datetime.now(timezone.utc)


async def ingest_episode(
    graphiti: Graphiti,
    store: Store,
    *,
    graph_id: str,
    episode_uuid: str | None,
    name: str,
    body: str,
    data_type: str,
    source_description: str,
    reference_time: datetime,
) -> EpisodicNode:
    """Ingest one episode, optionally under a caller-chosen UUID.

    Zep's batch API hands the episode UUID to the client at `batch.add` time, so
    it is fixed before ingestion starts. Graphiti's `add_episode(uuid=...)` does
    NOT mean "create with this UUID" — it calls `EpisodicNode.get_by_uuid` and
    raises NodeNotFoundError if absent, i.e. it means "process this existing
    episode". So persist the node first, then hand its UUID to add_episode.
    Keeping one UUID end to end matters: MiroFish records these and polls them.
    """
    entity_specs, edge_specs = store.get_ontology(graph_id)
    entity_types = build_entity_types(entity_specs)
    edge_types, edge_type_map = build_edge_types(edge_specs)
    source = EpisodeType.from_str(data_type or 'text')

    if episode_uuid is not None:
        episode = EpisodicNode(
            uuid=episode_uuid,
            name=name,
            group_id=graph_id,
            labels=[],
            source=source,
            content=body,
            source_description=source_description,
            created_at=datetime.now(timezone.utc),
            valid_at=reference_time,
            entity_edges=[],
        )
        await episode.save(graphiti.driver)

    result = await graphiti.add_episode(
        uuid=episode_uuid,
        group_id=graph_id,
        name=name,
        episode_body=body,
        source=source,
        source_description=source_description,
        reference_time=reference_time,
        entity_types=entity_types or None,
        edge_types=edge_types or None,
        edge_type_map=edge_type_map or None,
    )

    store.remember_uuids(graph_id, 'episode', [result.episode.uuid])
    store.remember_uuids(graph_id, 'node', [n.uuid for n in result.nodes])
    return result.episode


@dataclass
class BatchWorker:
    """Processes batches out of the store, one item at a time per slot.

    Zep ingestion is asynchronous: MiroFish POSTs items, POSTs /process, then
    polls the summary until a terminal status. We reproduce that contract.
    """

    pool: GraphitiPool
    store: Store
    concurrency: int = 4
    _tasks: set[asyncio.Task] = field(default_factory=set)
    _semaphore: asyncio.Semaphore | None = None

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.concurrency)

    def submit(self, batch_id: str) -> None:
        task = asyncio.create_task(self._run_batch(batch_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def resume_incomplete(self) -> None:
        for batch_id in self.store.claim_draft_batches():
            logger.info('Resuming batch %s after restart', batch_id)
            self.submit(batch_id)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run_batch(self, batch_id: str) -> None:
        self.store.set_batch_status(batch_id, 'processing')
        items = self.store.pending_items(batch_id)
        if not items:
            self.store.set_batch_status(batch_id, 'succeeded', completed=True)
            return

        await asyncio.gather(*(self._run_item(item) for item in items))

        counts = self.store.item_counts(batch_id)
        failed = counts.get('failed', 0)
        succeeded = counts.get('succeeded', 0) + counts.get('skipped', 0)
        if failed == 0:
            status = 'succeeded'
        elif succeeded == 0:
            status = 'failed'
        else:
            status = 'partial'
        self.store.set_batch_status(batch_id, status, completed=True)
        logger.info('Batch %s finished: %s (%s)', batch_id, status, counts)

    async def _run_item(self, item: dict[str, Any]) -> None:
        assert self._semaphore is not None
        async with self._semaphore:
            item_id = item['item_id']
            self.store.set_item_status(item_id, 'processing')
            graph_id = item.get('graph_id')
            if not graph_id:
                self.store.set_item_status(
                    item_id, 'failed', {'message': 'item has no graph_id'}
                )
                return
            try:
                graphiti = await self.pool.get(graph_id)
                await ingest_episode(
                    graphiti,
                    self.store,
                    graph_id=graph_id,
                    episode_uuid=item['episode_uuid'],
                    name=item.get('name') or f'batch-{item["sequence_index"]}',
                    body=item.get('payload') or '',
                    data_type=item.get('data_type') or 'text',
                    source_description=item.get('source_description') or '',
                    reference_time=_parse_time(item.get('reference_time')),
                )
                self.store.set_item_status(item_id, 'succeeded')
            except Exception as exc:  # noqa: BLE001 - one bad item must not kill the batch
                logger.exception('Batch item %s failed', item_id)
                self.store.set_item_status(
                    item_id,
                    'failed',
                    {'message': str(exc), 'type': type(exc).__name__},
                )


@dataclass
class Runtime:
    pool: GraphitiPool
    store: Store
    worker: BatchWorker

    @classmethod
    async def create(cls) -> Runtime:
        store = Store(_env('ZEP_COMPAT_DB_PATH', './data/zep_compat.sqlite3'))
        pool = GraphitiPool(build_indices=_env_bool('ZEP_COMPAT_BUILD_INDICES', True))
        worker = BatchWorker(
            pool=pool,
            store=store,
            concurrency=int(_env('ZEP_COMPAT_BATCH_CONCURRENCY', '4') or 4),
        )
        await worker.resume_incomplete()
        return cls(pool=pool, store=store, worker=worker)

    async def close(self) -> None:
        await self.worker.drain()
        await self.pool.close()
        self.store.close()
