# Driver Operations Redesign Spec

**Status:** Draft (in progress)

## Goals

1. Operations interfaces become the core behavior — adding a new DB backend is as simple as implementing a driver with the operations interfaces filled out.
2. Operations interfaces are organized by object type (not one monolith).
3. DB-related functionality is closely linked to the Graphiti client via namespaces (`graphiti.nodes.entity.save(node)`), not scattered across data model classes.
4. No awkward override threading — no passing interfaces through multiple levels.
5. Data model classes (`EntityNode`, `EntityEdge`, etc.) become pure data (Pydantic models with no DB logic).
6. Phase 1 is non-breaking: existing methods on `EntityNode`/`EntityEdge` continue to work.

## Architecture Overview

Three layers:

```
Graphiti Client (graphiti.py)
  └── Namespace Wrappers (thin orchestration: embeddings, tracing)
        └── Operations ABCs (pure DB I/O, implemented per driver)
              └── GraphDriver (connection + query execution)
```

### User-Facing API

```python
graphiti = Graphiti(uri, user, password)

# Node operations
await graphiti.nodes.entity.save(node)
await graphiti.nodes.entity.get_by_uuid("abc-123")
await graphiti.nodes.episode.retrieve_episodes(reference_time, last_n=5)

# Edge operations
await graphiti.edges.entity.save(edge)
await graphiti.edges.entity.get_between_nodes(source_uuid, target_uuid)

# Transactions
async with graphiti.driver.transaction() as tx:
    await graphiti.nodes.entity.save(node1, tx=tx)
    await graphiti.nodes.entity.save(node2, tx=tx)

# High-level search (orchestration stays on client)
results = await graphiti.search(query, ...)
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parameterized vs. bound instances | Parameterized (`save(node)`) | Data classes stay pure, no hidden state, easier testing |
| Generic base vs. flat ops classes | Flat | Decoupled, easier to understand and debug |
| Embedding generation | Namespace layer | Driver stays pure DB I/O; namespace has access to both embedder and driver |
| `driver` param on ops methods | `QueryExecutor` passed explicitly each call | Ops depend on slim `QueryExecutor` ABC, not full `GraphDriver` — zero import cycles |
| `build_fulltext_query` | Lives on `SearchOperations` | Only consumed by search code |
| `load_embeddings` methods | Live on respective ops classes | They're per-object-type DB reads |
| Backwards compatibility | Keep existing data model methods in Phase 1 | Non-breaking first, cleanup later |
| Transaction API | Context manager (`async with driver.transaction() as tx`) | Pythonic, clean, uniform across drivers |
| Transaction typing | Typed `Transaction` ABC | Type safety without coupling to specific drivers |

## QueryExecutor and Transaction: Breaking the Import Cycle

Operations ABCs need to call `execute_query()` and `session()` on the driver, but
they must not import `GraphDriver` (which imports them). We solve this with a slim
base class that `GraphDriver` extends. The `Transaction` ABC is also defined here
since ops methods accept an optional transaction parameter.

```python
# graphiti_core/driver/query_executor.py — standalone, no deps on ops or GraphDriver

class Transaction(ABC):
    """Minimal transaction interface. Yielded by GraphDriver.transaction()."""

    @abstractmethod
    async def run(self, query: str, **kwargs) -> Any: ...


class QueryExecutor(ABC):
    """Slim interface for executing queries. GraphDriver extends this."""

    @abstractmethod
    async def execute_query(self, query: str, **kwargs) -> Any: ...

    @abstractmethod
    def session(self, database: str | None = None) -> GraphDriverSession: ...
```

**Dependency graph (strictly one-directional, no cycles):**

```
QueryExecutor + Transaction    (standalone — no deps)
     ↑
Operations ABCs                (depend on QueryExecutor + Transaction only)
     ↑
GraphDriver                    (extends QueryExecutor, composes Operations ABCs)
     ↑
Namespaces                     (depend on GraphDriver)
     ↑
Graphiti                       (depends on Namespaces + GraphDriver)
```

All operations ABC methods take `executor: QueryExecutor` and optionally `tx: Transaction | None`.
At runtime, the concrete driver (which is-a `QueryExecutor`) is passed through.

## Transaction API

### User-facing pattern

```python
# Transactional — groups operations, auto-commits on exit, rolls back on exception
async with graphiti.driver.transaction() as tx:
    await graphiti.nodes.entity.save(node1, tx=tx)
    await graphiti.nodes.entity.save(node2, tx=tx)
    await graphiti.edges.entity.save(edge, tx=tx)

# Non-transactional — each operation executes independently (default)
await graphiti.nodes.entity.save(node)
```

### Driver contract

```python
# On GraphDriver
@abstractmethod
def transaction(self) -> AsyncContextManager[Transaction]: ...
```

### Per-driver behavior

| Driver | `transaction()` behavior |
|--------|--------------------------|
| **Neo4j** | Opens a real transaction via `session.begin_transaction()`. Commits on clean exit, rolls back on exception. |
| **FalkorDB** | Returns a lightweight session wrapper. Queries execute immediately. No rollback on failure. |
| **Kuzu** | Same as FalkorDB — session wrapper, no rollback. |
| **Neptune** | Same as FalkorDB — session wrapper, no rollback. |

Drivers that lack native transaction support are honest about it — the API is
uniform but the guarantees differ. This matches the current behavior (where
`execute_write` is faked on non-Neo4j drivers) but makes it explicit.

### How `tx` flows through the layers

```
User code                          Namespace                           Ops ABC
─────────                          ─────────                           ───────
graphiti.nodes.entity.save(        EntityNodeNamespace.save(           EntityNodeOperations.save(
    node, tx=tx                        node, tx=tx                        executor, node, tx=tx
)                                  )                                   )
                                   │                                   │
                                   ├─ generate embeddings              ├─ if tx: tx.run(query)
                                   └─ delegate to ops                  └─ else: executor.execute_query(query)
```

### Implementation sketch for Neo4j

```python
class Neo4jTransaction(Transaction):
    def __init__(self, neo4j_tx):
        self._tx = neo4j_tx

    async def run(self, query: str, **kwargs) -> Any:
        result = await self._tx.run(query, **kwargs)
        return await result.data()


class Neo4jDriver(GraphDriver):
    @asynccontextmanager
    async def transaction(self):
        async with self._driver.session(database=self._database) as session:
            async with await session.begin_transaction() as tx:
                yield Neo4jTransaction(tx)
                await tx.commit()
```

### Implementation sketch for non-transactional drivers (e.g., FalkorDB)

```python
class FalkorTransaction(Transaction):
    """Thin wrapper — no real transaction, queries execute immediately."""

    def __init__(self, graph):
        self._graph = graph

    async def run(self, query: str, **kwargs) -> Any:
        return await self._graph.query(query, kwargs)


class FalkorDBDriver(GraphDriver):
    @asynccontextmanager
    async def transaction(self):
        graph = self.client.select_graph(self._database)
        yield FalkorTransaction(graph)
        # No commit/rollback — queries already executed
```

## Layer 1: Operations ABCs

All operations ABCs are flat (no generic base class). Each object type defines its own complete set of methods independently.

### EntityNodeOperations

```python
class EntityNodeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, node: EntityNode,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, nodes: list[EntityNode],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, node: EntityNode,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_group_id(self, executor: QueryExecutor,
                                  group_id: str, tx: Transaction | None = None,
                                  batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str], tx: Transaction | None = None,
                               batch_size: int = 100) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> EntityNode: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor, uuids: list[str]) -> list[EntityNode]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[EntityNode]: ...

    @abstractmethod
    async def load_embeddings(self, executor: QueryExecutor, node: EntityNode) -> None: ...

    @abstractmethod
    async def load_embeddings_bulk(self, executor: QueryExecutor,
                                    nodes: list[EntityNode],
                                    batch_size: int = 100) -> None: ...
```

### EpisodeNodeOperations

```python
class EpisodeNodeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, node: EpisodicNode,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, nodes: list[EpisodicNode],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, node: EpisodicNode,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_group_id(self, executor: QueryExecutor,
                                  group_id: str, tx: Transaction | None = None,
                                  batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str], tx: Transaction | None = None,
                               batch_size: int = 100) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> EpisodicNode: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[EpisodicNode]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[EpisodicNode]: ...

    @abstractmethod
    async def get_by_entity_node_uuid(self, executor: QueryExecutor,
                                       entity_node_uuid: str) -> list[EpisodicNode]: ...

    @abstractmethod
    async def retrieve_episodes(self, executor: QueryExecutor, reference_time: datetime,
                                 last_n: int = 3, group_ids: list[str] | None = None,
                                 source: str | None = None,
                                 saga: str | None = None) -> list[EpisodicNode]: ...
```

### CommunityNodeOperations

```python
class CommunityNodeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, node: CommunityNode,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, nodes: list[CommunityNode],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, node: CommunityNode,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_group_id(self, executor: QueryExecutor,
                                  group_id: str, tx: Transaction | None = None,
                                  batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str], tx: Transaction | None = None,
                               batch_size: int = 100) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> CommunityNode: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[CommunityNode]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[CommunityNode]: ...

    @abstractmethod
    async def load_name_embedding(self, executor: QueryExecutor,
                                   node: CommunityNode) -> None: ...
```

### SagaNodeOperations

```python
class SagaNodeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, node: SagaNode,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, nodes: list[SagaNode],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, node: SagaNode,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_group_id(self, executor: QueryExecutor,
                                  group_id: str, tx: Transaction | None = None,
                                  batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str], tx: Transaction | None = None,
                               batch_size: int = 100) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> SagaNode: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[SagaNode]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[SagaNode]: ...
```

### EntityEdgeOperations

```python
class EntityEdgeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, edge: EntityEdge,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, edges: list[EntityEdge],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, edge: EntityEdge,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str],
                               tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> EntityEdge: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[EntityEdge]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[EntityEdge]: ...

    @abstractmethod
    async def get_between_nodes(self, executor: QueryExecutor,
                                 source_node_uuid: str,
                                 target_node_uuid: str) -> list[EntityEdge]: ...

    @abstractmethod
    async def get_by_node_uuid(self, executor: QueryExecutor,
                                node_uuid: str) -> list[EntityEdge]: ...

    @abstractmethod
    async def load_embeddings(self, executor: QueryExecutor, edge: EntityEdge) -> None: ...

    @abstractmethod
    async def load_embeddings_bulk(self, executor: QueryExecutor,
                                    edges: list[EntityEdge],
                                    batch_size: int = 100) -> None: ...
```

### EpisodicEdgeOperations

```python
class EpisodicEdgeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, edge: EpisodicEdge,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, edges: list[EpisodicEdge],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, edge: EpisodicEdge,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str],
                               tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> EpisodicEdge: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[EpisodicEdge]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[EpisodicEdge]: ...
```

### CommunityEdgeOperations

```python
class CommunityEdgeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, edge: CommunityEdge,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, edge: CommunityEdge,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str],
                               tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> CommunityEdge: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[CommunityEdge]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[CommunityEdge]: ...
```

### HasEpisodeEdgeOperations

```python
class HasEpisodeEdgeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, edge: HasEpisodeEdge,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, edges: list[HasEpisodeEdge],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, edge: HasEpisodeEdge,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str],
                               tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> HasEpisodeEdge: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[HasEpisodeEdge]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[HasEpisodeEdge]: ...
```

### NextEpisodeEdgeOperations

```python
class NextEpisodeEdgeOperations(ABC):
    @abstractmethod
    async def save(self, executor: QueryExecutor, edge: NextEpisodeEdge,
                   tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def save_bulk(self, executor: QueryExecutor, edges: list[NextEpisodeEdge],
                        tx: Transaction | None = None,
                        batch_size: int = 100) -> None: ...

    @abstractmethod
    async def delete(self, executor: QueryExecutor, edge: NextEpisodeEdge,
                     tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def delete_by_uuids(self, executor: QueryExecutor,
                               uuids: list[str],
                               tx: Transaction | None = None) -> None: ...

    @abstractmethod
    async def get_by_uuid(self, executor: QueryExecutor, uuid: str) -> NextEpisodeEdge: ...

    @abstractmethod
    async def get_by_uuids(self, executor: QueryExecutor,
                            uuids: list[str]) -> list[NextEpisodeEdge]: ...

    @abstractmethod
    async def get_by_group_ids(self, executor: QueryExecutor, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[NextEpisodeEdge]: ...
```

### SearchOperations

```python
class SearchOperations(ABC):
    # Node search
    @abstractmethod
    async def node_fulltext_search(self, executor: QueryExecutor, query: str,
                                    search_filter: Any, group_ids: list[str] | None = None,
                                    limit: int = 10) -> list[EntityNode]: ...

    @abstractmethod
    async def node_similarity_search(self, executor: QueryExecutor, search_vector: list[float],
                                      search_filter: Any, group_ids: list[str] | None = None,
                                      limit: int = 10,
                                      min_score: float = 0.6) -> list[EntityNode]: ...

    @abstractmethod
    async def node_bfs_search(self, executor: QueryExecutor,
                               origin_uuids: list[str], search_filter: Any,
                               max_depth: int, group_ids: list[str] | None = None,
                               limit: int = 10) -> list[EntityNode]: ...

    # Edge search
    @abstractmethod
    async def edge_fulltext_search(self, executor: QueryExecutor, query: str,
                                    search_filter: Any, group_ids: list[str] | None = None,
                                    limit: int = 10) -> list[EntityEdge]: ...

    @abstractmethod
    async def edge_similarity_search(self, executor: QueryExecutor, search_vector: list[float],
                                      source_node_uuid: str | None,
                                      target_node_uuid: str | None,
                                      search_filter: Any,
                                      group_ids: list[str] | None = None,
                                      limit: int = 10,
                                      min_score: float = 0.6) -> list[EntityEdge]: ...

    @abstractmethod
    async def edge_bfs_search(self, executor: QueryExecutor,
                               origin_uuids: list[str], max_depth: int,
                               search_filter: Any, group_ids: list[str] | None = None,
                               limit: int = 10) -> list[EntityEdge]: ...

    # Episode search
    @abstractmethod
    async def episode_fulltext_search(self, executor: QueryExecutor, query: str,
                                       search_filter: Any,
                                       group_ids: list[str] | None = None,
                                       limit: int = 10) -> list[EpisodicNode]: ...

    # Community search
    @abstractmethod
    async def community_fulltext_search(self, executor: QueryExecutor, query: str,
                                         group_ids: list[str] | None = None,
                                         limit: int = 10) -> list[CommunityNode]: ...

    @abstractmethod
    async def community_similarity_search(self, executor: QueryExecutor,
                                           search_vector: list[float],
                                           group_ids: list[str] | None = None,
                                           limit: int = 10,
                                           min_score: float = 0.6) -> list[CommunityNode]: ...

    # Rerankers
    @abstractmethod
    async def node_distance_reranker(self, executor: QueryExecutor,
                                      node_uuids: list[str],
                                      center_node_uuid: str,
                                      min_score: float = 0) -> list[EntityNode]: ...

    @abstractmethod
    async def episode_mentions_reranker(self, executor: QueryExecutor,
                                         node_uuids: list[str],
                                         min_score: float = 0) -> list[EntityNode]: ...

    # Filter builders (sync)
    @abstractmethod
    def build_node_search_filters(self, search_filters: Any) -> Any: ...

    @abstractmethod
    def build_edge_search_filters(self, search_filters: Any) -> Any: ...

    # Fulltext query builder
    @abstractmethod
    def build_fulltext_query(self, query: str, group_ids: list[str] | None = None,
                              max_query_length: int = 8000) -> str: ...
```

### GraphMaintenanceOperations

```python
class GraphMaintenanceOperations(ABC):
    @abstractmethod
    async def clear_data(self, executor: QueryExecutor,
                          group_ids: list[str] | None = None) -> None: ...

    @abstractmethod
    async def build_indices_and_constraints(self, executor: QueryExecutor,
                                             delete_existing: bool = False) -> None: ...

    @abstractmethod
    async def delete_all_indexes(self, executor: QueryExecutor) -> None: ...

    @abstractmethod
    async def get_community_clusters(self, executor: QueryExecutor,
                                      group_ids: list[str] | None = None) -> list: ...

    @abstractmethod
    async def remove_communities(self, executor: QueryExecutor) -> None: ...

    @abstractmethod
    async def determine_entity_community(self, executor: QueryExecutor,
                                          entity: EntityNode) -> None: ...

    @abstractmethod
    async def get_mentioned_nodes(self, executor: QueryExecutor,
                                   episodes: list[EpisodicNode]) -> list[EntityNode]: ...

    @abstractmethod
    async def get_communities_by_nodes(self, executor: QueryExecutor,
                                        nodes: list[EntityNode]) -> list[CommunityNode]: ...
```

## Layer 2: GraphDriver Composes Operations

```python
class GraphDriver(QueryExecutor, ABC):
    # --- Core connection methods ---
    # execute_query() and session() inherited from QueryExecutor

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def transaction(self) -> AsyncContextManager[Transaction]: ...

    # --- Operations interfaces (all required, all abstract) ---
    @property
    @abstractmethod
    def entity_node_ops(self) -> EntityNodeOperations: ...

    @property
    @abstractmethod
    def episode_node_ops(self) -> EpisodeNodeOperations: ...

    @property
    @abstractmethod
    def community_node_ops(self) -> CommunityNodeOperations: ...

    @property
    @abstractmethod
    def saga_node_ops(self) -> SagaNodeOperations: ...

    @property
    @abstractmethod
    def entity_edge_ops(self) -> EntityEdgeOperations: ...

    @property
    @abstractmethod
    def episodic_edge_ops(self) -> EpisodicEdgeOperations: ...

    @property
    @abstractmethod
    def community_edge_ops(self) -> CommunityEdgeOperations: ...

    @property
    @abstractmethod
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations: ...

    @property
    @abstractmethod
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations: ...

    @property
    @abstractmethod
    def search_ops(self) -> SearchOperations: ...

    @property
    @abstractmethod
    def graph_ops(self) -> GraphMaintenanceOperations: ...
```

Example driver implementation:

```python
class Neo4jDriver(GraphDriver):
    def __init__(self, uri, user, password):
        # ... connection setup ...
        self._entity_node_ops = Neo4jEntityNodeOps()
        self._episode_node_ops = Neo4jEpisodeNodeOps()
        self._community_node_ops = Neo4jCommunityNodeOps()
        self._saga_node_ops = Neo4jSagaNodeOps()
        self._entity_edge_ops = Neo4jEntityEdgeOps()
        self._episodic_edge_ops = Neo4jEpisodicEdgeOps()
        self._community_edge_ops = Neo4jCommunityEdgeOps()
        self._has_episode_edge_ops = Neo4jHasEpisodeEdgeOps()
        self._next_episode_edge_ops = Neo4jNextEpisodeEdgeOps()
        self._search_ops = Neo4jSearchOps()
        self._graph_ops = Neo4jGraphMaintenanceOps()

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    # ... etc for all ops properties ...
```

## Layer 3: Namespace Wrappers

Thin wrappers on the Graphiti client that orchestrate non-DB concerns
(embedding generation, tracing) before delegating to the driver's ops.

```python
class EntityNodeNamespace:
    def __init__(self, driver: GraphDriver, embedder: EmbedderClient):
        self._driver = driver
        self._embedder = embedder
        self._ops = driver.entity_node_ops

    async def save(self, node: EntityNode,
                   tx: Transaction | None = None) -> EntityNode:
        await node.generate_name_embedding(self._embedder)
        await self._ops.save(self._driver, node, tx=tx)
        return node

    async def save_bulk(self, nodes: list[EntityNode],
                         tx: Transaction | None = None,
                         batch_size: int = 100) -> None:
        await self._ops.save_bulk(self._driver, nodes, tx=tx, batch_size=batch_size)

    async def delete(self, node: EntityNode,
                     tx: Transaction | None = None) -> None:
        await self._ops.delete(self._driver, node, tx=tx)

    async def delete_by_group_id(self, group_id: str,
                                  tx: Transaction | None = None,
                                  batch_size: int = 100) -> None:
        await self._ops.delete_by_group_id(self._driver, group_id, tx=tx, batch_size=batch_size)

    async def delete_by_uuids(self, uuids: list[str],
                               tx: Transaction | None = None,
                               batch_size: int = 100) -> None:
        await self._ops.delete_by_uuids(self._driver, uuids, tx=tx, batch_size=batch_size)

    async def get_by_uuid(self, uuid: str) -> EntityNode:
        return await self._ops.get_by_uuid(self._driver, uuid)

    async def get_by_uuids(self, uuids: list[str]) -> list[EntityNode]:
        return await self._ops.get_by_uuids(self._driver, uuids)

    async def get_by_group_ids(self, group_ids: list[str],
                                limit: int | None = None,
                                uuid_cursor: str | None = None) -> list[EntityNode]:
        return await self._ops.get_by_group_ids(self._driver, group_ids, limit, uuid_cursor)

    async def load_embeddings(self, node: EntityNode) -> None:
        await self._ops.load_embeddings(self._driver, node)

    async def load_embeddings_bulk(self, nodes: list[EntityNode],
                                    batch_size: int = 100) -> None:
        await self._ops.load_embeddings_bulk(self._driver, nodes, batch_size)


class NodeNamespace:
    """Accessed as graphiti.nodes"""
    def __init__(self, driver: GraphDriver, embedder: EmbedderClient):
        self.entity = EntityNodeNamespace(driver, embedder)
        self.episode = EpisodeNodeNamespace(driver)
        self.community = CommunityNodeNamespace(driver, embedder)
        self.saga = SagaNodeNamespace(driver)


class EdgeNamespace:
    """Accessed as graphiti.edges"""
    def __init__(self, driver: GraphDriver, embedder: EmbedderClient):
        self.entity = EntityEdgeNamespace(driver, embedder)
        self.episodic = EpisodicEdgeNamespace(driver)
        self.community = CommunityEdgeNamespace(driver)
        self.has_episode = HasEpisodeEdgeNamespace(driver)
        self.next_episode = NextEpisodeEdgeNamespace(driver)
```

Wired up in the Graphiti client:

```python
class Graphiti:
    def __init__(self, ..., graph_driver: GraphDriver | None = None, ...):
        self.driver = graph_driver or Neo4jDriver(uri, user, password)
        self.embedder = embedder or OpenAIEmbedder()
        self.nodes = NodeNamespace(self.driver, self.embedder)
        self.edges = EdgeNamespace(self.driver, self.embedder)

        # High-level search orchestration stays as methods on Graphiti.
        # Low-level search queries delegate to self.driver.search_ops.
```

## File Layout

```
graphiti_core/
  driver/
    query_executor.py                # QueryExecutor ABC (standalone, no deps)
    driver.py                        # GraphDriver(QueryExecutor) ABC, GraphDriverSession ABC
    operations/
      __init__.py                    # Re-exports all operations ABCs
      entity_node_ops.py             # EntityNodeOperations ABC
      episode_node_ops.py            # EpisodeNodeOperations ABC
      community_node_ops.py          # CommunityNodeOperations ABC
      saga_node_ops.py               # SagaNodeOperations ABC
      entity_edge_ops.py             # EntityEdgeOperations ABC
      episodic_edge_ops.py           # EpisodicEdgeOperations ABC
      community_edge_ops.py          # CommunityEdgeOperations ABC
      has_episode_edge_ops.py        # HasEpisodeEdgeOperations ABC
      next_episode_edge_ops.py       # NextEpisodeEdgeOperations ABC
      search_ops.py                  # SearchOperations ABC
      graph_ops.py                   # GraphMaintenanceOperations ABC
    neo4j/
      driver.py                      # Neo4jDriver(GraphDriver)
      operations/
        entity_node_ops.py           # Neo4jEntityNodeOps
        episode_node_ops.py          # Neo4jEpisodeNodeOps
        community_node_ops.py        # Neo4jCommunityNodeOps
        saga_node_ops.py             # Neo4jSagaNodeOps
        entity_edge_ops.py           # Neo4jEntityEdgeOps
        episodic_edge_ops.py         # Neo4jEpisodicEdgeOps
        community_edge_ops.py        # Neo4jCommunityEdgeOps
        has_episode_edge_ops.py      # Neo4jHasEpisodeEdgeOps
        next_episode_edge_ops.py     # Neo4jNextEpisodeEdgeOps
        search_ops.py                # Neo4jSearchOps
        graph_ops.py                 # Neo4jGraphMaintenanceOps
    falkordb/
      driver.py
      operations/
        ...                          # Same structure as neo4j/operations/
  namespaces/
    __init__.py
    nodes.py                         # NodeNamespace + EntityNodeNamespace, etc.
    edges.py                         # EdgeNamespace + EntityEdgeNamespace, etc.
  graphiti.py                        # Graphiti client with .nodes, .edges properties
  nodes.py                           # Data models (existing DB methods kept, deprecated)
  edges.py                           # Data models (existing DB methods kept, deprecated)
  search/
    search.py                        # High-level search orchestration (unchanged)
    search_utils.py                  # Will gradually migrate to use driver.search_ops
```

## Migration Strategy

### Phase 1: Non-Breaking (this round)

1. Define all operations ABCs in `driver/operations/`
2. Create Neo4j ops implementations (extract query logic from `nodes.py`, `edges.py`, `search_utils.py`)
3. Create namespace wrappers in `namespaces/`
4. Wire `Graphiti` with `self.nodes`, `self.edges`
5. **Keep all existing methods on data model classes working as-is**
6. Internal code can start using namespaces incrementally

### Phase 2: Breaking Cleanup (later)

1. Remove DB methods from `EntityNode`, `EntityEdge`, etc.
2. Remove old `SearchInterface` and `GraphOperationsInterface`
3. Update all internal callers to use namespace API
4. Remove provider-branching from utility files
5. Remove `search_interface` and `graph_operations_interface` from driver

## Resolved Questions

- **Import cycles:** Resolved via `QueryExecutor` ABC. Ops ABCs depend on `QueryExecutor`, not `GraphDriver`. No cycles, no `__future__` workarounds.
- **Embedding loading methods:** Confirmed — live on the respective ops classes (per-object-type DB reads).
- **`build_fulltext_query`:** Confirmed — lives on `SearchOperations`.

## Open Questions

None — all design questions resolved.
