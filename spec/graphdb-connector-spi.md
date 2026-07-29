# Graph-DB Connector SPI Spec

**Status:** Draft (for review)
**Tracking issue:** [#1 — Pluggable graph-DB backends](https://github.com/ice1x/graphiti/issues/1)
**Related:** [`spec/driver-operations-redesign.md`](./driver-operations-redesign.md) (the in-flight "Phase 1" operations refactor this builds on)

> **Scope note (drevo in-fork).** drevo (#2) is being added *inside this fork*, not
> as an out-of-tree package. That does not need the full SPI (the closed
> `GraphProvider` enum can simply gain a `DREVO` value, and no connector registry is
> required). This spec's registry / open-identity parts (PRs C–D) remain the plan
> only if/when out-of-tree third-party connectors are pursued. The pieces drevo
> actually needs are the **capability model** (PR A, landed) and **capability-driven
> search + fallback** (PRs B/E).

## 1. Goal

Provide **one abstract base interface** from which a connector/adapter to *any*
graph database is derived, so a new backend can be added **out-of-tree, in a
separate package, with no changes to `graphiti-core`**.

Terminology note: what issue #1 calls a *connector/adapter* already exists in the
codebase as `GraphDriver`. We do **not** introduce a parallel abstraction — we
**promote the existing `GraphDriver` base into a public, documented, stable SPI**
and add the three pieces that are currently missing.

Acceptance criteria (from #1):

- A new backend can be added in a separate package implementing the SPI, with no
  changes to graphiti-core.
- Backends **without** native fulltext/vector indexes still work, via the
  external overlay / in-library fallback path.
- Built-in backends are refactored onto the same SPI (dogfooding).

## 2. Current state (what exists, what blocks)

| Piece | Status in code | Reference |
| --- | --- | --- |
| Base driver ABC | ✅ exists | `graphiti_core/driver/driver.py` — `GraphDriver(QueryExecutor, ABC)`, `GraphDriverSession(ABC)` |
| Per-backend adapters | ✅ exist (4) | `neo4j_driver.py`, `falkordb_driver.py`, `kuzu_driver.py`, `neptune_driver.py` + per-backend `operations/` dirs |
| Operations interfaces | 🟡 in progress ("Phase 1") | `driver/operations/*`, `driver/search_interface/`, `driver/graph_operations/` |
| Slim query interface | ✅ exists | `query_executor.py` — `QueryExecutor`, `Transaction` |
| External vector overlay | 🟡 partial | Milvus overlay; Neptune's `aoss_client` as a prototype of "external vector store" |
| **Capability negotiation** | ❌ **missing** | logic branches on identity: `if driver.provider == GraphProvider.X` — **89+ sites**, 33 in `search/search_utils.py` alone |
| **Open provider identity** | ❌ **missing** | `GraphProvider` is a **closed `Enum`** (`driver.py:59`) — an out-of-tree backend cannot add a value without patching core |
| **Connector registry** | ❌ **missing** | driver is hardcoded (`Neo4jDriver(...)` at `graphiti.py:212`); no entry points |
| **"Write a connector" docs** | ❌ **missing** | — |

**The two structural blockers** for the "no changes to core" criterion are:

1. **Closed `GraphProvider` Enum.** Third-party connectors have no legal provider
   identity.
2. **Branching on identity instead of capability.** `search_utils.py` and the
   query builders ask *"are you Neo4j?"* when they really mean *"do you have a
   native vector index?"*. A new backend that answers the second question can't
   express it.

Everything else (the ABC, the adapters, the operations split) is already the
right shape and just needs to be **frozen and documented as public API**.

## 3. Design

Four additions, each independently mergeable. None of them, on their own,
changes runtime behavior for the existing backends.

### 3.1 The base interface (SPI surface)

The SPI is the existing `GraphDriver` + `GraphDriverSession` + the operations
ABCs, with a documented, stable contract. A connector author implements:

```
GraphConnector  (alias / promoted GraphDriver)
├── lifecycle          connect / session() / close() / teardown
├── query execution    execute_query(cypher, **params)  → records
├── transactions       transaction()  (real, or the provided immediate-mode fallback)
├── schema             build_indices_and_constraints() / delete_all_indexes()
├── identity           provider  (built-in enum value OR opaque string id)
├── capabilities       capabilities: GraphCapabilities   ← NEW (§3.2)
└── operations         entity_node_ops, entity_edge_ops, … search_ops, graph_ops
                       (the Phase-1 operations ABCs)
```

No method signatures change in this spec. The deliverable is: mark this surface
public (export from `graphiti_core.driver`), document each method's contract,
and add a conformance test suite an adapter can run against itself.

### 3.2 Capability negotiation (NEW)

A connector declares what it supports natively. The search/index layer branches
on the **declared capability**, not on provider identity.

```python
# graphiti_core/driver/capabilities.py  (new)
from pydantic import BaseModel

class GraphCapabilities(BaseModel):
    fulltext_search: bool = False   # native fulltext (db.index.fulltext.* or equiv)
    vector_search:   bool = False   # native ANN / cosine similarity in-query
    vector_index:    bool = False   # can create a persistent vector index
    transactions:    bool = False   # real commit/rollback (not immediate-mode)
    parallel_runtime: bool = False  # Neo4j enterprise parallel runtime
    # room to grow: range_index, bfs_native, etc.
```

Declared per backend (illustrative — exact flags confirmed during PR B):

| Backend | fulltext | vector_search | vector_index | transactions |
| --- | --- | --- | --- | --- |
| Neo4j | ✅ | ✅ | ✅ | ✅ |
| FalkorDB | ✅ | ✅ | ✅ | ⚠️ |
| Kuzu | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Neptune | via AOSS overlay | via AOSS overlay | overlay | ✅ |
| **minimal Bolt/Cypher (e.g. drevo)** | ❌ | ❌ | ❌ | depends |

`provider` identity is **kept**, but only for genuine *Cypher-dialect* forks
(e.g. Kuzu's return syntax, Neptune's openCypher). Anything that is really a
"does this backend have feature X" question moves to `capabilities.X`. This is
the refactor tracked as PR B; it is behavior-preserving because the built-in
capability tables reproduce today's per-provider branches exactly.

### 3.3 Open provider identity (NEW)

Keep `GraphProvider` for built-ins, but widen the accepted type so out-of-tree
connectors are not forced into the enum:

```python
ProviderId = GraphProvider | str          # built-in enum OR opaque string

class GraphDriver(...):
    provider: ProviderId
```

Core code must never assume `provider` is enum-exhaustive: an unknown
`provider` is legal and simply routes through the capability-driven / fallback
paths. (This is why §3.2 must land — once branches key off capabilities, an
unknown provider id is no longer a problem.)

### 3.4 Connector registry (NEW)

Two ways to select a backend by name; third-party packages ship connectors via
entry points.

```python
# graphiti_core/driver/registry.py  (new)
def register_connector(name: str, factory: Callable[..., GraphDriver]) -> None: ...
def get_connector(name: str) -> Callable[..., GraphDriver]: ...
def available_connectors() -> list[str]: ...
```

- **Explicit:** `register_connector("drevo", DrevoDriver)` at import time.
- **Entry points:** a third-party package declares
  ```toml
  [project.entry-points."graphiti.connectors"]
  drevo = "graphiti_drevo:DrevoDriver"
  ```
  and `get_connector("drevo")` resolves it via `importlib.metadata.entry_points`
  (the same mechanism already used in `telemetry.py`).

Built-in drivers self-register under their `GraphProvider` value, so selection
is uniform: `get_connector("neo4j")` and `get_connector("drevo")` work the same
way. `Graphiti(...)` continues to accept a concrete driver instance as today
(no breaking change); name-based selection is additive.

### 3.5 Graceful fallback (design hook, implemented in a later PR)

When `not driver.capabilities.fulltext_search` / `not vector_search`, the search
layer routes to the **external vector-store overlay** (the Milvus path) and/or an
**in-library BM25** instead of emitting backend-native index procedures. This
spec only fixes the *branch point* (§3.2 makes it `capabilities`-driven); the
fallback wiring itself is issue #1's point 3 and lands as its own PR (E), and is
what makes a minimal Bolt/Cypher backend like drevo (#2) usable.

## 4. Backward compatibility

- Existing `GraphProvider` enum values unchanged.
- Existing driver classes and their public constructors unchanged.
- `Graphiti(uri, user, password)` and passing a concrete driver keep working.
- PR A (capabilities) and PR C (open identity) are pure additions — no runtime
  behavior change. PR B (branch-on-capability) is a behavior-preserving refactor
  guarded by the existing test suite plus new capability-table tests.

## 5. Delivery plan (PR sequence)

Each PR is small, independently reviewable, TDD (tests first), and targets the
fork (`ice1x/graphiti`).

| PR | Scope | Risk | Unblocks |
| --- | --- | --- | --- |
| **A** | `GraphCapabilities` type + per-backend declarations + tests | low (additive) | everything |
| **B** | Replace identity-branches with capability-branches where the branch is about a capability | medium (touches `search_utils`) | fallback |
| **C** | Widen `provider` to `GraphProvider | str`; stop assuming enum-exhaustiveness | low | registry |
| **D** | `register_connector` + entry-point discovery + name-based selection | low | #2 (drevo) |
| **E** | Capability-driven fallback to external overlay / BM25 | medium | #2 |
| **F** | "Writing a connector" guide + minimal reference connector + conformance test kit | low | closes #1 |

## 6. Testing strategy (TDD)

- **PR A:** unit tests asserting each built-in driver's declared capabilities;
  `GraphCapabilities` defaults.
- **PR B:** the existing driver/search suites must stay green (behavior
  preservation); add tests that an unknown-provider driver with capabilities off
  takes the fallback branch.
- **PR D:** unit tests for register/get/available; an entry-point discovery test
  using a dummy in-repo connector.
- **PR F:** a reusable **connector conformance suite** any adapter (built-in or
  third-party) can run — the reference minimal connector must pass it. Where a
  live DB is needed, mark `_int` per repo convention.

## 7. Open questions (for review)

1. **Name.** Keep the term `GraphDriver` in the public API (and treat
   "connector/adapter" as docs-level synonyms), or introduce a
   `GraphConnector` alias as the documented public name? Recommendation: keep
   `GraphDriver`, document "connector = driver", avoid churn.
2. **Capability granularity.** Is a flat boolean set enough, or do we need
   capability *values* (e.g. max vector dimensions, distance metrics)?
   Recommendation: start flat (booleans), extend later — non-breaking.
3. **Fallback ownership.** Should BM25 fallback live in `graphiti-core`, or only
   the external-overlay path, with BM25 as a separate helper package?
4. **Upstream intent.** Is this fork-only, or eventually a PR to `getzep/graphiti`?
   Affects how aggressively we can move `provider`-branching.
