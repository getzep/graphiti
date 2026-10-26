# Oracle AI Database 26ai driver

The optional Oracle backend stores Graphiti nodes and edges in Oracle AI Database 26ai using
native SQL, JSON/CLOB, VECTOR, Oracle Text and SQL/PGQ. It does not translate Cypher. The backend
implements all eleven driver operation families and the existing legacy namespace interfaces.

## Installation and prerequisites

```sh
pip install 'graphiti-core[oracle]'
```

Use a dedicated Oracle AI Database 26ai service with native VECTOR, Oracle Text and SQL Property
Graph capabilities enabled. The driver owns two tables, a migration ledger, one Property Graph,
two Text indexes and two vector indexes. Schema setup requires privileges to create and maintain
those objects; a separate restricted runtime account should receive only the permissions needed
for its graph operations. The schema uses 1024-dimensional FLOAT32 embeddings.

Python 3.12 is the runtime exercised by installed-wheel native smoke tests for this contribution.
Other supported core Python versions are not implied to have passed live-database qualification.
Use a valid TLS certificate chain for the selected runtime; never disable certificate validation.
In particular, Python 3.13 can reject a legacy wallet CA that Python 3.12 accepts.

Configure the connection through environment variables; do not commit their values:

```sh
export ORACLE_DSN='your_service_low'
export ORACLE_USER='your_graphiti_user'
# Supply ORACLE_PASSWORD through your secret manager or interactive environment.
export ORACLE_CONFIG_DIR='/path/to/your-wallet'
export ORACLE_WALLET_DIR='/path/to/your-wallet'
# Supply ORACLE_WALLET_PASSWORD through your secret manager when required.
```

After configuring a dedicated test service, `python examples/oracle26ai/quickstart.py` applies
forward-only driver migrations and prints schema readiness. It makes no LLM or embedding call.
Do not run schema provisioning against an unrelated application or production schema.

## Using the driver with Graphiti

```python
import os
from pydantic import SecretStr
from graphiti_core import Graphiti
from graphiti_core.driver.oracle_driver import Oracle26aiConfig, Oracle26aiDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

async def connect_graphiti() -> Graphiti:
    driver = Oracle26aiDriver(Oracle26aiConfig(
        dsn=os.environ['ORACLE_DSN'],
        user=os.environ['ORACLE_USER'],
        password=SecretStr(os.environ['ORACLE_PASSWORD']),
        config_dir=os.environ.get('ORACLE_CONFIG_DIR'),
        wallet_location=os.environ.get('ORACLE_WALLET_DIR'),
        wallet_password=(SecretStr(os.environ['ORACLE_WALLET_PASSWORD'])
                         if os.environ.get('ORACLE_WALLET_PASSWORD') else None),
    ))
    return Graphiti(
        graph_driver=driver,
        embedder=OpenAIEmbedder(OpenAIEmbedderConfig(embedding_dim=1024)),
    )
```

The Graphiti example uses its normal OpenAI client configuration, including an API key. You may
instead inject another LLM client and a compatible 1024-dimensional embedder. Model-provider
configuration is independent of Oracle persistence support. Construct/use the async driver inside
your application's async lifecycle and call `await driver.close()` during shutdown.

## Transactions and group isolation

Composite driver writes use one connection and commit together or roll back together. A commit
exception also attempts rollback before releasing the connection. Graphiti ingestion itself can
make several driver commits; this backend does not promise a single transaction around an entire
LLM-assisted ingestion workflow.

UUID/kind/group identity cannot be reassigned by a write. Scoped clones cannot be rebound to a
different group and cannot perform global schema/index maintenance. Edges lock and validate their
endpoints in deterministic UUID order before persistence. Endpoint and edge groups must agree.

Large allowlisted CLOB/JSON binds are transported with temporary connection-owned CLOB locators
above 4000 UTF-8 bytes. Payloads are neither truncated nor rewritten, including multibyte text.

## Oracle Text readiness

Oracle Text index maintenance is not an immediate commit-to-search guarantee. After a completed
write batch has committed, callers that require immediate text visibility must use the root driver:

```python
await driver.sync_fulltext_indexes()
```

The barrier synchronizes only the two fixed driver-owned Text indexes. A scoped clone cannot call
it. The application should verify unique per-batch canaries before publishing a searchable receipt:

`writes -> commit -> root Text barrier -> unique canaries -> searchable receipt`.

The driver does not implement an external ingestion ledger, source connector or CDC/checkpoint
system. Applications own retries, raw-source custody and replay/reconciliation policy.

## Episode removal and replay boundaries

Oracle episode removal uses MENTIONS and reverse fact provenance to identify surviving source
support. Shared facts keep surviving episode UUIDs. Entity summaries are prepared from remaining
raw episodes before persistence; missing, empty or duplicate results abort removal. A locked graph
snapshot is checked again, then changes are committed atomically. This path does not fall through
to the generic Cypher mention-count query.

Removal fails closed for Saga membership, missing source rows/content and incomplete incident
fact provenance. Entity attributes are preserved, not semantically retracted. Community/Saga
summaries, historical invalidations, Saga chains/watermarks and semantic retraction after direct
fact deletion are outside this removal contract. A group's snapshot and row locks have costs
proportional to that group: serialize competing maintenance or retry the complete operation after
a transient conflict/deadlock. No throughput or scale guarantee follows from the unit tests.

Graphiti-generated Oracle MENTIONS have deterministic group/source/target identity. Direct driver
UUIDs remain caller-owned. Replay coverage uses unchanged input, stable canonical node/fact IDs
and deterministic extraction doubles; it does not establish repeatability of a real LLM.
Changed source payloads require application-owned versioned episode IDs. Historical random
MENTIONS and already lost provenance are not automatically repaired.

Oracle Saga summaries persist a composite creation-time/UUID cursor so tied timestamps are not
lost. Existing backend three-field summary interfaces and positional limit arguments remain
compatible; the Oracle cursor is not a claim of new native behavior in the other backends.

## Tests

Hermetic Oracle contracts use fake connections/executors and deterministic semantic doubles.
They need no database, credentials or model-provider call:

```sh
uv sync --extra dev --extra oracle
GRAPHITI_TELEMETRY_ENABLED=false DISABLE_NEO4J=1 DISABLE_FALKORDB=1 \\
  DISABLE_KUZU=1 DISABLE_NEPTUNE=1 \\
  uv run pytest tests/driver/oracle -m 'not integration' -q
```

Native tests have the `_int.py` suffix and `integration` marker and are disabled by default.
They use synthetic owned groups. Point them only at a dedicated, configured Oracle test schema:

```sh
RUN_ORACLE_TESTS=1 GRAPHITI_TELEMETRY_ENABLED=false \\
  uv run pytest tests/driver/oracle -m 'integration and not falkor' -q
```

FalkorDB differential tests additionally require `RUN_FALKOR_TESTS=1` and an explicitly configured
FalkorDB service. A successful hermetic gate is not a live Oracle run, a live LLM-quality test,
application acceptance, capacity certification or production approval.

### Validation scope

Local validation on Python 3.12.12 passed the repository's `make check`: Ruff, Pyright and
1008 tests, with 53 skipped and 39 deselected. The dedicated Oracle hermetic selection passed
174 cases with 29 integration cases deselected. These counts are not live-database counts.

A clean installation of the built wheel also passed two native Oracle smoke tests: large UTF-8
CLOB/JSON round-trip with Text tail-token canaries, and shared-fact episode removal with a
deterministic summary helper. Imports resolved to the installed wheel, not an editable source
tree. Test-owned fixtures were cleaned and the existing schema/index readiness was unchanged.
No live model call was used. This does not qualify every native test, runtime, TLS configuration
or application workload; in particular, no Python 3.13 native qualification is claimed.
