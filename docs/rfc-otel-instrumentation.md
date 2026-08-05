# RFC: Add OpenTelemetry metrics + span instrumentation to core memory operations

> **Status:** Draft — ready to file as upstream issue on getzep/graphiti
> **Implementation:** Available in fork `henrikrexed/graphiti` at commit `c8c8c6b`

## Summary

Add optional OpenTelemetry instrumentation to Graphiti's core memory operations so that users who
deploy Graphiti with an OTel SDK can observe the system's behaviour in their existing observability
backends (Jaeger, Prometheus, Grafana Tempo, Honeycomb, Datadog, etc.).

## Motivation

Graphiti is increasingly used as the memory layer for AI agents in production. Production
deployments need:

1. **Latency observability** — how long does `add_episode`, `search`, or `remove_episode` take?
2. **Volume tracking** — how many nodes/edges/episodes are being stored and invalidated?
3. **Query efficiency** — how many results does each search return; what is the hit rate?
4. **Error rate** — which operations are failing and at what frequency?

Without instrumentation, operators must rely on DB-level metrics and cannot attribute latency to
specific Graphiti operations or distinguish between LLM calls and graph writes.

## Design

### Dual-impl, zero-dependency pattern

The design mirrors the existing `tracer.py` abstraction already in the codebase:

- A `GraphitiMeter` abstract base class with `NoOpMeter` (default, zero dependencies) and
  `OpenTelemetryMeter` (used when the caller passes an `opentelemetry.metrics.Meter` instance).
- `create_meter(otel_meter=None)` factory: returns `NoOpMeter` when `otel_meter` is `None` or when
  `opentelemetry-api` is not installed.
- `Graphiti.__init__` gains an optional `meter: GraphitiMeter | None = None` parameter alongside
  the existing `tracer` parameter.
- **Backwards-compatible**: existing code with no `meter` argument gets `NoOpMeter` — no behaviour
  change, no new runtime dependencies.

### Metric instruments (OTel naming)

| Instrument | Kind | Unit | Description |
|---|---|---|---|
| `memory.operation.duration` | Histogram | `ms` | Wall-clock duration per operation |
| `memory.operation.count` | Counter | `{operation}` | Total operations (labelled `memory.operation.name`, `memory.operation.status`) |
| `memory.items.stored` | Counter | `{item}` | Items written (labelled `memory.item.type`: node/edge/episode) |
| `memory.items.invalidated` | Counter | `{item}` | Items invalidated/deleted (same label) |
| `memory.query.result_count` | Histogram | `{item}` | Results returned per search |

### Spans

Operations instrumented with spans (building on the existing `tracer.py` pattern):

| Operation | Span existed? | Metrics added? |
|---|---|---|
| `add_episode` | Yes | Yes (node/edge/episode stored counts, invalidated edges) |
| `add_episode_bulk` | Yes | Yes (same, bulk) |
| `search` | **No — new** | Yes (duration, result count, error count) |
| `search_` | **No — new** | Yes (duration, result count, error count) |
| `remove_episode` | **No — new** | Yes (duration, invalidated node/edge/episode counts) |

### Files changed

- `graphiti_core/meter.py` *(new)* — `GraphitiMeter` ABC, `NoOpMeter`, `OpenTelemetryMeter`,
  `create_meter()`
- `graphiti_core/graphiti.py` — wire `meter` param; add metric calls alongside existing spans
- `graphiti_core/__init__.py` — re-export meter types

### Example usage

```python
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider

tracer_provider = TracerProvider()
meter_provider = MeterProvider()

otel_tracer = tracer_provider.get_tracer("graphiti")
otel_meter  = meter_provider.get_meter("graphiti", version="0.1.0")

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="...",
    tracer=OpenTelemetryTracer(otel_tracer),
    meter=create_meter(otel_meter),
)
```

## Alternatives considered

1. **Auto-instrument via a separate package** (e.g. `graphiti-opentelemetry`): More decoupled but
   means users have two packages to install and configure; the existing `tracer.py` sets the
   precedent for in-tree instrumentation.
2. **Use `opentelemetry.metrics.get_meter()` globally**: Simpler API but requires the user to
   configure the global meter provider before importing Graphiti, which is surprising and hard to
   test.
3. **Instrument only through the existing PostHog telemetry path**: PostHog is anonymous product
   analytics — not suitable for production observability or benchmark ingestion.

## Backwards compatibility

- Zero behaviour change for existing users: default `meter=None` → `NoOpMeter`.
- No new required dependencies: `opentelemetry-api` remains optional.
- The `make check` suite (Ruff + Pyright + Pytest) passes with no changes to existing tests.

## Open questions for reviewers

1. Should `memory.query.result_count` be a histogram or an UpDownCounter? Histogram lets you
   compute percentiles on result-set sizes; counter is simpler but only gives you totals.
2. Should the `memory.operation.name` attribute values match a specific semconv registry string or
   are the current values (`add_episode`, `search`, etc.) the right names?
3. Is there a preferred way to expose the `meter` parameter in the MCP server or FastAPI server
   layers (currently only the core library is instrumented)?

## Implementation reference

Fork with full implementation: https://github.com/henrikrexed/graphiti/commit/c8c8c6b
