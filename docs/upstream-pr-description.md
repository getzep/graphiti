# feat(otel): OpenTelemetry metrics + span instrumentation for core memory operations

> **Base:** `getzep/graphiti:main` · **Head:** `henrikrexed/graphiti`
> **Linked RFC:** _(fill in issue #)_ — see `docs/rfc-otel-instrumentation.md` in this branch
> ⚠️ **RFC gate:** this PR must reference an accepted design issue or it will be labelled `needs-rfc`. File the RFC issue first and link it above before requesting review.

## Summary

Adds **optional** OpenTelemetry metrics and extends span instrumentation to Graphiti's core memory
operations. Users who run Graphiti with an OTel SDK can now observe latency, volume, and query
efficiency of `add_episode`, `search`, `search_`, and `remove_episode` in their existing backends
(Jaeger/Tempo for traces; Prometheus/Grafana/Datadog/Honeycomb for metrics). With no OTel meter
passed, everything is a no-op — **zero behaviour change and no new required dependencies** for
existing users.

This mirrors the design of the existing `graphiti_core/tracer.py` abstraction, so the meter follows
the same NoOp/OTel dual-impl discipline already established in the codebase.

## Motivation

Graphiti is increasingly used as the production memory layer for AI agents. Operators currently
cannot attribute latency to specific Graphiti operations or distinguish LLM calls from graph writes,
and there is **no metrics signal at all**. This PR closes that gap:

1. **Latency** — per-operation duration histograms.
2. **Volume** — nodes/edges/episodes stored and invalidated.
3. **Query efficiency** — result-set sizes per search.
4. **Error rate** — operation count labelled by status.

## Design

### Dual-impl, zero-dependency pattern

- New `GraphitiMeter` ABC with `NoOpMeter` (default, no deps) and `OpenTelemetryMeter`
  (active when the caller passes an `opentelemetry.metrics.Meter`).
- `create_meter(otel_meter=None)` factory returns `NoOpMeter` when `otel_meter is None` or when
  `opentelemetry-api` is not installed. All instrumentation errors are suppressed — telemetry never
  breaks a memory operation.
- `Graphiti.__init__` gains an optional `meter: GraphitiMeter | None = None` parameter alongside the
  existing `tracer`. Default `None` → `NoOpMeter`.

### Metric instruments (`memory.*`, OTel naming)

| Instrument | Kind | Unit | Key attributes |
|---|---|---|---|
| `memory.operation.duration` | Histogram | `ms` | `memory.operation.name` |
| `memory.operation.count` | Counter | `1` | `memory.operation.name`, `memory.operation.status` (`ok`/`error`) |
| `memory.items.stored` | Counter | `1` | `memory.item.type` (`node`/`edge`/`episode`) |
| `memory.items.invalidated` | Counter | `1` | `memory.item.type` |
| `memory.query.result_count` | Histogram | `1` | `memory.operation.name` |

> Units are plain UCUM `1`/`ms` (not `{item}`-style annotations) so OTLP payloads are accepted by
> strict backends.

### Spans + metrics per operation

| Operation | Span | Metrics added |
|---|---|---|
| `add_episode` | existing | node/edge/episode stored counts, invalidated edges |
| `add_episode_bulk` | existing | stored/invalidated counts (bulk) |
| `search` | **new** | duration, result count, op count/status |
| `search_` | **new** | duration, result count, op count/status |
| `remove_episode` | **new** | duration, invalidated node/edge/episode counts, op count/status |

### Example usage

```python
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from graphiti_core import Graphiti
from graphiti_core.meter import create_meter

otel_tracer = trace.get_tracer("graphiti")
otel_meter  = metrics.get_meter("graphiti", version="0.1.0")

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="...",
    tracer=otel_tracer,
    meter=create_meter(otel_meter),
)
```

## Files changed (upstream scope)

| File | Change |
|---|---|
| `graphiti_core/meter.py` | **new** — `GraphitiMeter` ABC, `NoOpMeter`, `OpenTelemetryMeter`, `create_meter()` |
| `graphiti_core/graphiti.py` | wire `meter` param; add metric calls; add spans to `search`/`search_`/`remove_episode` |
| `graphiti_core/__init__.py` | re-export meter types |
| `OTEL_TRACING.md` | document the metrics layer alongside existing tracing docs |
| `pyproject.toml` | `tracing` extra already provides `opentelemetry-api`/`-sdk`; no new required dep |

> **Out of scope for this PR** (fork/benchmark-specific, not upstreamed): `server/`, `examples/`,
> `Dockerfile`, `uv.lock`, deploy overlays, and the Dynatrace-specific validation harness in
> `tests/validate_dynatrace.py` (vendor-specific; kept in the fork).

## Testing

- `make check` (Ruff + Pyright + Pytest) passes; no existing tests modified.
- Validated end-to-end: all 5 instrumented operations emit spans and metrics through a local
  OTel Collector into a backend (18 metric data points + 5 spans observed). Existing users with no
  `meter` argument exercise the `NoOpMeter` path — confirmed zero overhead / no new deps.

## Backwards compatibility

- Default `meter=None` → `NoOpMeter`: no behaviour change.
- `opentelemetry-api` remains optional (only pulled by the existing `tracing` extra).
- Instrumentation failures are swallowed and never propagate into memory operations.

## Open questions for reviewers

1. Should `memory.query.result_count` stay a `Histogram` (percentiles) or become an `UpDownCounter`?
2. Are `add_episode` / `search` / ... the desired `memory.operation.name` attribute values, or should
   they map to a specific semconv registry string?
3. Preferred way to surface the `meter` param through the MCP/FastAPI server layers later (this PR
   instruments only the core library)?
