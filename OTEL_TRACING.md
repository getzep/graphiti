# OpenTelemetry Tracing in Graphiti

Graphiti supports OpenTelemetry distributed tracing. Tracing is optional - without a tracer, operations use no-op implementations with zero overhead.

## Installation

```bash
uv add opentelemetry-sdk
```

## Basic Usage

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from graphiti_core import Graphiti

# Set up OpenTelemetry
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Get tracer and pass to Graphiti
tracer = trace.get_tracer(__name__)
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    tracer=tracer,
    trace_span_prefix="myapp.graphiti"  # Optional, defaults to "graphiti"
)
```

## With Kuzu (In-Memory)

```python
from graphiti_core.driver.kuzu_driver import KuzuDriver

kuzu_driver = KuzuDriver()
graphiti = Graphiti(graph_driver=kuzu_driver, tracer=tracer)
```

## Metrics

In addition to spans, Graphiti can emit OpenTelemetry metrics for its core memory
operations. Like tracing, metrics are optional — without a meter, a no-op
implementation is used with zero overhead.

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from graphiti_core import Graphiti
from graphiti_core.meter import create_meter

metrics.set_meter_provider(MeterProvider())
otel_meter = metrics.get_meter("graphiti", version="0.1.0")

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    tracer=tracer,
    meter=create_meter(otel_meter),  # omit → NoOpMeter, no overhead
)
```

### Instruments

| Instrument | Kind | Unit | Key attributes |
|---|---|---|---|
| `memory.operation.duration` | Histogram | `ms` | `memory.operation.name` |
| `memory.operation.count` | Counter | `1` | `memory.operation.name`, `memory.operation.status` (`ok`/`error`) |
| `memory.items.stored` | Counter | `1` | `memory.item.type` (`node`/`edge`/`episode`) |
| `memory.items.invalidated` | Counter | `1` | `memory.item.type` |
| `memory.query.result_count` | Histogram | `1` | `memory.operation.name` |

Instrumented operations: `add_episode`, `add_episode_bulk`, `search`, `search_`,
and `remove_episode`. All instrumentation errors are suppressed so telemetry never
interferes with a memory operation.

## Example

See `examples/opentelemetry/` for a complete working example with stdout tracing

