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

## With Ladybug (In-Memory)

```python
from graphiti_core.driver.ladybug_driver import LadybugDriver

ladybug_driver = LadybugDriver()
graphiti = Graphiti(graph_driver=ladybug_driver, tracer=tracer)
```

## Example

See `examples/opentelemetry/` for a complete working example with stdout tracing
