# OpenTelemetry Tracing in Graphiti

## Overview

Graphiti supports OpenTelemetry distributed tracing through dependency injection. Tracing is optional - without a tracer, operations are no-op with zero overhead.

## Installation

To use tracing, install the OpenTelemetry SDK:

```bash
pip install opentelemetry-sdk
```

## Basic Usage

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Set up OpenTelemetry
provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Get a tracer
tracer = trace.get_tracer(__name__)

# Create Graphiti with tracing enabled
from graphiti_core import Graphiti

graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    tracer=tracer,
    trace_span_prefix="myapp.graphiti"  # Optional, defaults to "graphiti"
)
```

## Configuration

### Span Name Prefix

You can configure the prefix for all span names:

```python
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    tracer=tracer,
    trace_span_prefix="myapp.kg"
)
```

