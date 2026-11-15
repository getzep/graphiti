# OpenTelemetry Stdout Tracing Example

Configure Graphiti with OpenTelemetry to output trace spans to stdout.

## Setup

A running Neo4j instance is required. Set the Neo4j connection variables if the
instance does not use the default connection values.

```bash
uv sync
export OPENAI_API_KEY=your_api_key_here
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
uv run otel_stdout_example.py
```

## Configure OpenTelemetry with Graphiti

```python
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from graphiti_core import Graphiti

# Set up OpenTelemetry with stdout exporter
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Get tracer and pass to Graphiti
tracer = trace.get_tracer(__name__)
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    tracer=tracer,
    trace_span_prefix='graphiti.example',
)
```
