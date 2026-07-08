# OpenTelemetry Stdout Tracing Example

Configure Graphiti with OpenTelemetry to output trace spans to stdout.

## Setup

```bash
uv sync
export OPENAI_API_KEY=your_api_key_here
uv run otel_stdout_example.py
```

## Configure OpenTelemetry with Graphiti

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Set up OpenTelemetry with stdout exporter
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Get tracer and pass to Graphiti
tracer = trace.get_tracer(__name__)
graphiti = Graphiti(
    graph_driver=kuzu_driver,
    tracer=tracer,
    trace_span_prefix='graphiti.example'
)
```
