"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import Any

try:
    from opentelemetry.metrics import Counter, Histogram, Meter

    OTEL_METRICS_AVAILABLE = True
except ImportError:
    OTEL_METRICS_AVAILABLE = False


class GraphitiMeter(ABC):
    """Abstract base class for Graphiti meters.

    Mirrors the NoOp/OTel dual-impl discipline used in tracer.py.
    All metric methods are fire-and-forget; errors are suppressed silently.
    """

    @abstractmethod
    def record_operation_duration(
        self,
        operation: str,
        duration_ms: float,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record the wall-clock duration of a memory operation in milliseconds."""

    @abstractmethod
    def increment_operation_count(
        self,
        operation: str,
        success: bool = True,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Increment the total count of memory operations."""

    @abstractmethod
    def record_items_stored(
        self,
        item_type: str,
        count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record the number of items (nodes/edges/episodes) written to the graph."""

    @abstractmethod
    def record_items_invalidated(
        self,
        item_type: str,
        count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record the number of items invalidated/deleted from the graph."""

    @abstractmethod
    def record_query_results(
        self,
        result_count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record the number of results returned from a search/query operation."""


class NoOpMeter(GraphitiMeter):
    """No-op meter implementation that does nothing."""

    def record_operation_duration(
        self,
        operation: str,
        duration_ms: float,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass

    def increment_operation_count(
        self,
        operation: str,
        success: bool = True,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass

    def record_items_stored(
        self,
        item_type: str,
        count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass

    def record_items_invalidated(
        self,
        item_type: str,
        count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass

    def record_query_results(
        self,
        result_count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass


class OpenTelemetryMeter(GraphitiMeter):
    """OTel metrics implementation aligned with memory-semconv v0.1.0.

    Metric names follow the pattern: ``memory.<signal>``
    Instrument names and units match the semconv attribute registry.
    """

    def __init__(self, meter: 'Meter') -> None:
        if not OTEL_METRICS_AVAILABLE:
            raise ImportError(
                'opentelemetry-api is not installed. Install it with: pip install opentelemetry-api'
            )
        self._meter = meter

        # Histograms (latency / result sizes)
        self._operation_duration: Histogram = meter.create_histogram(
            name='memory.operation.duration',
            unit='ms',
            description='Wall-clock duration of a Graphiti memory operation.',
        )
        self._query_results: Histogram = meter.create_histogram(
            name='memory.query.result_count',
            unit='1',
            description='Number of items returned by a memory query/search.',
        )

        # Counters (monotonically increasing)
        self._operation_count: Counter = meter.create_counter(
            name='memory.operation.count',
            unit='1',
            description='Total number of Graphiti memory operations.',
        )
        self._items_stored: Counter = meter.create_counter(
            name='memory.items.stored',
            unit='1',
            description='Total number of graph items (nodes/edges/episodes) written.',
        )
        self._items_invalidated: Counter = meter.create_counter(
            name='memory.items.invalidated',
            unit='1',
            description='Total number of graph items invalidated or deleted.',
        )

    def record_operation_duration(
        self,
        operation: str,
        duration_ms: float,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        try:
            attrs = {'memory.operation.name': operation, **(attributes or {})}
            self._operation_duration.record(duration_ms, attrs)
        except Exception:
            pass

    def increment_operation_count(
        self,
        operation: str,
        success: bool = True,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        try:
            attrs = {
                'memory.operation.name': operation,
                'memory.operation.status': 'ok' if success else 'error',
                **(attributes or {}),
            }
            self._operation_count.add(1, attrs)
        except Exception:
            pass

    def record_items_stored(
        self,
        item_type: str,
        count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        try:
            if count > 0:
                attrs = {'memory.item.type': item_type, **(attributes or {})}
                self._items_stored.add(count, attrs)
        except Exception:
            pass

    def record_items_invalidated(
        self,
        item_type: str,
        count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        try:
            if count > 0:
                attrs = {'memory.item.type': item_type, **(attributes or {})}
                self._items_invalidated.add(count, attrs)
        except Exception:
            pass

    def record_query_results(
        self,
        result_count: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        with contextlib.suppress(Exception):
            self._query_results.record(result_count, attributes or {})


def create_meter(otel_meter: Any | None = None) -> GraphitiMeter:
    """
    Create a meter instance.

    Parameters
    ----------
    otel_meter : opentelemetry.metrics.Meter | None, optional
        An OpenTelemetry Meter instance. If None, a no-op meter is returned.

    Returns
    -------
    GraphitiMeter
        A meter instance (either OpenTelemetryMeter or NoOpMeter).

    Examples
    --------
    Using with OpenTelemetry:

    >>> from opentelemetry import metrics
    >>> otel_meter = metrics.get_meter('graphiti', version='0.1.0')
    >>> meter = create_meter(otel_meter)

    Using no-op meter:

    >>> meter = create_meter()  # Returns NoOpMeter
    """
    if otel_meter is None:
        return NoOpMeter()

    if not OTEL_METRICS_AVAILABLE:
        return NoOpMeter()

    return OpenTelemetryMeter(otel_meter)
