"""Write-path extension hooks for graphiti-core.

This module provides an explicit extension seam for downstream applications that
need to customize add_episode's write path without monkey-patching graphiti-core
module globals. Hooks are optional and fail-open by default: if no hook is
registered, graphiti-core behaves like upstream.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class WritePathContext:
    """Per-add_episode context shared by write-path hooks.

    Downstream callers own the shape of most values. graphiti-core only stores
    and passes the context through; hook implementations interpret config and
    stats according to their own contracts.
    """

    identity_hints: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    driver: Any | None = None
    edge_similarity_scores: dict[str, float] = field(default_factory=dict)
    edge_fact_query_vectors: dict[str, Any] = field(default_factory=dict)


_WRITE_PATH_CONTEXT: ContextVar[WritePathContext | None] = ContextVar(
    'graphiti_write_path_context', default=None
)
_HOOKS: dict[str, Any] = {}


def set_write_path_context(
    *,
    identity_hints: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    driver: Any | None = None,
) -> Token[WritePathContext | None]:
    """Set per-episode write-path context and return a reset token."""

    return _WRITE_PATH_CONTEXT.set(
        WritePathContext(
            identity_hints=[] if identity_hints is None else identity_hints,
            config={} if config is None else config,
            stats={} if stats is None else stats,
            driver=driver,
        )
    )


def reset_write_path_context(token: Token[WritePathContext | None]) -> None:
    """Reset the write-path context returned by set_write_path_context()."""

    _WRITE_PATH_CONTEXT.reset(token)


@contextmanager
def write_path_context(**kwargs: Any):
    """Context manager wrapper around set/reset_write_path_context."""

    token = set_write_path_context(**kwargs)
    try:
        yield get_write_path_context()
    finally:
        reset_write_path_context(token)


def get_write_path_context() -> WritePathContext | None:
    """Return the current write-path context, if any."""

    return _WRITE_PATH_CONTEXT.get()


def register_hook(name: str, hook: Any) -> None:
    """Register or replace a write-path hook implementation."""

    _HOOKS[name] = hook


def unregister_hook(name: str) -> None:
    """Remove a hook implementation if registered."""

    _HOOKS.pop(name, None)


def get_hook(name: str) -> Any | None:
    """Return a hook implementation by name."""

    return _HOOKS.get(name)


def clear_hooks() -> None:
    """Clear all registered hooks. Primarily useful in tests."""

    _HOOKS.clear()


@runtime_checkable
class NodeResolutionHook(Protocol):
    async def resolve_extracted_nodes(
        self,
        original: Any,
        *args: Any,
        context: WritePathContext,
        **kwargs: Any,
    ) -> Any:
        """Resolve extracted nodes, delegating to original when appropriate."""


@runtime_checkable
class NodeAttributeHook(Protocol):
    async def extract_attributes_from_nodes(
        self,
        original: Any,
        *args: Any,
        context: WritePathContext,
        **kwargs: Any,
    ) -> Any:
        """Extract or skip attributes for resolved nodes."""


@runtime_checkable
class CandidateNodeHook(Protocol):
    async def collect_candidate_nodes(
        self,
        original: Any,
        *args: Any,
        context: WritePathContext,
        **kwargs: Any,
    ) -> Any:
        """Collect/filter candidate nodes."""


@runtime_checkable
class EdgeResolutionHook(Protocol):
    async def resolve_extracted_edge(
        self,
        original: Any,
        *args: Any,
        context: WritePathContext,
        **kwargs: Any,
    ) -> Any:
        """Resolve an extracted edge, delegating to original when appropriate."""


@runtime_checkable
class EdgeSimilaritySearchHook(Protocol):
    async def filter_edges(
        self,
        edges: list[Any],
        *,
        driver: Any,
        search_vector: Any,
        group_ids: list[str] | None,
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        record_scores: dict[str, float | None] | None,
        context: WritePathContext,
    ) -> list[Any]:
        """Filter/rank edge similarity search results before they reach dedupe."""
        ...
