"""Zep Cloud v2 compatibility layer backed by Graphiti.

See WIRE_SPEC.md for the exact contract this implements and why.

Nothing is imported eagerly on purpose: `models`, `ontology`, and `store` are
pure and must stay importable (and unit-testable) without graphiti_core, a
graph database, or an LLM. Import `graph_service.zep_compat.app:app` to serve.
"""

__all__ = ['models', 'ontology', 'store', 'runtime', 'router', 'app']
