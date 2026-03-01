try:
    from .graphiti import Graphiti

    __all__ = ['Graphiti']
except ImportError as exc:
    # Re-raise if an internal graphiti_core module fails — that indicates a
    # real bug (syntax error, missing internal file, etc.) and must not be
    # silently swallowed.
    if exc.name and exc.name.startswith('graphiti_core.'):
        raise
    # External optional heavy dependencies (python-dotenv, pydantic, neo4j,
    # openai, etc.) are unavailable in this environment.
    # Utility sub-packages (graphiti_core.utils.*) remain importable.
    __all__ = []
