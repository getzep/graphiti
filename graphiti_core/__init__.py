try:
    from .graphiti import Graphiti

    __all__ = ['Graphiti']
except ImportError:
    # Optional heavy deps (python-dotenv, pydantic, etc.) are unavailable.
    # Utility sub-packages (graphiti_core.utils.*) remain importable.
    __all__ = []
