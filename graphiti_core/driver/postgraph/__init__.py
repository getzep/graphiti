"""PostGraph driver package.

The driver is re-exported lazily. postgraph_driver imports this package's
graph_operations_interface, so a plain `from ... import PostGraphDriver` at
module scope made the two import each other: importing the package first
happened to work, importing the driver module directly raised ImportError, and
the tests took the working order so nothing showed it.

PEP 562 module __getattr__ defers the import to first attribute access, by
which point both modules are fully built, so either order works.
"""

from typing import Any

__all__ = ['PostGraphDriver', 'PostGraphDriverSession']


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = __import__('graphiti_core.driver.postgraph_driver', fromlist=[name])
        return getattr(mod, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
