from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

from .graphiti import Graphiti

try:
    __version__ = _get_version('graphiti-core')
except PackageNotFoundError:
    # Package metadata is unavailable when running from a source checkout
    # that has not been installed (e.g. `PYTHONPATH=.` without `pip install`).
    __version__ = 'unknown'

__all__ = ['Graphiti', '__version__']
