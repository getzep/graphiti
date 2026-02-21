from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('graphiti-core')
except PackageNotFoundError:
    __version__ = 'unknown'

from .graphiti import Graphiti

__all__ = ['Graphiti', '__version__']
