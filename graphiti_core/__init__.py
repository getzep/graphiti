from importlib.metadata import PackageNotFoundError, version

from .graphiti import Graphiti

try:
    __version__ = version('graphiti-core')
except PackageNotFoundError:
    __version__ = 'unknown'

__all__ = ['Graphiti', '__version__']
