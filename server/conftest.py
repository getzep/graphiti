import os
import sys
import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

pytest_plugins = ('pytest_asyncio',)
