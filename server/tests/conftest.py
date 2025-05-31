"""
Configuration for pytest for Graphiti server tests.
"""
import os
import sys
import pytest

# Add the server directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
