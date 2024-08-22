import os
import sys

# This code adds the project root directory to the Python path, allowing imports to work correctly when running tests.
# Without this file, you might encounter ModuleNotFoundError when trying to import modules from your project, especially when running tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
