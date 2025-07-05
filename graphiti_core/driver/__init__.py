"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from logging import getLogger

logger = getLogger(__name__)

__all__ = []

FALKORDB_AVAILABLE = False
NEO4J_AVAILABLE = False

try:
    # ruff: noqa: F401
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    __all__.append('FalkorDriver')
    FALKORDB_AVAILABLE = True
except ImportError:
    logger.debug(
        "FalkorDB driver not available. Ensure 'falkordb' package is installed.",
        exc_info=True,
    )

try:
    # ruff: noqa: F401
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    __all__.append('Neo4jDriver')
    NEO4J_AVAILABLE = True
except ImportError:
    logger.debug(
        "Neo4j driver not available. Ensure 'neo4j' package is installed.",
        exc_info=True,
    )
