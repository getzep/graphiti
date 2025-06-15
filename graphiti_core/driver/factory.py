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

from .driver import GraphDriver

class GraphDriverFactory:
    @staticmethod
    def get_driver(uri: str, user: str, password: str) -> GraphDriver:
        scheme = uri.lower()
        if scheme.startswith("falkor://"):
            from .falkordb_driver import FalkorDriver
            return FalkorDriver(uri, user, password)
        elif scheme.startswith("neo4j://"):
            from .neo4j_driver import Neo4jDriver
            return Neo4jDriver(uri, user, password)
        else:
            raise ValueError(f"Unknown database URI scheme in {uri}")
