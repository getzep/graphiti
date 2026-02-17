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

import json
import os
import sqlite3
import typing


class LLMCache:
    """Simple SQLite + JSON cache for LLM responses.

    Replaces diskcache to avoid unsafe pickle deserialization (CVE in diskcache <= 5.6.3).
    Only stores JSON-serializable data.
    """

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        db_path = os.path.join(directory, 'cache.db')
        self._conn = sqlite3.connect(db_path)
        self._conn.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)')
        self._conn.commit()

    def get(self, key: str) -> dict[str, typing.Any] | None:
        row = self._conn.execute('SELECT value FROM cache WHERE key = ?', (key,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: dict[str, typing.Any]) -> None:
        self._conn.execute(
            'INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)',
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
