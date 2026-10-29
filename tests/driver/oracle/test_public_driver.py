"""
Copyright 2026, Zep Software, Inc.

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

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.oracle_driver import Oracle26aiConfig, Oracle26aiDriver


def test_oracle_is_a_registered_optional_provider():
    assert GraphProvider.ORACLE.value == 'oracle'
    assert Oracle26aiDriver.provider == GraphProvider.ORACLE
    assert (
        Oracle26aiConfig(dsn='example_low', user='graphiti', password='test-only').embedding_dim
        == 1024
    )


def test_missing_optional_dependency_has_an_actionable_error():
    import subprocess
    import sys

    code = """
import importlib.abc
import sys

class BlockOracle(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'oracledb' or fullname.startswith('oracledb.'):
            raise ModuleNotFoundError('simulated missing optional dependency')

sys.meta_path.insert(0, BlockOracle())
from graphiti_core.driver.driver import GraphProvider
assert GraphProvider.ORACLE.value == 'oracle'
try:
    from graphiti_core.driver.oracle_driver import Oracle26aiDriver
except ImportError as exc:
    assert 'pip install graphiti-core[oracle]' in str(exc)
else:
    raise AssertionError('optional Oracle import unexpectedly succeeded')
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
