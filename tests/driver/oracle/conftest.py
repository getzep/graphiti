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

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import pytest_asyncio

oracledb = pytest.importorskip('oracledb')


def pytest_configure(config):
    config.addinivalue_line('markers', 'oracle: requires an explicitly enabled Oracle test service')
    config.addinivalue_line(
        'markers', 'falkor: requires an explicitly enabled FalkorDB test service'
    )


def pytest_collection_modifyitems(items):
    for item in items:
        if not item.path.is_relative_to(Path(__file__).parent):
            continue
        if item.get_closest_marker('oracle') and os.environ.get('RUN_ORACLE_TESTS') != '1':
            item.add_marker(
                pytest.mark.skip(reason='set RUN_ORACLE_TESTS=1 for native Oracle tests')
            )
        if item.get_closest_marker('falkor') and os.environ.get('RUN_FALKOR_TESTS') != '1':
            item.add_marker(
                pytest.mark.skip(reason='set RUN_FALKOR_TESTS=1 for differential tests')
            )


@pytest_asyncio.fixture
async def oracle_connection():
    if os.environ.get('RUN_ORACLE_TESTS') != '1':
        pytest.skip('set RUN_ORACLE_TESTS=1 for native Oracle tests')
    if not os.environ.get('ORACLE_PASSWORD'):
        pytest.skip('Oracle test connection is not configured')
    connection = await oracledb.connect_async(
        user=os.environ.get('ORACLE_USER', 'graphiti'),
        password=os.environ['ORACLE_PASSWORD'],
        dsn=os.environ['ORACLE_DSN'],
        config_dir=os.environ.get('ORACLE_CONFIG_DIR'),
        wallet_location=os.environ.get('ORACLE_WALLET_DIR'),
        wallet_password=os.environ.get('ORACLE_WALLET_PASSWORD'),
    )
    try:
        yield connection
    finally:
        await connection.rollback()
        await connection.close()


@pytest_asyncio.fixture
async def live_driver(request):
    from tests.driver.oracle.operations.test_live_surface_int import live_driver_fixture

    async with asynccontextmanager(live_driver_fixture)(request) as value:
        yield value
