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

# Fixtures for the Spanner driver tests.
#
# Hermetic tests need nothing. Integration tests (``test_*_int.py``) use ``spanner_driver``,
# which needs the Spanner emulator (``SPANNER_EMULATOR_HOST=localhost:9010``) or a real
# instance (``SPANNER_PROJECT`` + ``SPANNER_INSTANCE``); without either they are skipped.
# Each test session gets a fresh database ``graphiti_<8 hex>`` that is dropped afterwards.

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress

import pytest
import pytest_asyncio
from google.api_core.exceptions import AlreadyExists

from graphiti_core.driver.spanner import SpannerConfig, SpannerDriver

TEST_EMBEDDING_DIM = 384
DDL_TIMEOUT_SECONDS = 600


def spanner_integration_enabled() -> bool:
    return bool(os.getenv('SPANNER_EMULATOR_HOST') or os.getenv('SPANNER_PROJECT'))


def new_database_id() -> str:
    """A valid Spanner database id: starts with a letter, at most 30 characters."""
    return f'graphiti_{uuid.uuid4().hex[:8]}'


def ensure_instance(driver: SpannerDriver) -> None:
    """Create the instance on the emulator if it is missing. Real instances must exist."""
    if driver.instance.exists():
        return
    if not driver.config.emulator_host:
        pytest.fail(f'Spanner instance {driver.config.instance} does not exist')
    instance = driver.client.instance(
        driver.config.instance,
        configuration_name=f'projects/{driver.config.project}/instanceConfigs/emulator-config',
        node_count=1,
    )
    with suppress(AlreadyExists):
        instance.create().result(120)


def create_database(driver: SpannerDriver) -> None:
    ensure_instance(driver)
    driver.database.create().result(DDL_TIMEOUT_SECONDS)


@pytest.fixture(scope='session')
def spanner_config() -> SpannerConfig:
    if not spanner_integration_enabled():
        pytest.skip('Spanner integration tests need SPANNER_EMULATOR_HOST or SPANNER_PROJECT')
    return SpannerConfig(
        project=os.getenv('SPANNER_PROJECT') or 'test-project',
        instance=os.getenv('SPANNER_INSTANCE', 'test-instance'),
        database=new_database_id(),
        emulator_host=os.getenv('SPANNER_EMULATOR_HOST'),
        embedding_dim=TEST_EMBEDDING_DIM,
    )


@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def spanner_driver(spanner_config: SpannerConfig) -> AsyncIterator[SpannerDriver]:
    driver = SpannerDriver(spanner_config)
    create_database(driver)
    try:
        await driver.build_indices_and_constraints()
        yield driver
    finally:
        await driver.close()
        with suppress(Exception):
            driver.database.drop()


@pytest_asyncio.fixture(loop_scope='session')
async def clean_driver(spanner_driver: SpannerDriver) -> SpannerDriver:
    """The shared driver with every table emptied first, for tests that write data."""
    await spanner_driver.clear_tables()
    return spanner_driver


@pytest_asyncio.fixture(loop_scope='session')
async def scratch_driver(spanner_driver: SpannerDriver) -> AsyncIterator[SpannerDriver]:
    """A driver on its own fresh, empty database (no schema), dropped after the test."""
    driver = spanner_driver.with_database(new_database_id())
    create_database(driver)
    try:
        yield driver
    finally:
        with suppress(Exception):
            driver.database.drop()
