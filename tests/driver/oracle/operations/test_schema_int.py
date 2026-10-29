# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from graphiti_core.driver.oracle.migrations import MigrationRunner
from graphiti_core.driver.oracle.schema import SchemaInspector
from graphiti_core.driver.oracle_driver import Oracle26aiConfig, Oracle26aiDriver


@pytest.mark.oracle
async def test_schema_apply_is_idempotent_against_oracle_26ai() -> None:
    root = Path(__file__).parents[4]
    load_dotenv(root / '.env.example')
    driver = Oracle26aiDriver(
        Oracle26aiConfig(
            dsn=os.environ.get('ORACLE_DSN', 'myatp_low'),
            user=os.environ.get('ORACLE_USER', 'admin'),
            password=os.environ['ORACLE_PASSWORD'],
            config_dir=str(root / os.environ['ORACLE_CONFIG_DIR']),
            wallet_location=str(root / os.environ['ORACLE_WALLET_DIR']),
            wallet_password=os.environ['ORACLE_WALLET_PASSWORD'],
        )
    )
    try:
        runner = MigrationRunner(driver, root / 'graphiti_core/driver/oracle/sql/migrations')
        await runner.apply()
        await runner.apply()
        assert (await SchemaInspector(driver).readiness()).ready is True
    finally:
        await driver.close()


pytestmark = [pytest.mark.integration, pytest.mark.oracle]
