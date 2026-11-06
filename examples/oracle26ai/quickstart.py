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

# Provision the driver schema on an explicitly configured dedicated Oracle test service.

import asyncio
import os

from pydantic import SecretStr

from graphiti_core.driver.oracle_driver import Oracle26aiConfig, Oracle26aiDriver


async def main() -> None:
    wallet_password = os.environ.get('ORACLE_WALLET_PASSWORD')
    driver = Oracle26aiDriver(
        Oracle26aiConfig(
            dsn=os.environ['ORACLE_DSN'],
            user=os.environ['ORACLE_USER'],
            password=SecretStr(os.environ['ORACLE_PASSWORD']),
            config_dir=os.environ.get('ORACLE_CONFIG_DIR'),
            wallet_location=os.environ.get('ORACLE_WALLET_DIR'),
            wallet_password=SecretStr(wallet_password) if wallet_password else None,
        )
    )
    try:
        await driver.build_indices_and_constraints()
        print((await driver.schema_readiness()).model_dump_json())
    finally:
        await driver.close()


if __name__ == '__main__':
    asyncio.run(main())
