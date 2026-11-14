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

import pytest
from pydantic import ValidationError

from graphiti_core.driver.oracle.config import Oracle26aiConfig


def test_config_rejects_wrong_embedding_dimension() -> None:
    with pytest.raises(ValidationError, match='embedding_dim must be 1024'):
        Oracle26aiConfig(
            dsn='myatp_low',
            user='app',
            password='secret',
            embedding_dim=1536,
        )


def test_config_rejects_untrusted_graph_identifier() -> None:
    with pytest.raises(ValidationError, match='graph_name'):
        Oracle26aiConfig(
            dsn='myatp_low',
            user='app',
            password='secret',
            graph_name='graphiti_graph; drop table users',
        )


def test_config_is_immutable_and_redacts_passwords() -> None:
    config = Oracle26aiConfig(
        dsn='myatp_low',
        user='app',
        password='database-secret',
        wallet_password='wallet-secret',
    )

    with pytest.raises(ValidationError, match='frozen'):
        config.user = 'other'
    assert 'database-secret' not in repr(config)
    assert 'wallet-secret' not in repr(config)
