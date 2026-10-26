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

import asyncio
import os
import re
from uuid import uuid4

import pytest

from tests.driver.oracle.conformance.backends import (
    close_surface,
    falkor_surface_from_env,
    oracle_surface_from_env,
    reference_surface,
)
from tests.driver.oracle.conformance.scenarios import run_core_scenario

TEST_NAMESPACE = os.environ.get('ORACLE_TEST_NAMESPACE', 'graphiti-oracle-' + uuid4().hex)


def scoped_group_id(name: str) -> str:
    namespace = re.sub(
        r'[^A-Za-z0-9_-]',
        '-',
        TEST_NAMESPACE,
    )[:64]
    return f'{namespace}-{name}'[:240]


@pytest.mark.asyncio
async def test_reference_scenario_pins_the_normalized_contract() -> None:
    surface = reference_surface()

    result = await run_core_scenario(surface, 'conformance-reference')

    assert [node['name'] for node in result.nodes] == ['Alpha', 'Beta']
    assert result.edges[0]['fact'] == 'Alpha is related to Beta.'
    assert [node['name'] for node in result.reachable] == ['Beta']
    assert result.group_id == 'conformance-reference'
    assert result.evidence_kind == 'hermetic-or-live-backend'


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_oracle_driver_matches_the_reference_contract() -> None:
    group_id = scoped_group_id('conformance-oracle')
    reference = await run_core_scenario(reference_surface(), group_id)
    oracle = await oracle_surface_from_env()
    try:
        actual = await run_core_scenario(oracle, group_id)
        assert actual == reference
    finally:
        try:
            await oracle.driver.graph_ops.clear_data(oracle.driver, [group_id])
        finally:
            await close_surface(oracle)


@pytest.mark.falkor
@pytest.mark.asyncio
async def test_falkor_driver_matches_the_reference_contract() -> None:
    reference = await run_core_scenario(reference_surface(), 'conformance-falkor')
    falkor = await falkor_surface_from_env()
    try:
        actual = await run_core_scenario(falkor, 'conformance-falkor')
        assert actual == reference
    finally:
        await close_surface(falkor)


@pytest.mark.oracle
@pytest.mark.falkor
@pytest.mark.asyncio
async def test_oracle_and_falkor_normalize_to_the_same_contract() -> None:
    group_id = scoped_group_id('conformance-differential')
    oracle = await oracle_surface_from_env()
    falkor = await falkor_surface_from_env()
    try:
        oracle_result, falkor_result = await asyncio.gather(
            run_core_scenario(oracle, group_id),
            run_core_scenario(falkor, group_id),
        )
        assert oracle_result == falkor_result
    finally:
        try:
            await oracle.driver.graph_ops.clear_data(oracle.driver, [group_id])
        finally:
            await asyncio.gather(close_surface(oracle), close_surface(falkor))


@pytest.mark.oracle
@pytest.mark.asyncio
async def test_one_oracle_driver_isolates_twenty_concurrent_groups() -> None:
    oracle = await oracle_surface_from_env()
    groups = [scoped_group_id(f'concurrency-{index:02d}') for index in range(20)]
    try:
        results = await asyncio.gather(
            *(run_core_scenario(oracle, group_id) for group_id in groups)
        )
        assert {result.group_id for result in results} == set(groups)
        assert all(
            {node['group_id'] for node in result.nodes} == {result.group_id} for result in results
        )
    finally:
        try:
            await oracle.driver.graph_ops.clear_data(oracle.driver, groups)
        finally:
            await close_surface(oracle)


pytestmark = pytest.mark.integration
