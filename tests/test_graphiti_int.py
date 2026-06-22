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

import logging
import sys

import pytest

from graphiti_core.graphiti import Graphiti
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from graphiti_core.search.search_helpers import search_results_to_context_string
from graphiti_core.utils.datetime_utils import utc_now
from tests.helpers_test import GraphProvider

pytestmark = pytest.mark.integration
pytest_plugins = ('pytest_asyncio',)


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logging level to INFO

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


@pytest.mark.asyncio
async def test_graphiti_init(graph_driver):
    if graph_driver.provider == GraphProvider.FALKORDB:
        pytest.skip('Skipping as tests fail on Falkordb')

    logger = setup_logging()
    graphiti = Graphiti(graph_driver=graph_driver)

    await graphiti.build_indices_and_constraints()

    search_filter = SearchFilters(
        node_labels=['Person', 'City'],
        created_at=[
            [DateFilter(date=None, comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=utc_now(), comparison_operator=ComparisonOperator.less_than)],
            [DateFilter(date=None, comparison_operator=ComparisonOperator.is_not_null)],
        ],
    )

    results = await graphiti.search_(
        query='Who is Tania',
        search_filter=search_filter,
    )

    pretty_results = search_results_to_context_string(results)
    logger.info(pretty_results)

    await graphiti.close()
