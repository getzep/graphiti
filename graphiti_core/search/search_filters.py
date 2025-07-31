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

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from graphiti_core.driver.driver import GraphDriver


class ComparisonOperator(Enum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='


class DateFilter(BaseModel):
    date: datetime = Field(description='A datetime to filter on')
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for date filter'
    )


class SearchFilters(BaseModel):
    node_labels: list[str] | None = Field(
        default=None, description='List of node labels to filter on'
    )
    edge_types: list[str] | None = Field(
        default=None, description='List of edge types to filter on'
    )
    valid_at: list[list[DateFilter]] | None = Field(default=None)
    invalid_at: list[list[DateFilter]] | None = Field(default=None)
    created_at: list[list[DateFilter]] | None = Field(default=None)
    expired_at: list[list[DateFilter]] | None = Field(default=None)

def helix_map_comparison_operator(operator: ComparisonOperator) -> str:
    return {
        ComparisonOperator.equals: '==',
        ComparisonOperator.not_equals: '!=',
        ComparisonOperator.greater_than: '>',
        ComparisonOperator.less_than: '<',
        ComparisonOperator.greater_than_equal: '>=',
        ComparisonOperator.less_than_equal: '<='
    }[operator]

def helix_date_filter(filters: list[list[DateFilter]], key: str) -> list[list[dict[str, Any]]]:
    or_filters = []
    for or_list in filters:
        and_filters = []
        for date_filter in or_list:
            and_filters.append({
                'key': key,
                'value': date_filter.date.isoformat(),
                'operator': helix_map_comparison_operator(date_filter.comparison_operator)
            })
    
        or_filters.append(and_filters)
    return or_filters

def helix_edge_search_filter_edge_types(
    filters: SearchFilters,
) -> list[dict[str, Any]]:
    properties: list[dict[str, Any]] = []
    
    # Filter name in edge_types
    if filters.edge_types is not None:
        properties.append({'key': 'name', 'value': filters.edge_types, 'operator': '=='})
    
    return properties

async def helix_edge_search_filter_date_filters(
    filters: SearchFilters,
    driver: GraphDriver,
    connection_id: str,
):
    filter_items = []

    # Filter valid_at
    if filters.valid_at is not None:
        filter_items.append(helix_date_filter(filters.valid_at, 'valid_at'))

    # Filter invalid_at
    if filters.invalid_at is not None:
        filter_items.append(helix_date_filter(filters.invalid_at, 'invalid_at'))

    # Filter created_at
    if filters.created_at is not None:
        filter_items.append(helix_date_filter(filters.created_at, 'created_at'))

    # Filter expired_at
    if filters.expired_at is not None:
        filter_items.append(helix_date_filter(filters.expired_at, 'expired_at'))

    for filter_item in filter_items:
        if len(filter_item) > 0:
            await driver.execute_query(
                "",
                query="mcp/filter_items",
                connection_id=connection_id,
                data={'filter': {'properties': filter_item}}
            )

def helix_edge_search_filter_traversals(
    filters: SearchFilters,
    source_node_uuids: list[str] | None = None,
    target_node_uuids: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    source_properties = []
    target_properties = []

    if filters.node_labels is not None and len(filters.node_labels) > 0:
        source_properties.append({'key': 'labels', 'value': filters.node_labels, 'operator': '=='})
        target_properties.append({'key': 'labels', 'value': filters.node_labels, 'operator': '=='})

    if source_node_uuids is not None and len(source_node_uuids) > 0:
        if len(source_node_uuids) == 1:
            source_properties.append({'key': 'uuid', 'value': source_node_uuids[0], 'operator': '=='})
        else:
            source_properties.append({'key': 'uuid', 'value': source_node_uuids, 'operator': '=='})

    if target_node_uuids is not None and len(target_node_uuids) > 0:
        if len(target_node_uuids) == 1:
            target_properties.append({'key': 'uuid', 'value': target_node_uuids[0], 'operator': '=='})
        else:
            target_properties.append({'key': 'uuid', 'value': target_node_uuids, 'operator': '=='})

    # Filter labels in node_labels
    filter_traversals = []

    if len(source_properties) > 0:
        filter_traversals.append(
            {
                'tool_name': 'in_step',
                'args': {
                    'edge_label': 'Entity_Fact',
                    'edge_type': 'node',
                    'filter': {
                        'properties': [
                            source_properties
                        ]
                    }
                }
            }
        )

    if len(target_properties) > 0:
        filter_traversals.append(
            {
                'tool_name': 'out_step',
                'args': {
                    'edge_label': 'Fact_Entity',
                    'edge_type': 'node',
                    'filter': {
                        'properties': [
                            target_properties
                        ]
                    }
                }
            }
        )
    
    return filter_traversals

async def helix_edge_search_filter(
    driver: GraphDriver,
    connection_id: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    source_node_uuid: str | None = None,
    target_node_uuid: str | None = None,
):
    # Add properties to filter facts
    fact_properties = []
    
    # group_id in group_ids
    if group_ids is not None and isinstance(group_ids, list) and len(group_ids) > 0:
        fact_properties.append({
            'key': 'group_id',
            'operator': '==',
            'value': group_ids
        })
    
    # Filter edge types (search_filter)
    fact_properties.extend(helix_edge_search_filter_edge_types(search_filter))

    # Filter traversals (source and target node filters)
    filter_traversals = []

    # Filter source and target node
    source_target_filters = []
    if source_node_uuid is not None:
        source_target_filters.append(source_node_uuid)
    if target_node_uuid is not None:
        source_target_filters.append(target_node_uuid)

    if len(source_target_filters) == 0:
        source_target_filters = None

    search_traversals = helix_edge_search_filter_traversals(search_filter, source_target_filters, source_target_filters)
    if search_traversals is not None:
        filter_traversals.extend(search_traversals)

    # Execute filter items
    args = {}
    if len(filter_traversals) > 0:
        args['filter_traversals'] = filter_traversals
    if len(fact_properties) > 0:
        args['properties'] = [fact_properties]

    if len(args) > 0:
        await driver.execute_query(
            "",
            query="mcp/filter_items",
            connection_id=connection_id,
            data={'filter': args}
        )

    # Filter date (search_filter)
    await helix_edge_search_filter_date_filters(search_filter, driver, connection_id)

async def helix_edge_bfs_search(
    driver: GraphDriver,
    connection_id: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    max_depth: int = 1,
) -> list[str]: 
    results = set()

    # Get all facts edges
    await driver.execute_query("", query="mcp/out_step", connection_id=connection_id, data={'edge_label': 'Entity_Fact', 'edge_type': 'node'})

    await helix_edge_search_filter(driver, connection_id, search_filter, group_ids)

    for depth in range(max_depth):
        await driver.execute_query("", query="mcp/out_step", connection_id=connection_id, data={'edge_label': 'Fact_Entity', 'edge_type': 'node'})

        await driver.execute_query("", query="mcp/out_step", connection_id=connection_id, data={'edge_label': 'Entity_Fact', 'edge_type': 'node'})

        await helix_edge_search_filter(driver, connection_id, search_filter, group_ids)

        res = await driver.execute_query("", query="mcp/collect", connection_id=connection_id, drop=False)

        results.update([edge.get('uuid') for edge in res])
        print(f'depth {depth} : {len(res)}')
        print('--------------------------------')

    await driver.execute_query("", query="mcp/reset", connection_id=connection_id)

    return list(results)

async def helix_node_search_filter(
    driver: GraphDriver,
    connection_id: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
):
    # Add properties to filter facts
    fact_properties = []

    # group_id in group_ids
    if group_ids is not None and isinstance(group_ids, list) and len(group_ids) > 0:
        fact_properties.append({
            'key': 'group_id',
            'operator': '==',
            'value': group_ids
        })
    

    if search_filter.node_labels is not None and len(search_filter.node_labels) > 0:
        fact_properties.append({
            'key': 'labels', 
            'operator': '==',
            'value': search_filter.node_labels
        })

    # Execute filter items
    if len(fact_properties) > 0:
        await driver.execute_query(
            "", 
            query="mcp/filter_items", 
            connection_id=connection_id, 
            data={'filter': {'properties': [fact_properties]}}
        )

    # Filter created_at date
    if search_filter.created_at is not None:
        filter_item = helix_date_filter(search_filter.created_at, 'created_at')
        if len(filter_item) > 0:
            await driver.execute_query(
                "",
                query="mcp/filter_items",
                connection_id=connection_id,
                data={'filter': {'properties': filter_item}}
            )

async def helix_node_bfs_search(
    driver: GraphDriver,
    connection_id: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    max_depth: int = 1,
) -> list[str]:
    results = set()
    
    for depth in range(max_depth):
        await driver.execute_query("", query="mcp/out_step", connection_id=connection_id, data={'edge_label': 'Entity_Fact', 'edge_type': 'node'})

        await driver.execute_query("", query="mcp/out_step", connection_id=connection_id, data={'edge_label': 'Fact_Entity', 'edge_type': 'node'})

        await helix_node_search_filter(driver, connection_id, search_filter, group_ids)

        res = await driver.execute_query("", query="mcp/collect", connection_id=connection_id, drop=False)

        results.update([node.get('uuid') for node in res])
        print(f'depth {depth} : {len(res)}')
        print('--------------------------------')
    
    await driver.execute_query("", query="mcp/reset", connection_id=connection_id)
    
    return list(results)

def node_search_filter_query_constructor(
    filters: SearchFilters,
) -> tuple[str, dict[str, Any]]:
    filter_query: str = ''
    filter_params: dict[str, Any] = {}

    if filters.node_labels is not None:
        node_labels = '|'.join(filters.node_labels)
        node_label_filter = ' AND n:' + node_labels
        filter_query += node_label_filter

    return filter_query, filter_params

def edge_search_filter_query_constructor(
    filters: SearchFilters,
) -> tuple[str, dict[str, Any]]:
    filter_query: str = ''
    filter_params: dict[str, Any] = {}

    if filters.edge_types is not None:
        edge_types = filters.edge_types
        edge_types_filter = '\nAND r.name in $edge_types'
        filter_query += edge_types_filter
        filter_params['edge_types'] = edge_types

    if filters.node_labels is not None:
        node_labels = '|'.join(filters.node_labels)
        node_label_filter = '\nAND n:' + node_labels + ' AND m:' + node_labels
        filter_query += node_label_filter

    if filters.valid_at is not None:
        valid_at_filter = '\nAND ('
        for i, or_list in enumerate(filters.valid_at):
            for j, date_filter in enumerate(or_list):
                filter_params['valid_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.valid_at ' + date_filter.comparison_operator.value + f' $valid_at_{j})'
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            valid_at_filter += and_filter_query

            if i == len(filters.valid_at) - 1:
                valid_at_filter += ')'
            else:
                valid_at_filter += ' OR '

        filter_query += valid_at_filter

    if filters.invalid_at is not None:
        invalid_at_filter = ' AND ('
        for i, or_list in enumerate(filters.invalid_at):
            for j, date_filter in enumerate(or_list):
                filter_params['invalid_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.invalid_at ' + date_filter.comparison_operator.value + f' $invalid_at_{j})'
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            invalid_at_filter += and_filter_query

            if i == len(filters.invalid_at) - 1:
                invalid_at_filter += ')'
            else:
                invalid_at_filter += ' OR '

        filter_query += invalid_at_filter

    if filters.created_at is not None:
        created_at_filter = ' AND ('
        for i, or_list in enumerate(filters.created_at):
            for j, date_filter in enumerate(or_list):
                filter_params['created_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.created_at ' + date_filter.comparison_operator.value + f' $created_at_{j})'
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            created_at_filter += and_filter_query

            if i == len(filters.created_at) - 1:
                created_at_filter += ')'
            else:
                created_at_filter += ' OR '

        filter_query += created_at_filter

    if filters.expired_at is not None:
        expired_at_filter = ' AND ('
        for i, or_list in enumerate(filters.expired_at):
            for j, date_filter in enumerate(or_list):
                filter_params['expired_at_' + str(j)] = date_filter.date

            and_filters = [
                '(r.expired_at ' + date_filter.comparison_operator.value + f' $expired_at_{j})'
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            expired_at_filter += and_filter_query

            if i == len(filters.expired_at) - 1:
                expired_at_filter += ')'
            else:
                expired_at_filter += ' OR '

        filter_query += expired_at_filter

    return filter_query, filter_params
