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
from typing import Any, LiteralString

from pydantic import BaseModel, Field


class DateFilter(BaseModel):
    date_match: datetime | None = Field(default=None, description='An exact datetime to filter on')
    date_range: tuple[datetime, datetime] | None = Field(
        default=None, description='A date range to filter on'
    )
    invert_filter: bool = Field(
        default=False,
        description='Invert the filter. This will add a NOT to the conditions if true',
    )


class SearchFilters(BaseModel):
    valid_at: DateFilter | None = Field(default=None)
    invalid_at: DateFilter | None = Field(default=None)
    created_at: DateFilter | None = Field(default=None)
    expired_at: DateFilter | None = Field(default=None)


def search_filter_query_constructor(filters: SearchFilters) -> tuple[LiteralString, dict[str, Any]]:
    filter_query: LiteralString = ''
    filter_params: dict[str, Any] = {}

    if filters.valid_at is not None:
        if filters.valid_at.date_match is not None and not filters.valid_at.invert_filter:
            filter_query += 'AND r.valid_at = $valid_at'
            filter_params['valid_at'] = filters.valid_at.date_match
        if filters.valid_at.date_match is not None and filters.valid_at.invert_filter:
            filter_query += 'AND r.valid_at <> $valid_at'
            filter_params['valid_at'] = filters.valid_at.date_match

        if filters.valid_at.date_range is not None and not filters.valid_at.invert_filter:
            filter_query += 'AND r.valid_at > $valid_at_start AND r.valid_at < $valid_at_stop'
            filter_params['valid_at_start'] = filters.valid_at.date_range[0]
            filter_params['valid_at_stop'] = filters.valid_at.date_range[1]
        if filters.valid_at.date_range is not None and filters.valid_at.invert_filter:
            filter_query += 'AND r.valid_at < $valid_at_start OR r.valid_at > $valid_at_stop'
            filter_params['valid_at_start'] = filters.valid_at.date_range[0]
            filter_params['valid_at_stop'] = filters.valid_at.date_range[1]

    if filters.invalid_at is not None:
        if filters.invalid_at.date_match is not None and not filters.invalid_at.invert_filter:
            filter_query += 'AND r.invalid_at = $invalid_at'
            filter_params['invalid_at'] = filters.invalid_at.date_match
        if filters.invalid_at.date_match is not None and filters.invalid_at.invert_filter:
            filter_query += 'AND r.invalid_at <> $invalid_at'
            filter_params['invalid_at'] = filters.invalid_at.date_match

        if filters.invalid_at.date_range is not None and not filters.invalid_at.invert_filter:
            filter_query += (
                'AND r.invalid_at > $invalid_at_start AND r.invalid_at < $invalid_at_stop'
            )
            filter_params['invalid_at_start'] = filters.invalid_at.date_range[0]
            filter_params['invalid_at_stop'] = filters.invalid_at.date_range[1]
        if filters.invalid_at.date_range is not None and filters.invalid_at.invert_filter:
            filter_query += (
                'AND r.invalid_at < $invalid_at_start OR r.invalid_at > $invalid_at_stop'
            )
            filter_params['invalid_at_start'] = filters.invalid_at.date_range[0]
            filter_params['invalid_at_stop'] = filters.invalid_at.date_range[1]

    if filters.created_at is not None:
        if filters.created_at.date_match is not None and not filters.created_at.invert_filter:
            filter_query += 'AND r.created_at = $created_at'
            filter_params['created_at'] = filters.created_at.date_match
        if filters.created_at.date_match is not None and filters.created_at.invert_filter:
            filter_query += 'AND r.created_at <> $created_at'
            filter_params['created_at'] = filters.created_at.date_match

        if filters.created_at.date_range is not None and not filters.created_at.invert_filter:
            filter_query += (
                'AND r.created_at > $created_at_start AND r.created_at < $created_at_stop'
            )
            filter_params['created_at_start'] = filters.created_at.date_range[0]
            filter_params['created_at_stop'] = filters.created_at.date_range[1]
        if filters.created_at.date_range is not None and filters.created_at.invert_filter:
            filter_query += (
                'AND r.created_at < $created_at_start OR r.created_at > $created_at_stop'
            )
            filter_params['created_at_start'] = filters.created_at.date_range[0]
            filter_params['created_at_stop'] = filters.created_at.date_range[1]

    if filters.expired_at is not None:
        if filters.expired_at.date_match is not None and not filters.expired_at.invert_filter:
            filter_query += 'AND r.expired_at = $expired_at'
            filter_params['expired_at'] = filters.expired_at.date_match
        if filters.expired_at.date_match is not None and filters.expired_at.invert_filter:
            filter_query += 'AND r.expired_at <> $expired_at'
            filter_params['expired_at'] = filters.expired_at.date_match

        if filters.expired_at.date_range is not None and not filters.expired_at.invert_filter:
            filter_query += (
                'AND r.expired_at > $expired_at_start AND r.expired_at < $expired_at_stop'
            )
            filter_params['expired_at_start'] = filters.expired_at.date_range[0]
            filter_params['expired_at_stop'] = filters.expired_at.date_range[1]
        if filters.expired_at.date_range is not None and filters.expired_at.invert_filter:
            filter_query += (
                'AND r.expired_at < $expired_at_start OR r.expired_at > $expired_at_stop'
            )
            filter_params['expired_at_start'] = filters.expired_at.date_range[0]
            filter_params['expired_at_stop'] = filters.expired_at.date_range[1]

    return filter_query, filter_params
