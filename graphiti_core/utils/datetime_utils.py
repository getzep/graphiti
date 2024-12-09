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

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Returns the current UTC datetime with timezone information."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """
    Ensures a datetime is timezone-aware and in UTC.
    If the datetime is naive (no timezone), assumes it's in UTC.
    If the datetime has a different timezone, converts it to UTC.
    Returns None if input is None.
    """
    if dt is None:
        return None

    if dt.tzinfo is None:
        # If datetime is naive, assume it's UTC
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        # If datetime has a different timezone, convert to UTC
        return dt.astimezone(timezone.utc)

    return dt
