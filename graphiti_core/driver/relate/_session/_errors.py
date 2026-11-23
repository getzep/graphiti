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


class CoreError(Exception):
    """Base class for errors defined by relate-session."""


class StoreClosedError(CoreError):
    """Raised when work is requested after store shutdown has begun."""


class LeaseOwnershipError(CoreError):
    """Raised when a task attempts to use another task's exclusive lease."""


class ReadOnlyStoreError(CoreError):
    """Raised when transaction write scope is requested on a snapshot."""


class TransactionStateError(CoreError):
    """Raised for an invalid transaction transition or failed recovery."""


class TransactionOutcomeIndeterminateError(CoreError):
    """Raised when the native outcome of a transaction control call is unknown."""


class ResultNormalizationError(CoreError):
    """Raised when a native result cannot be detached without losing meaning."""

    def __init__(self, location: str, value_type: type | str) -> None:
        type_name = value_type if isinstance(value_type, str) else value_type.__name__
        super().__init__(f'Cannot normalize value of type {type_name} at {location}')
