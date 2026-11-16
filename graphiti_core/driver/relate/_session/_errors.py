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
