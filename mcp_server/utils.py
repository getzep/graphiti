"""Utility functions and decorators for improving robustness and observability."""

import asyncio
import functools
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional, Type, TypeVar, cast

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic function types
F = TypeVar('F', bound=Callable[..., Any])

class RetryableError(Exception):
    """Base class for errors that can be retried."""
    pass

class PermanentError(Exception):
    """Base class for errors that should not be retried."""
    pass

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker pattern implementation.
    
    Tracks failures and opens circuit when threshold is exceeded.
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open

    def record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failures} failures")

    def record_success(self):
        """Record a success and reset the circuit breaker."""
        self.failures = 0
        self.state = "closed"

    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit state."""
        if self.state == "closed":
            return True
        
        now = time.time()
        if self.state == "open":
            # Check if enough time has passed to try half-open
            if now - self.last_failure_time >= self.reset_timeout:
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
        
        # In half-open state
        if now - self.last_failure_time >= self.half_open_timeout:
            return True
        return False

def with_circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    half_open_timeout: float = 30.0
) -> Callable[[F], F]:
    """Decorator to apply circuit breaker pattern to a function.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Time in seconds before attempting reset
        half_open_timeout: Time in seconds in half-open state
    """
    breaker = CircuitBreaker(failure_threshold, reset_timeout, half_open_timeout)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not breaker.can_execute():
                raise CircuitBreakerError("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
                
        return cast(F, wrapper)
    return decorator

def with_retry(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (RetryableError,),
) -> Callable[[F], F]:
    """Decorator to retry functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        retryable_exceptions: Tuple of exception types to retry
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Final retry attempt failed for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.1f}s. Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(
                        f"Non-retryable error in {func.__name__}",
                        exc_info=True
                    )
                    raise
            
            # Should never reach here due to raise in final attempt
            assert last_exception is not None
            raise last_exception
            
        return cast(F, wrapper)
    return decorator

def with_logging(
    include_args: bool = True,
    log_result: bool = False,
    truncate_length: int = 1000
) -> Callable[[F], F]:
    """Decorator to add structured logging to functions.
    
    Args:
        include_args: Whether to log function arguments
        log_result: Whether to log function result
        truncate_length: Maximum length for logged strings
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate request ID if not present in kwargs
            request_id = kwargs.pop('request_id', str(uuid.uuid4()))
            
            # Prepare log context
            context = {
                'request_id': request_id,
                'function': func.__name__,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add arguments if requested
            if include_args:
                # Truncate long strings in args/kwargs
                safe_args = [
                    str(arg)[:truncate_length] + '...' if isinstance(arg, str) and len(str(arg)) > truncate_length
                    else arg
                    for arg in args
                ]
                safe_kwargs = {
                    k: v[:truncate_length] + '...' if isinstance(v, str) and len(str(v)) > truncate_length
                    else v
                    for k, v in kwargs.items()
                }
                context['args'] = safe_args
                context['kwargs'] = safe_kwargs
            
            start_time = time.time()
            logger.info(f"Starting operation", extra=context)
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Add result and timing to context
                context['duration'] = f"{duration:.3f}s"
                if log_result:
                    if isinstance(result, str):
                        safe_result = result[:truncate_length] + '...' if len(result) > truncate_length else result
                    else:
                        safe_result = result
                    context['result'] = safe_result
                
                logger.info(f"Operation completed successfully", extra=context)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                context['duration'] = f"{duration:.3f}s"
                context['error'] = str(e)
                context['error_type'] = e.__class__.__name__
                
                logger.error(f"Operation failed", extra=context, exc_info=True)
                raise
            
        return cast(F, wrapper)
    return decorator

# Common retryable exceptions
RETRYABLE_NETWORK_ERRORS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

# Common permanent errors
PERMANENT_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)