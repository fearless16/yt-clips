"""
resilience.py — Circuit breaker and retry utilities for API resilience.
"""
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures when external APIs are down.
    
    States:
      - CLOSED: Normal operation, requests go through
      - OPEN: API is down, fail fast without calling
      - HALF_OPEN: Testing if API recovered
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    @property
    def state(self) -> str:
        if self._state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                logger.info("CircuitBreaker: HALF_OPEN — testing API...")
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            raise Exception(f"CircuitBreaker: OPEN — skipping call to {func.__name__}")

        try:
            result = func(*args, **kwargs)
            # Success: reset failure count
            if self._state == "HALF_OPEN":
                logger.info("CircuitBreaker: CLOSED — API recovered!")
                self._failure_count = 0
                self._state = "CLOSED"
            return result

        except self.expected_exception as e:
            self._record_failure()
            raise

    def allow_request(self) -> bool:
        """Return True if request is allowed, False otherwise."""
        return self.state != "OPEN"

    def record_success(self) -> None:
        """Record success and reset the circuit."""
        self._failure_count = 0
        self._state = "CLOSED"
        self._last_failure_time = None

    def record_failure(self) -> None:
        """Record a failure."""
        self._record_failure()

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"CircuitBreaker: OPEN — {self.failure_threshold} failures detected, "
                f"pausing for {self.recovery_timeout}s"
            )


def circuit_breaker(
    failure_threshold: int = 3,
    recovery_timeout: float = 60.0,
    default_return: Any = None,
):
    """
    Decorator to add circuit breaker behavior to a function.
    
    Usage:
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def fetch_trends():
            ...
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return breaker.call(func, *args, **kwargs)
            except Exception as e:
                logger.error(f"Circuit breaker open for {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retry with exponential backoff.
    
    Usage:
        @retry_with_backoff(max_attempts=3, base_delay=2.0)
        def unstable_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise

                    delay = base_delay * (2 ** (attempt - 1)) if exponential else base_delay
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            raise last_exception
        return wrapper
    return decorator