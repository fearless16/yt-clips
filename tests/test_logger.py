import logging
import pytest
from utils.logger import get_logger


class TestLoggerDeduplication:
    def setup_method(self):
        _REGISTERED_LOGGERS.discard("test_dupe")

    def test_same_name_returns_same_logger(self):
        l1 = get_logger("test_dupe", level="DEBUG")
        l2 = get_logger("test_dupe", level="INFO")
        assert l1 is l2

    def test_different_names_are_different_loggers(self):
        l1 = get_logger("test_a", level="DEBUG")
        l2 = get_logger("test_b", level="DEBUG")
        assert l1 is not l2

    def test_handler_count_stable_after_multiple_calls(self):
        _REGISTERED_LOGGERS.discard("test_handler_count")
        l1 = get_logger("test_handler_count", level="DEBUG")
        n1 = len(l1.handlers)
        l2 = get_logger("test_handler_count", level="INFO")
        n2 = len(l2.handlers)
        assert n1 == n2, f"Handlers grew: {n1} → {n2}"

    def test_propagation_disabled(self):
        _REGISTERED_LOGGERS.discard("test_prop")
        logger = get_logger("test_prop", level="DEBUG")
        assert logger.propagate is False
