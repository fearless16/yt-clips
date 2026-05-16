import logging
import pytest
from utils.logger import get_logger


class TestLoggerDeduplication:
    def test_same_name_returns_same_logger(self):
        l1 = get_logger("test_dupe", level="DEBUG")
        l2 = get_logger("test_dupe", level="INFO")
        assert l1 is l2

    def test_different_names_are_different_loggers(self):
        l1 = get_logger("test_a", level="DEBUG")
        l2 = get_logger("test_b", level="DEBUG")
        assert l1 is not l2

    def test_handler_count_stable_after_multiple_calls(self):
        l1 = get_logger("test_handler_count", level="DEBUG")
        n1 = len(l1.handlers)
        l2 = get_logger("test_handler_count", level="INFO")
        n2 = len(l2.handlers)
        assert n1 == n2, f"Handlers grew: {n1} → {n2}"

    def test_propagation_disabled(self):
        logger = get_logger("test_prop", level="DEBUG")
        assert logger.propagate is False


class TestLoggerNoDuplicateHandlers:
    """Test that repeated get_logger calls don't stack handlers."""

    def test_handlers_not_duplicated_on_repeat_call(self):
        logger = get_logger("test_no_dupe")
        initial_handlers = list(logger.handlers)
        get_logger("test_no_dupe")
        assert len(logger.handlers) == len(initial_handlers)

    def test_console_handler_added_once(self):
        logger = get_logger("test_console_once")
        rich_handlers = [h for h in logger.handlers if "RichHandler" in type(h).__name__]
        assert len(rich_handlers) == 1

    def test_file_handler_added_once(self):
        logger = get_logger("test_file_once")
        json_handlers = [h for h in logger.handlers if "JsonFileHandler" in type(h).__name__]
        assert len(json_handlers) == 1


class TestLoggerRegistryCleanup:
    """Logger registry should not accumulate stale loggers."""

    def test_calling_get_logger_does_not_leak_handlers(self):
        import logging as logging_mod
        root = logging_mod.getLogger()
        root_count = len(root.handlers)
        logger = get_logger("test_registry_leak")
        logger.info("test")
        assert len(root.handlers) == root_count
