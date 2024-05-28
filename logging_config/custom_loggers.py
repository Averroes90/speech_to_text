import logging
from logging_config.output_controller import get_output_stream
from logging_config.base_config import debug_mode


class ConditionalDebugFilter(logging.Filter):
    def __init__(self, **conditions):
        super().__init__()
        self.conditions = conditions

    def filter(self, record):
        if not self.conditions:  # If no conditions are provided
            return True  # Allow all records through

        for key, value in self.conditions.items():
            record_value = getattr(record, key, None)
            if record_value == value:
                return True

        return False  # Only return False if conditions exist and none are met


def get_conditional_debug_logger(name, **conditions):
    logger = logging.getLogger(name)
    logger.setLevel(
        logging.DEBUG if debug_mode else logging.WARNING
    )  # This logger operates at DEBUG level
    if not any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):  # Avoid adding multiple handlers
        handler = logging.StreamHandler(get_output_stream())
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        handler.addFilter(ConditionalDebugFilter(**conditions))
        logger.addHandler(handler)
    return logger


def get_debug_logger(name):
    """Return a logger configured to output debug messages."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug_mode else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler(get_output_stream())
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
