import logging
from logging_config.output_controller import get_output_stream

debug_mode = True


def setup_root_logger():
    """Configure the root logger to output warnings and above to the configured output stream."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.WARNING)
    stream_handler = logging.StreamHandler(get_output_stream())
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Ensure no other handlers interfere, by removing any that might have been added automatically (common in notebooks)
    root_logger.handlers = [stream_handler]
