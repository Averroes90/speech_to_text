import sys
import os
from typing import Union, TextIO

# Assume OUTPUT_TO_NOTEBOOK is set as an environment variable that determines the output target
OUTPUT_TO_NOTEBOOK = os.getenv("OUTPUT_TO_NOTEBOOK", "false").lower() == "true"


class JupyterOutput:
    """Custom output class for displaying logs in Jupyter notebooks using HTML."""

    def write(self, msg: str) -> None:
        from IPython.core.display import display, HTML
        display(HTML(f"<pre>{msg}</pre>"))

    def flush(self) -> None:
        pass


def get_output_stream() -> Union[TextIO, JupyterOutput]:
    """
    Return the appropriate output stream based on the environment setting.
    If running in a Jupyter notebook, returns a custom JupyterOutput object that uses IPython's display mechanisms.
    Otherwise, returns sys.stdout as the default output stream.
    """
    if OUTPUT_TO_NOTEBOOK:
        try:
            from IPython.core.getipython import get_ipython

            if (
                "IPKernelApp" not in get_ipython().config
            ):  # Check if not in IPython shell
                raise ImportError("Not running in a Jupyter environment.")
            return JupyterOutput()
        except ImportError:
            return sys.stdout  # Fallback to stdout if not in a Jupyter environment
    else:
        return sys.stdout  # Default to standard output
