import sys
import os

# Assume OUTPUT_TO_NOTEBOOK is set as an environment variable that determines the output target
OUTPUT_TO_NOTEBOOK = os.getenv("OUTPUT_TO_NOTEBOOK", "false").lower() == "true"


def get_output_stream():
    """Return the appropriate output stream based on the environment setting."""
    if OUTPUT_TO_NOTEBOOK:
        # Output to Jupyter notebook's print output
        try:
            from IPython.core.getipython import get_ipython

            if (
                "IPKernelApp" not in get_ipython().config
            ):  # Check if not in IPython shell
                raise ImportError("Not running in a Jupyter environment.")
            from IPython.core.display import display, HTML

            class JupyterOutput:
                def write(self, msg):
                    display(HTML(f"<pre>{msg}</pre>"))

                def flush(self):
                    pass

            return JupyterOutput()
        except:
            return sys.stdout  # Fallback to stdout if not in a Jupyter environment
    else:
        return sys.stdout  # Default to standard output
