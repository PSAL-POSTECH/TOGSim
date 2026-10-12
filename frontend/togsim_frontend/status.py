"""The exit statuses of the two entry points, togsim_frontend.onnx_frontend and togsim_frontend.acs. Each says
what the caller has to change and no two share a number. 2 is argparse's, named so that it stays taken, and 1 is
left to Python, for an exception none of them names: nothing here returns either."""
import sys

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NODES_NOT_RUN = 3
EXIT_UNSUPPORTED_MACHINE = 5


def fail(status: int, message: str) -> int:
    """Say `message` on stderr as an error and hand back `status`, for an entry point to return."""
    print(f"error: {message}", file=sys.stderr)
    return status
