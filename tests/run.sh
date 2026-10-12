#!/bin/bash
# Every test, against a built tree: tests/run.py under its old name, with the same arguments (tests/run.py --help).
exec python3 "$(dirname "${BASH_SOURCE[0]}")/run.py" "$@"
