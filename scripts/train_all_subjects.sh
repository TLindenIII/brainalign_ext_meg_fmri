#!/bin/bash

set -e

PYTHON_BIN="python"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -x "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
fi

exec "$PYTHON_BIN" scripts/train_all_subjects.py "$@"
