#!/bin/sh
# OwnRun runner launcher — uses the project venv so KB parsers (ebooklib, etc.) are available.
cd "$(dirname "$0")" || exit 1

export OWNRUN_SERVER_URL="${OWNRUN_SERVER_URL:-http://localhost/~jonboy/ownrun-server}"
export OWNRUN_RUNNER_TOKEN="${OWNRUN_RUNNER_TOKEN:-8c3f944e9daa77d826acfc80534018e3281571a61a641a5cbf3bd2809b54ee66}"
export OLLAMA_API_KEY="${OLLAMA_API_KEY:-464075d570864723bd0ff2023912db30.zOltd9ejRMykrF9tT-NOJb2f}"

if [ -x ./.venv/bin/python ]; then
  exec ./.venv/bin/python run.py
fi
if [ -x ./ownrun.venv/bin/python ]; then
  exec ./ownrun.venv/bin/python run.py
fi
echo "No venv found (.venv or ownrun.venv). Create one and: pip install -e ." >&2
exit 1
