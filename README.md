# ownrun-runner

Python service that runs on your home computer. It polls the OwnRun server for jobs, executes them against local Ollama (and Autogen AgentChat for agent jobs), and uploads results.

## Requirements

- Python 3.10+
- Ollama running on localhost (default `http://127.0.0.1:11434`)
- For agent jobs: a tools-capable model (e.g. `llama3.2`, `qwen2.5`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with editable install:

```bash
pip install -e .
```

## Configuration

Environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OWNRUN_SERVER_URL` | yes | — | Site origin, e.g. `https://example.com` (runner appends `/api/index.php`) |
| `OWNRUN_RUNNER_TOKEN` | yes | — | Token from `scripts/setup.php` |
| `OWNRUN_POLL_INTERVAL` | no | `10` | Seconds between poll cycles |
| `OLLAMA_BASE_URL` | no | `http://127.0.0.1:11434` | Ollama API base |
| `OWNRUN_JOB_TIMEOUT` | no | `3600` | Per-job Ollama timeout (seconds) |
| `OWNRUN_UPLOAD_RETRIES` | no | `5` | Retries for complete/fail uploads |
| `OWNRUN_AGENT_WORKSPACE` | no | `~/ownrun-workspace` | Sandbox root for agent file tools |
| `OWNRUN_AGENT_PROFILES` | no | `workspace,web,workspace_web` | Enabled agent profiles |
| `OWNRUN_AGENT_WALL_CLOCK` | no | `1200` | Max seconds per agent job |
| `OWNRUN_AGENT_FETCH_TIMEOUT` | no | `15` | HTTP fetch timeout (seconds) |

## Run

```bash
export OWNRUN_SERVER_URL=https://your-domain.com
export OWNRUN_RUNNER_TOKEN=your-runner-token
python run.py
```

## Behavior

- Sends heartbeat every poll interval with Ollama models plus agent profiles / tool-capable models
- Claims one job at a time
- Sends job heartbeats every 30s while running (response may request cancel)
- Supports `ollama.chat` and `agent.task` jobs
- Agent jobs use Autogen AgentChat + sandboxed tools; progress traces upload while running
