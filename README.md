# ownrun-runner

Python service that runs on your home computer. It polls the OwnRun server for jobs, executes them against local Ollama, and uploads results.

## Requirements

- Python 3.10+
- Ollama running on localhost (default `http://127.0.0.1:11434`)

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

## Run

```bash
export OWNRUN_SERVER_URL=https://your-domain.com
export OWNRUN_RUNNER_TOKEN=your-runner-token
python run.py
```

## systemd (optional)

```ini
[Unit]
Description=OwnRun runner
After=network.target

[Service]
Type=simple
Environment=OWNRUN_SERVER_URL=https://your-domain.com
Environment=OWNRUN_RUNNER_TOKEN=your-runner-token
ExecStart=/path/to/.venv/bin/python /path/to/ownrun-runner/run.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Save as `~/.config/systemd/user/ownrun-runner.service`, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now ownrun-runner
```

## Behavior

- Sends heartbeat every poll interval with current Ollama model list
- Claims one job at a time
- Sends job heartbeats every 30s while running
- Supports `ollama.chat` jobs (non-streaming `/api/chat`)
- Reports failures clearly (Ollama down, missing model, timeout)
