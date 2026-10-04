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
| `OWNRUN_OLLAMA_THINK` | no | `0` | Set to `1` to allow Qwen3.x “thinking” tokens during agent tool calls (can cause truncated tool JSON and HTTP 500) |
| `OWNRUN_OLLAMA_NUM_PREDICT` | no | `-1` | Ollama `num_predict` for agent jobs (`-1` = no cap) |
| `OWNRUN_OLLAMA_NUM_CTX` | no | `32768` | Ollama `num_ctx` for agent jobs. `0` = model default. Raise this if tool follow-ups 500 with “No user query found in messages” (Qwen templates after context truncation) |
| `OWNRUN_JOB_TIMEOUT` | no | `3600` | Per-job Ollama timeout (seconds) |
| `OWNRUN_UPLOAD_RETRIES` | no | `5` | Retries for complete/fail uploads |
| `OWNRUN_AGENT_WORKSPACE` | no | `~/ownrun-workspace` | Parent directory for agent projects (each job sandboxes to a subdirectory) |
| `OWNRUN_AGENT_PROFILES` | no | `workspace,web,workspace_web` | Enabled agent profiles |
| `OWNRUN_AGENT_WALL_CLOCK` | no | `1200` | Max seconds per agent job |
| `OWNRUN_AGENT_FETCH_TIMEOUT` | no | `15` | HTTP fetch and web search timeout (seconds) |
| `OLLAMA_API_KEY` | no | — | Bearer token for Ollama cloud web search (`https://ollama.com/api/web_search`) |
| `OWNRUN_AGENT_WEB_SEARCH` | no | `1` | Set to `0` to disable `web_search` globally on this runner |
| `OWNRUN_KB_DIR` | no | `~/ownrun-kb` | Persistent local vector index directory |
| `OWNRUN_KB_EMBED_MODEL` | no | `nomic-embed-text` | Ollama embedding model for KB ingest/search |
| `OWNRUN_KB_CHUNK_SIZE` | no | `1200` | Characters per chunk |
| `OWNRUN_KB_CHUNK_OVERLAP` | no | `200` | Chunk overlap |
| `OWNRUN_KB_MAX_CHUNKS_PER_DOC` | no | `2000` | Safety cap per document |
| `OWNRUN_AGENT_KB` | no | `1` | Set to `0` to disable `kb_*` tools globally |
| `OWNRUN_SD_ENABLED` | no | `0` | Set to `1` to enable image jobs + exclusive Ollama↔Forge gate |
| `OWNRUN_SD_BASE_URL` | no | `http://127.0.0.1:7860` | Forge / A1111 API base |
| `OWNRUN_OLLAMA_START_SCRIPT` | if SD on | — | Executable that starts Ollama |
| `OWNRUN_OLLAMA_STOP_SCRIPT` | if SD on | — | Executable that stops Ollama |
| `OWNRUN_SD_START_SCRIPT` | if SD on | — | Executable that starts Forge |
| `OWNRUN_SD_STOP_SCRIPT` | if SD on | — | Executable that stops Forge |
| `OWNRUN_BACKEND_SCRIPT_TIMEOUT_SEC` | no | `60` | Max seconds for a start/stop script to exit |
| `OWNRUN_BACKEND_STOP_TIMEOUT_SEC` | no | `60` | Wait for API down after stop |
| `OWNRUN_BACKEND_START_TIMEOUT_SEC` | no | `180` | Wait for API up after start |
| `OWNRUN_SD_JOB_TIMEOUT_SEC` | no | `600` | txt2img HTTP timeout |
| `OWNRUN_BACKEND_IDLE` | no | `ollama` | Backend to start on runner boot (`ollama` or `stablediffusion`) |

## Run

```bash
export OWNRUN_SERVER_URL=https://your-domain.com
export OWNRUN_RUNNER_TOKEN=your-runner-token
python run.py
```

### Image generation (Forge)

Ollama and Forge must not run at the same time on this laptop. With `OWNRUN_SD_ENABLED=1`, the runner fully stops one stack before starting the other via **your** scripts.

```bash
mkdir -p ~/ownrun-scripts
cp scripts/examples/*.sh ~/ownrun-scripts/
chmod +x ~/ownrun-scripts/*.sh
# Edit start-sd.sh FORGE_DIR; adjust start/stop-ollama.sh for your setup

export OWNRUN_SD_ENABLED=1
export OWNRUN_OLLAMA_START_SCRIPT=$HOME/ownrun-scripts/start-ollama.sh
export OWNRUN_OLLAMA_STOP_SCRIPT=$HOME/ownrun-scripts/stop-ollama.sh
export OWNRUN_SD_START_SCRIPT=$HOME/ownrun-scripts/start-sd.sh
export OWNRUN_SD_STOP_SCRIPT=$HOME/ownrun-scripts/stop-sd.sh
```

Do not manually start Forge while Ollama is up (or the reverse). Let OwnRun own the switch.

## Behavior

- Sends heartbeat every poll interval with Ollama models, agent profiles / tool-capable models, known project directories, and (when enabled) `backend_state` / `sd_models`
- Claims one job at a time
- Sends job heartbeats every 30s while running (response may request cancel)
- Supports `ollama.chat`, `agent.task`, `kb.ingest`, `kb.delete`, and `stablediffusion.txt2img` jobs
- Agent jobs use Autogen AgentChat + tools sandboxed to a selected **project** under `OWNRUN_AGENT_WORKSPACE`; progress traces upload while running
- Knowledge base ingest parses PDF/DOCX/EPUB/Markdown/TXT, embeds with Ollama, and stores vectors under `OWNRUN_KB_DIR`
- Image jobs call Forge `/sdapi/v1/txt2img`, upload PNG artifacts, and leave Forge warm until the next Ollama job

## License

OwnRun is distributed under the [GNU Affero General Public License v3](COPYING.txt) (AGPL-3.0).
