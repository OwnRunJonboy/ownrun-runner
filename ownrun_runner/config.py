from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _default_workspace() -> str:
    return str(Path.home() / "ownrun-workspace")


def _default_kb_dir() -> str:
    return str(Path.home() / "ownrun-kb")


@dataclass(frozen=True)
class Config:
    server_url: str
    runner_token: str
    poll_interval: float
    ollama_base_url: str
    job_timeout: float
    upload_retries: int
    agent_workspace: str
    agent_profiles: tuple[str, ...]
    agent_wall_clock: float
    agent_fetch_timeout: float
    timezone: str
    ollama_api_key: str
    agent_web_search: bool
    ollama_think: bool
    ollama_num_predict: int
    ollama_num_ctx: int
    kb_dir: str
    kb_embed_model: str
    kb_chunk_size: int
    kb_chunk_overlap: int
    kb_max_chunks_per_doc: int
    agent_kb: bool
    sd_enabled: bool
    sd_base_url: str
    ollama_start_script: str
    ollama_stop_script: str
    sd_start_script: str
    sd_stop_script: str
    backend_script_timeout: float
    backend_stop_timeout: float
    backend_start_timeout: float
    sd_job_timeout: float
    backend_idle: str

    @classmethod
    def from_env(cls) -> Config:
        server_url = os.environ.get("OWNRUN_SERVER_URL", "").rstrip("/")
        token = os.environ.get("OWNRUN_RUNNER_TOKEN", "")
        if not server_url or not token:
            raise SystemExit(
                "Set OWNRUN_SERVER_URL and OWNRUN_RUNNER_TOKEN environment variables."
            )

        profiles_raw = os.environ.get(
            "OWNRUN_AGENT_PROFILES", "workspace,web,workspace_web"
        )
        profiles = tuple(
            p.strip() for p in profiles_raw.split(",") if p.strip()
        ) or ("workspace",)

        timezone = (os.environ.get("OWNRUN_TIMEZONE") or "America/New_York").strip()
        if not timezone:
            timezone = "America/New_York"
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, Exception) as e:
            raise SystemExit(
                f"Invalid OWNRUN_TIMEZONE {timezone!r}: {e}"
            ) from e

        backend_idle = (os.environ.get("OWNRUN_BACKEND_IDLE") or "ollama").strip().lower()
        if backend_idle not in ("ollama", "stablediffusion"):
            backend_idle = "ollama"

        return cls(
            server_url=server_url,
            runner_token=token,
            poll_interval=float(os.environ.get("OWNRUN_POLL_INTERVAL", "10")),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip(
                "/"
            ),
            job_timeout=float(os.environ.get("OWNRUN_JOB_TIMEOUT", "3600")),
            upload_retries=int(os.environ.get("OWNRUN_UPLOAD_RETRIES", "5")),
            agent_workspace=os.environ.get("OWNRUN_AGENT_WORKSPACE", _default_workspace()),
            agent_profiles=profiles,
            agent_wall_clock=float(os.environ.get("OWNRUN_AGENT_WALL_CLOCK", "1200")),
            agent_fetch_timeout=float(os.environ.get("OWNRUN_AGENT_FETCH_TIMEOUT", "15")),
            timezone=timezone,
            ollama_api_key=(os.environ.get("OLLAMA_API_KEY") or "").strip(),
            agent_web_search=os.environ.get("OWNRUN_AGENT_WEB_SEARCH", "1") != "0",
            ollama_think=os.environ.get("OWNRUN_OLLAMA_THINK", "0") == "1",
            ollama_num_predict=int(os.environ.get("OWNRUN_OLLAMA_NUM_PREDICT", "-1")),
            # 0 = leave Ollama's model default (often 4096). Agent tool loops
            # need more: truncation can drop the user turn and Qwen templates 500.
            ollama_num_ctx=max(0, int(os.environ.get("OWNRUN_OLLAMA_NUM_CTX", "32768"))),
            kb_dir=os.environ.get("OWNRUN_KB_DIR", _default_kb_dir()),
            kb_embed_model=(os.environ.get("OWNRUN_KB_EMBED_MODEL") or "nomic-embed-text").strip(),
            kb_chunk_size=max(200, int(os.environ.get("OWNRUN_KB_CHUNK_SIZE", "1200"))),
            kb_chunk_overlap=max(0, int(os.environ.get("OWNRUN_KB_CHUNK_OVERLAP", "200"))),
            kb_max_chunks_per_doc=max(1, int(os.environ.get("OWNRUN_KB_MAX_CHUNKS_PER_DOC", "2000"))),
            agent_kb=os.environ.get("OWNRUN_AGENT_KB", "1") != "0",
            sd_enabled=os.environ.get("OWNRUN_SD_ENABLED", "0") == "1",
            sd_base_url=os.environ.get("OWNRUN_SD_BASE_URL", "http://127.0.0.1:7860").rstrip("/"),
            ollama_start_script=os.path.expanduser(
                (os.environ.get("OWNRUN_OLLAMA_START_SCRIPT") or "").strip()
            ),
            ollama_stop_script=os.path.expanduser(
                (os.environ.get("OWNRUN_OLLAMA_STOP_SCRIPT") or "").strip()
            ),
            sd_start_script=os.path.expanduser(
                (os.environ.get("OWNRUN_SD_START_SCRIPT") or "").strip()
            ),
            sd_stop_script=os.path.expanduser(
                (os.environ.get("OWNRUN_SD_STOP_SCRIPT") or "").strip()
            ),
            backend_script_timeout=float(
                os.environ.get("OWNRUN_BACKEND_SCRIPT_TIMEOUT_SEC", "60")
            ),
            backend_stop_timeout=float(os.environ.get("OWNRUN_BACKEND_STOP_TIMEOUT_SEC", "60")),
            backend_start_timeout=float(
                os.environ.get("OWNRUN_BACKEND_START_TIMEOUT_SEC", "180")
            ),
            sd_job_timeout=float(os.environ.get("OWNRUN_SD_JOB_TIMEOUT_SEC", "600")),
            backend_idle=backend_idle,
        )

    @property
    def api_base(self) -> str:
        return f"{self.server_url}/api/index.php"
