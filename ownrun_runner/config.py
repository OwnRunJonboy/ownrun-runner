from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _default_workspace() -> str:
    return str(Path.home() / "ownrun-workspace")


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
        )

    @property
    def api_base(self) -> str:
        return f"{self.server_url}/api/index.php"
