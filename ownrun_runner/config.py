from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
        )

    @property
    def api_base(self) -> str:
        return f"{self.server_url}/api/index.php"
