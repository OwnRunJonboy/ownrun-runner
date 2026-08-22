from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    server_url: str
    runner_token: str
    poll_interval: float
    ollama_base_url: str
    job_timeout: float
    upload_retries: int

    @classmethod
    def from_env(cls) -> Config:
        server_url = os.environ.get("OWNRUN_SERVER_URL", "").rstrip("/")
        token = os.environ.get("OWNRUN_RUNNER_TOKEN", "")
        if not server_url or not token:
            raise SystemExit(
                "Set OWNRUN_SERVER_URL and OWNRUN_RUNNER_TOKEN environment variables."
            )

        return cls(
            server_url=server_url,
            runner_token=token,
            poll_interval=float(os.environ.get("OWNRUN_POLL_INTERVAL", "10")),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip(
                "/"
            ),
            job_timeout=float(os.environ.get("OWNRUN_JOB_TIMEOUT", "3600")),
            upload_retries=int(os.environ.get("OWNRUN_UPLOAD_RETRIES", "5")),
        )

    @property
    def api_base(self) -> str:
        return f"{self.server_url}/api/index.php"
