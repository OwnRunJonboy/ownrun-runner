from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ownrun_runner.config import Config

log = logging.getLogger(__name__)


class ServerClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session = requests.Session()
        bearer = f"Bearer {config.runner_token}"
        self.session.headers.update(
            {
                "Authorization": bearer,
                "X-OwnRun-Authorization": bearer,
                "Accept": "application/json",
            }
        )

    def _url(self, path: str) -> str:
        return f"{self.config.api_base}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> dict[str, Any] | None:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.request(
                    method,
                    self._url(path),
                    json=json,
                    timeout=30,
                )
                if resp.status_code == 401:
                    raise SystemExit("Runner authentication failed — check OWNRUN_RUNNER_TOKEN")
                if resp.status_code >= 500:
                    resp.raise_for_status()
                if resp.status_code >= 400:
                    detail = resp.text
                    try:
                        parsed = resp.json()
                        if isinstance(parsed, dict) and parsed.get("error"):
                            detail = parsed["error"]
                    except ValueError:
                        pass
                    raise requests.HTTPError(f"{resp.status_code}: {detail}", response=resp)
                if not resp.content:
                    return {}
                try:
                    return resp.json()
                except ValueError as e:
                    preview = (resp.text or "")[:200]
                    raise ValueError(
                        f"non-JSON response ({resp.status_code}) from {resp.url}: {preview!r}"
                    ) from e
            except (requests.RequestException, ValueError) as e:
                last_error = e
                wait = min(2**attempt, 30)
                log.warning("Request failed (%s %s), retry in %ss: %s", method, path, wait, e)
                time.sleep(wait)
        if last_error:
            raise last_error
        return None

    def heartbeat(self, models: list[str] | None = None) -> None:
        body: dict[str, Any] = {}
        if models is not None:
            body["available_models"] = models
        self._request("POST", "/runner/heartbeat", json=body)

    def claim(self) -> dict[str, Any] | None:
        data = self._request("POST", "/runner/jobs/claim", json={})
        if not data:
            return None
        job = data.get("job")
        return job if isinstance(job, dict) else None

    def job_heartbeat(self, job_id: int) -> None:
        self._request("POST", f"/runner/jobs/{job_id}/heartbeat", json={})

    def complete(self, job_id: int, result: dict[str, Any]) -> None:
        self._with_upload_retries(
            lambda: self._request("POST", f"/runner/jobs/{job_id}/complete", json={"result": result})
        )

    def fail(self, job_id: int, error_message: str) -> None:
        self._with_upload_retries(
            lambda: self._request(
                "POST",
                f"/runner/jobs/{job_id}/fail",
                json={"error_message": error_message},
            )
        )

    def _with_upload_retries(self, fn: Any) -> None:
        last_error: Exception | None = None
        for attempt in range(self.config.upload_retries):
            try:
                fn()
                return
            except Exception as e:
                last_error = e
                wait = min(2**attempt, 30)
                log.warning("Upload failed, retry in %ss: %s", wait, e)
                time.sleep(wait)
        if last_error:
            raise last_error
