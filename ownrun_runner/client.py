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
        params: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> dict[str, Any] | None:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.request(
                    method,
                    self._url(path),
                    json=json,
                    params=params,
                    timeout=30,
                )
                if resp.status_code == 401:
                    raise SystemExit("Runner authentication failed — check OWNRUN_RUNNER_TOKEN")
                if 400 <= resp.status_code < 500:
                    detail = resp.text
                    try:
                        parsed = resp.json()
                        if isinstance(parsed, dict) and parsed.get("error"):
                            detail = parsed["error"]
                    except ValueError:
                        pass
                    raise requests.HTTPError(f"{resp.status_code}: {detail}", response=resp)
                if resp.status_code >= 500:
                    resp.raise_for_status()
                if not resp.content:
                    return {}
                try:
                    return resp.json()
                except ValueError as e:
                    preview = (resp.text or "")[:200]
                    raise ValueError(
                        f"non-JSON response ({resp.status_code}) from {resp.url}: {preview!r}"
                    ) from e
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if 400 <= code < 500:
                    raise
                last_error = e
                wait = min(2**attempt, 30)
                log.warning("Request failed (%s %s), retry in %ss: %s", method, path, wait, e)
                time.sleep(wait)
            except (requests.RequestException, ValueError) as e:
                last_error = e
                wait = min(2**attempt, 30)
                log.warning("Request failed (%s %s), retry in %ss: %s", method, path, wait, e)
                time.sleep(wait)
        if last_error:
            raise last_error
        return None

    def heartbeat(
        self,
        models: list[str] | None = None,
        agent_profiles: list[str] | None = None,
        agent_models: list[str] | None = None,
        agent_projects: list[str] | None = None,
        timezone: str | None = None,
        ollama_web_search: bool | None = None,
        agent_tool_ids: list[str] | None = None,
    ) -> None:
        body: dict[str, Any] = {}
        if models is not None:
            body["available_models"] = models
        if agent_profiles is not None:
            body["agent_profiles"] = agent_profiles
        if agent_models is not None:
            body["agent_models"] = agent_models
        if agent_projects is not None:
            body["agent_projects"] = agent_projects
        if timezone is not None:
            body["timezone"] = timezone
        if ollama_web_search is not None:
            body["ollama_web_search"] = ollama_web_search
        if agent_tool_ids is not None:
            body["agent_tool_ids"] = agent_tool_ids
        self._request("POST", "/runner/heartbeat", json=body)

    def tool_policy(self, job_id: int) -> list[str]:
        data = self._request("GET", f"/runner/jobs/{job_id}/tool-policy", retries=1) or {}
        raw = data.get("disabled_tools") or []
        if not isinstance(raw, list):
            return []
        return [str(x) for x in raw if str(x).strip()]

    def claim(self) -> dict[str, Any] | None:
        data = self._request("POST", "/runner/jobs/claim", json={})
        if not data:
            return None
        job = data.get("job")
        return job if isinstance(job, dict) else None

    def job_heartbeat(self, job_id: int) -> bool:
        """Extend lease. Returns True if cancel was requested."""
        data = self._request("POST", f"/runner/jobs/{job_id}/heartbeat", json={}) or {}
        return bool(data.get("cancel_requested"))

    def progress(self, job_id: int, result: dict[str, Any]) -> None:
        self._request(
            "POST",
            f"/runner/jobs/{job_id}/progress",
            json={"result": result},
            retries=1,
        )

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

    def calendar_search(
        self,
        job_id: int,
        start: str,
        end: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"from": start, "to": end}
        if query:
            params["q"] = query
        return self._request(
            "GET",
            f"/runner/jobs/{job_id}/calendar",
            params=params,
            retries=1,
        ) or {}

    def calendar_get(self, job_id: int, entry_id: int) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/runner/jobs/{job_id}/calendar/{entry_id}",
            retries=1,
        ) or {}

    def calendar_add(self, job_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/calendar",
            json=body,
            retries=1,
        ) or {}

    def calendar_edit(self, job_id: int, entry_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/calendar/{entry_id}",
            json=body,
            retries=1,
        ) or {}

    def calendar_delete(self, job_id: int, entry_id: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/calendar/{entry_id}/delete",
            json={},
            retries=1,
        ) or {}

    def journal_search(
        self,
        job_id: int,
        start: str,
        end: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"from": start, "to": end}
        if query:
            params["q"] = query
        return self._request(
            "GET",
            f"/runner/jobs/{job_id}/journal",
            params=params,
            retries=1,
        ) or {}

    def journal_get(self, job_id: int, entry_id: int) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/runner/jobs/{job_id}/journal/{entry_id}",
            retries=1,
        ) or {}

    def journal_add(self, job_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/journal",
            json=body,
            retries=1,
        ) or {}

    def journal_edit(self, job_id: int, entry_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/journal/{entry_id}",
            json=body,
            retries=1,
        ) or {}

    def journal_delete(self, job_id: int, entry_id: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/journal/{entry_id}/delete",
            json={},
            retries=1,
        ) or {}

    def task_search(
        self,
        job_id: int,
        *,
        status: str | None = None,
        priority: str | None = None,
        query: str | None = None,
        due_from: str | None = None,
        due_to: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority
        if query:
            params["q"] = query
        if due_from:
            params["due_from"] = due_from
        if due_to:
            params["due_to"] = due_to
        return self._request(
            "GET",
            f"/runner/jobs/{job_id}/tasks",
            params=params or None,
            retries=1,
        ) or {}

    def task_get(self, job_id: int, entry_id: int) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/runner/jobs/{job_id}/tasks/{entry_id}",
            retries=1,
        ) or {}

    def task_add(self, job_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/tasks",
            json=body,
            retries=1,
        ) or {}

    def task_edit(self, job_id: int, entry_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/tasks/{entry_id}",
            json=body,
            retries=1,
        ) or {}

    def task_delete(self, job_id: int, entry_id: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/runner/jobs/{job_id}/tasks/{entry_id}/delete",
            json={},
            retries=1,
        ) or {}

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
