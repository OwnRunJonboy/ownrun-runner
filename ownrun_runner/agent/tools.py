from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from ownrun_runner.client import ServerClient

log = logging.getLogger(__name__)

TOOL_RESULT_MAX = 8 * 1024
READ_FILE_MAX = 64 * 1024
WRITE_FILE_MAX = 64 * 1024
LIST_DIR_MAX = 200
FETCH_MAX = 256 * 1024
FETCH_REDIRECTS = 3


def truncate(text: str, limit: int = TOOL_RESULT_MAX) -> str:
    if len(text) <= limit:
        return text
    log.info("Truncated tool/trace content from %d to %d chars", len(text), limit)
    return text[: limit - 20] + "\n…[truncated]"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._parts)).strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


PROJECT_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def validate_project_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not PROJECT_NAME_RE.fullmatch(cleaned):
        raise ValueError(
            "Invalid project name: use 1–64 chars starting with a letter or digit; "
            "only letters, digits, '.', '_', and '-' allowed"
        )
    return cleaned


def list_projects(workspace_root: Path) -> list[str]:
    root = workspace_root.expanduser().resolve()
    if not root.is_dir():
        return []
    names: list[str] = []
    try:
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir() and not entry.name.startswith(".") and PROJECT_NAME_RE.fullmatch(
                entry.name
            ):
                names.append(entry.name)
    except OSError as e:
        log.warning("Could not list projects under %s: %s", root, e)
    return names


def project_workspace(workspace_root: Path, project: str) -> "Workspace":
    name = validate_project_name(project)
    root = workspace_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Reject symlink escape of the project directory itself.
    project_dir = (root / name).resolve()
    try:
        project_dir.relative_to(root)
    except ValueError as e:
        raise ValueError("Project path escapes workspace") from e
    return Workspace(project_dir)


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> Path:
        raw = (rel or ".").strip() or "."
        # Models often pass "/file.txt" meaning the workspace root. In pathlib,
        # joining with an absolute path replaces the left side, which would
        # escape to the host filesystem — strip leading slashes instead.
        raw = raw.lstrip("/") or "."
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise ValueError("Path escapes workspace") from e
        # Reject symlink escape: if the path exists, its real path must stay inside.
        if candidate.exists():
            real = candidate.resolve()
            try:
                real.relative_to(self.root)
            except ValueError as e:
                raise ValueError("Symlink escapes workspace") from e
            return real
        # For new files, ensure parent is inside workspace.
        parent = candidate.parent
        if parent.exists():
            real_parent = parent.resolve()
            try:
                real_parent.relative_to(self.root)
            except ValueError as e:
                raise ValueError("Path escapes workspace") from e
        return candidate

    def relpath(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace("\\", "/")


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are allowed")
    host = parsed.hostname
    if not host:
        raise ValueError("URL missing host")
    if host.lower() in {"localhost", "metadata.google.internal"}:
        raise ValueError("Host is not allowed")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed: {e}") from e

    if not infos:
        raise ValueError("DNS resolution returned no addresses")

    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if not _is_public_ip(ip):
            raise ValueError(f"Refusing non-public address {ip_str}")


def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def clock_payload(timezone_name: str, when: datetime | None = None) -> dict[str, Any]:
    """UTC + local clock, plus ready-made calendar_search windows."""
    tz = ZoneInfo(timezone_name)
    utc = (when or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    local = utc.astimezone(tz)
    local_midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    days_since_sunday = (local.weekday() + 1) % 7
    week_start = local_midnight - timedelta(days=days_since_sunday)
    week_end = week_start + timedelta(days=7)
    today_end = local_midnight + timedelta(days=1)
    next_7_end = local_midnight + timedelta(days=7)
    return {
        "utc": _to_utc_z(utc),
        "timezone": timezone_name,
        "local": local.isoformat(timespec="seconds"),
        "weekday": local.strftime("%A"),
        "today": {"start": _to_utc_z(local_midnight), "end": _to_utc_z(today_end)},
        "this_week": {"start": _to_utc_z(week_start), "end": _to_utc_z(week_end)},
        "next_7_days": {"start": _to_utc_z(local_midnight), "end": _to_utc_z(next_7_end)},
    }


def _dump(data: Any) -> str:
    return truncate(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def _calendar_error(exc: Exception) -> str:
    import requests

    if isinstance(exc, requests.HTTPError):
        return f"Error: {exc}"
    return f"Error: {exc}"


def build_tools(
    workspace: Workspace | None,
    *,
    include_workspace: bool,
    include_web: bool,
    fetch_timeout: float,
    timezone_name: str = "America/New_York",
    calendar_client: ServerClient | None = None,
    job_id: int | None = None,
) -> list[Callable[..., str]]:
    tools: list[Callable[..., str]] = []

    def now() -> str:
        """Current date/time, plus UTC windows for calendar_search.

        No arguments. Copy this_week.start/end (or today / next_7_days) into
        calendar_search. Do not invent other calendar tools.
        """
        return json.dumps(clock_payload(timezone_name), ensure_ascii=False)

    tools.append(now)

    if calendar_client is not None and job_id is not None:
        client = calendar_client
        jid = job_id

        def calendar_search(start: str = "", end: str = "", query: str = "") -> str:
            """List calendar entries between two UTC times.

            For 'this week', copy start/end from the system prompt or from now().this_week.
            If start and end are omitted, this week is used. Other ranges (including past years)
            are searched as given.
            """
            try:
                start_s = (start or "").strip()
                end_s = (end or "").strip()
                if not start_s or not end_s:
                    windows = clock_payload(timezone_name)
                    start_s = start_s or windows["this_week"]["start"]
                    end_s = end_s or windows["this_week"]["end"]
                data = client.calendar_search(jid, start_s, end_s, query.strip() or None)
                log.info("calendar_search job=%s", jid)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def calendar_get(id: int) -> str:
            """Get one calendar entry by id, including description.

            Args:
                id: Calendar entry id from calendar_search.
            """
            try:
                entry_id = int(id)
                data = client.calendar_get(jid, entry_id)
                log.info("calendar_get job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def calendar_add(
            title: str,
            starts_at: str,
            ends_at: str = "",
            all_day: bool = False,
            description: str = "",
        ) -> str:
            """Create a calendar entry. Project is set from this job automatically.

            Args:
                title: Short title (max 200 chars).
                starts_at: Start instant, UTC ISO-8601.
                ends_at: Optional end instant, UTC ISO-8601.
                all_day: True for a civil-date event (use local midnight converted to UTC).
                description: Optional notes.
            """
            try:
                body: dict[str, Any] = {
                    "title": title,
                    "starts_at": starts_at,
                    "all_day": bool(all_day),
                }
                if ends_at and ends_at.strip():
                    body["ends_at"] = ends_at.strip()
                if description and description.strip():
                    body["description"] = description
                data = client.calendar_add(jid, body)
                entry_id = data.get("id")
                log.info("calendar_add job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def calendar_edit(
            id: int,
            title: str = "",
            starts_at: str = "",
            ends_at: str = "",
            all_day: str = "",
            description: str = "",
        ) -> str:
            """Edit a calendar entry in this job's project write set. Omit unchanged fields.

            Args:
                id: Calendar entry id.
                title: New title, or empty to leave unchanged.
                starts_at: New start UTC ISO-8601, or empty to leave unchanged.
                ends_at: New end UTC ISO-8601, empty to leave unchanged, or "null" to clear.
                all_day: "true"/"false", or empty to leave unchanged.
                description: New description, or empty to leave unchanged.
            """
            try:
                entry_id = int(id)
                body: dict[str, Any] = {}
                if title and title.strip():
                    body["title"] = title.strip()
                if starts_at and str(starts_at).strip():
                    body["starts_at"] = str(starts_at).strip()
                end_raw = str(ends_at or "").strip()
                if end_raw.lower() == "null":
                    body["ends_at"] = None
                elif end_raw:
                    body["ends_at"] = end_raw
                if str(all_day).strip() != "":
                    body["all_day"] = str(all_day).strip().lower() in {"1", "true", "yes", "on"}
                if description:
                    body["description"] = description
                if not body:
                    return "Error: no fields to update"
                data = client.calendar_edit(jid, entry_id, body)
                log.info("calendar_edit job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def calendar_delete(id: int) -> str:
            """Delete a calendar entry in this job's project write set.

            Args:
                id: Calendar entry id.
            """
            try:
                entry_id = int(id)
                data = client.calendar_delete(jid, entry_id)
                log.info("calendar_delete job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        tools.extend(
            [calendar_search, calendar_get, calendar_add, calendar_edit, calendar_delete]
        )

        def journal_search(start: str = "", end: str = "", query: str = "") -> str:
            """List journal entries between two civil dates (YYYY-MM-DD).

            For 'this month' or 'recent', derive dates from now().local (first 10 chars are YYYY-MM-DD).
            start and end are required ISO dates. Optional query filters title/body.
            """
            try:
                start_s = (start or "").strip()
                end_s = (end or "").strip()
                if not start_s or not end_s:
                    local_date = str(clock_payload(timezone_name)["local"])[:10]
                    start_s = start_s or local_date
                    end_s = end_s or local_date
                # Accept full ISO timestamps by taking the date prefix.
                start_s = start_s[:10]
                end_s = end_s[:10]
                data = client.journal_search(jid, start_s, end_s, query.strip() or None)
                log.info("journal_search job=%s", jid)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def journal_get(id: int) -> str:
            """Get one journal entry by id, including full body.

            Args:
                id: Journal entry id from journal_search.
            """
            try:
                entry_id = int(id)
                data = client.journal_get(jid, entry_id)
                log.info("journal_get job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def journal_add(body: str, title: str = "", entry_date: str = "") -> str:
            """Create a journal entry for this job's project. Fails if the job has no project.

            Args:
                body: Required journal text.
                title: Optional short title.
                entry_date: Optional civil date YYYY-MM-DD (defaults to today).
            """
            try:
                payload: dict[str, Any] = {"body": body}
                if title and title.strip():
                    payload["title"] = title.strip()
                if entry_date and str(entry_date).strip():
                    payload["entry_date"] = str(entry_date).strip()[:10]
                data = client.journal_add(jid, payload)
                entry_id = data.get("id")
                log.info("journal_add job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def journal_edit(
            id: int,
            body: str = "",
            title: str = "",
            entry_date: str = "",
        ) -> str:
            """Edit a journal entry in this job's project. Omit unchanged fields.

            Args:
                id: Journal entry id.
                body: New body, or empty to leave unchanged.
                title: New title, empty to leave unchanged, or "null" to clear.
                entry_date: New civil date YYYY-MM-DD, or empty to leave unchanged.
            """
            try:
                entry_id = int(id)
                payload: dict[str, Any] = {}
                if body:
                    payload["body"] = body
                title_raw = str(title or "").strip()
                if title_raw.lower() == "null":
                    payload["title"] = None
                elif title_raw:
                    payload["title"] = title_raw
                if entry_date and str(entry_date).strip():
                    payload["entry_date"] = str(entry_date).strip()[:10]
                if not payload:
                    return "Error: no fields to update"
                data = client.journal_edit(jid, entry_id, payload)
                log.info("journal_edit job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        def journal_delete(id: int) -> str:
            """Soft-delete a journal entry in this job's project write set.

            Args:
                id: Journal entry id.
            """
            try:
                entry_id = int(id)
                data = client.journal_delete(jid, entry_id)
                log.info("journal_delete job=%s id=%s", jid, entry_id)
                return _dump(data)
            except Exception as e:
                return _calendar_error(e)

        tools.extend(
            [journal_search, journal_get, journal_add, journal_edit, journal_delete]
        )

    if include_workspace:
        if workspace is None:
            raise ValueError("Workspace tools require a project workspace")
        ws = workspace

        def list_dir(path: str = ".") -> str:
            """List files and directories under a project-relative path.

            Args:
                path: Path relative to the project root (leading "/" is ok; default ".").
            """
            try:
                target = ws.resolve(path)
            except ValueError as e:
                return f"Error: {e}"
            if not target.exists():
                return f"Error: path not found: {path}"
            if not target.is_dir():
                return f"Error: not a directory: {ws.relpath(target)}"
            entries = sorted(target.iterdir(), key=lambda p: p.name.lower())
            lines: list[str] = []
            for i, entry in enumerate(entries):
                if i >= LIST_DIR_MAX:
                    lines.append(f"…and {len(entries) - LIST_DIR_MAX} more")
                    break
                kind = "dir" if entry.is_dir() else "file"
                lines.append(f"{kind}\t{ws.relpath(entry)}")
            return truncate("\n".join(lines) if lines else "(empty)")

        def read_file(path: str) -> str:
            """Read a text file from the project.

            Args:
                path: Path relative to the project root (leading "/" is ok).
            """
            try:
                target = ws.resolve(path)
            except ValueError as e:
                return f"Error: {e}"
            if not target.is_file():
                return f"Error: not a file: {path}"
            data = target.read_bytes()
            if b"\x00" in data[:4096]:
                return "Error: binary files are not supported"
            if len(data) > READ_FILE_MAX:
                return f"Error: file larger than {READ_FILE_MAX} bytes"
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
            return truncate(text)

        def write_file(path: str, content: str) -> str:
            """Write a text file inside the project (creates parent dirs).

            Args:
                path: Path relative to the project root (leading "/" is ok).
                content: Text content to write.
            """
            if len(content.encode("utf-8")) > WRITE_FILE_MAX:
                return f"Error: content larger than {WRITE_FILE_MAX} bytes"
            try:
                target = ws.resolve(path)
            except ValueError as e:
                return f"Error: {e}"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            except OSError as e:
                return f"Error: {e}"
            return f"Wrote {ws.relpath(target)} ({len(content)} chars)"

        tools.extend([list_dir, read_file, write_file])

    if include_web:

        def http_fetch(url: str) -> str:
            """Fetch a public http(s) URL with GET and return text content.

            Args:
                url: Absolute http or https URL.
            """
            current = url.strip()
            try:
                for _ in range(FETCH_REDIRECTS + 1):
                    _assert_public_url(current)
                    req = Request(
                        current,
                        method="GET",
                        headers={"User-Agent": "OwnRun-agent/1.0", "Accept": "text/*,application/json,application/xml"},
                    )
                    with urlopen(req, timeout=fetch_timeout) as resp:  # noqa: S310 — SSRF checked above
                        if 300 <= getattr(resp, "status", 200) < 400:
                            location = resp.headers.get("Location")
                            if not location:
                                return "Error: redirect without Location"
                            current = location
                            continue
                        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                        raw = resp.read(FETCH_MAX + 1)
                        if len(raw) > FETCH_MAX:
                            return f"Error: response larger than {FETCH_MAX} bytes"
                        if content_type.startswith("image/") or content_type in {
                            "application/octet-stream",
                            "application/pdf",
                            "application/zip",
                        }:
                            return f"Error: unsupported content type {content_type or 'unknown'}"
                        if b"\x00" in raw[:4096] and not content_type.startswith("text/"):
                            return "Error: binary response rejected"
                        text = raw.decode("utf-8", errors="replace")
                        if "html" in content_type or text.lstrip().lower().startswith("<!doctype html") or "<html" in text[:200].lower():
                            text = _html_to_text(text)
                        return truncate(text)
                return "Error: too many redirects"
            except ValueError as e:
                return f"Error: {e}"
            except Exception as e:
                return f"Error: fetch failed: {e}"

        tools.append(http_fetch)

    return tools


def profile_flags(profile: str) -> tuple[bool, bool]:
    if profile == "workspace":
        return True, False
    if profile == "web":
        return False, True
    if profile == "workspace_web":
        return True, True
    raise ValueError(f"Unknown agent profile: {profile}")


PROFILE_SYSTEM: dict[str, str] = {
    "workspace": (
        "You are OwnRun's workspace agent. Use the provided tools to work with files "
        "inside the selected project. Paths are relative to the project root. "
        "Prefer tools over guessing file contents. When finished, give a concise final answer."
    ),
    "web": (
        "You are OwnRun's web agent. Use http_fetch for public http(s) URLs only. "
        "Do not invent page contents. When finished, give a concise final answer."
    ),
    "workspace_web": (
        "You are OwnRun's agent with project files and public HTTP fetch. "
        "Use tools for facts. Paths are project-relative. When finished, give a concise final answer."
    ),
}


def system_message_for(
    profile: str,
    extra: str | None,
    *,
    timezone: str = "America/New_York",
    project: str | None = None,
) -> str:
    base = PROFILE_SYSTEM.get(profile, PROFILE_SYSTEM["workspace"])
    clock = clock_payload(timezone)
    local_date = str(clock["local"])[:10]
    year = local_date[:4]
    project_label = project if project else "none (journal writes disabled)"
    calendar = (
        f"Today is {clock['weekday']}, {local_date} ({timezone}). "
        f"The current year is {year}. Use that date for 'today' / 'this week'; "
        "if the user names another year, search that year.\n"
        "Calendar tools (do not invent names; there is no project_calendar): "
        "now, calendar_search, calendar_get, calendar_add, calendar_edit, calendar_delete.\n"
        "For 'what's on the calendar this week', call calendar_search with these exact UTC values "
        f"(or omit start/end): start={clock['this_week']['start']} "
        f"end={clock['this_week']['end']}. "
        "You may call now() if you need another window (today / next_7_days).\n"
        f"This job's project is {project_label}; calendar adds are stamped with that project "
        "(or unscoped if none). "
        "You can read unscoped events; you can only add/edit/delete this job's project "
        "(or unscoped events if this job has no project)."
    )
    journal = (
        "Journal tools (do not invent names): "
        "now, journal_search, journal_get, journal_add, journal_edit, journal_delete.\n"
        f"Today's civil date is {local_date}. Use YYYY-MM-DD for journal_search start/end "
        "and for journal_add entry_date.\n"
        f"This job's project is {project_label}. "
        "You can read global (unscoped) journal entries for context. "
        "You can only add, edit, or soft-delete entries for this job's project; "
        "journal_add fails if this job has no project. "
        "Prefer journal_add at the end of a long task to record decisions; "
        "use journal_search before repeating prior work."
    )
    parts = [base, calendar, journal]
    if extra and extra.strip():
        parts.append(f"Additional instructions:\n{extra.strip()}")
    return "\n\n".join(parts)


def serialize_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return {str(k): v for k, v in args.items()}
    if args is None:
        return {}
    return {"value": str(args)}
