from __future__ import annotations

import ipaddress
import logging
import re
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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


def build_tools(
    workspace: Workspace | None,
    *,
    include_workspace: bool,
    include_web: bool,
    fetch_timeout: float,
) -> list[Callable[..., str]]:
    tools: list[Callable[..., str]] = []

    def now() -> str:
        """Return the current UTC date and time in ISO-8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tools.append(now)

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


def system_message_for(profile: str, extra: str | None) -> str:
    base = PROFILE_SYSTEM.get(profile, PROFILE_SYSTEM["workspace"])
    if extra and extra.strip():
        return f"{base}\n\nAdditional instructions:\n{extra.strip()}"
    return base


def serialize_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return {str(k): v for k, v in args.items()}
    if args is None:
        return {}
    return {"value": str(args)}
