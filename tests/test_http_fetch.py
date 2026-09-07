from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from ownrun_runner.agent.tools import (
    PublicTarget,
    fetch_public_url,
    http_get_pinned,
    resolve_public_url,
)


class ResolvePublicUrlTests(unittest.TestCase):
    def test_rejects_loopback_and_lan_literals(self) -> None:
        blocked = (
            "http://127.0.0.1/",
            "http://127.0.0.1:11434/api/tags",
            "http://localhost/",
            "http://[::1]/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "http://169.254.169.254/",
            "http://metadata.google.internal/",
        )
        for url in blocked:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    resolve_public_url(url)

    def test_rejects_non_http_and_credentials(self) -> None:
        with self.assertRaises(ValueError):
            resolve_public_url("file:///etc/passwd")
        with self.assertRaises(ValueError):
            resolve_public_url("http://user:pass@8.8.8.8/")

    def test_pins_public_literal(self) -> None:
        target = resolve_public_url("http://8.8.8.8:53/lookup?q=1#frag")
        self.assertEqual(target.ip, "8.8.8.8")
        self.assertEqual(target.hostname, "8.8.8.8")
        self.assertEqual(target.port, 53)
        self.assertEqual(target.request_path, "/lookup?q=1")
        self.assertNotIn("#", target.url)

    def test_redirect_target_to_loopback_is_rejected(self) -> None:
        from urllib.parse import urljoin

        nxt = urljoin("https://example.com/page", "http://127.0.0.1:7860/sdapi/v1/sd-models")
        with self.assertRaises(ValueError) as ctx:
            resolve_public_url(nxt)
        self.assertIn("127.0.0.1", str(ctx.exception))


class PinnedGetTests(unittest.TestCase):
    def setUp(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/secret":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"should-not-follow")
                    return
                if self.path == "/redir":
                    self.send_response(302)
                    self.send_header("Location", "/secret")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                body = f"host={self.headers.get('Host')} path={self.path}"
                self.wfile.write(body.encode())

            def log_message(self, *_args: object) -> None:
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _target(self, path: str) -> PublicTarget:
        return PublicTarget(
            url=f"http://example.test:{self.port}{path}",
            hostname="example.test",
            port=self.port,
            scheme="http",
            ip="127.0.0.1",
            request_path=path,
        )

    def test_connects_to_pinned_ip_with_url_host_header(self) -> None:
        status, _headers, body = http_get_pinned(self._target("/ok"), 2)
        self.assertEqual(status, 200)
        self.assertIn(f"host=example.test:{self.port}", body.decode())
        self.assertIn("path=/ok", body.decode())

    def test_does_not_follow_redirects(self) -> None:
        status, headers, body = http_get_pinned(self._target("/redir"), 2)
        self.assertEqual(status, 302)
        self.assertEqual(headers.get("location"), "/secret")
        self.assertNotIn(b"should-not-follow", body)


class FetchPublicUrlTests(unittest.TestCase):
    def test_refuses_redirect_to_private_ip(self) -> None:
        def getter(_target: PublicTarget, _timeout: float) -> tuple[int, dict[str, str], bytes]:
            return 302, {"location": "http://127.0.0.1:11434/api/tags"}, b""

        with self.assertRaises(ValueError) as ctx:
            fetch_public_url("http://8.8.8.8/", 2, get=getter)
        self.assertIn("127.0.0.1", str(ctx.exception))

    def test_follows_relative_redirect_then_returns_body(self) -> None:
        hops: list[str] = []

        def getter(target: PublicTarget, _timeout: float) -> tuple[int, dict[str, str], bytes]:
            hops.append(target.request_path)
            if target.request_path == "/a":
                return 302, {"location": "/b"}, b""
            return 200, {"content-type": "text/plain"}, b"ok-from-b"

        text = fetch_public_url("http://8.8.8.8/a", 2, get=getter)
        self.assertEqual(text, "ok-from-b")
        self.assertEqual(hops, ["/a", "/b"])

    def test_caps_redirects(self) -> None:
        def getter(_target: PublicTarget, _timeout: float) -> tuple[int, dict[str, str], bytes]:
            return 302, {"location": "http://8.8.8.8/next"}, b""

        self.assertEqual(
            fetch_public_url("http://8.8.8.8/", 2, get=getter),
            "Error: too many redirects",
        )


if __name__ == "__main__":
    unittest.main()
