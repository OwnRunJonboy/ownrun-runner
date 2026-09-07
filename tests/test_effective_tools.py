from __future__ import annotations

import unittest

from ownrun_runner.agent.tools import compute_effective_tools, system_message_for


class EffectiveToolsTests(unittest.TestCase):
    def test_web_default_drops_http_fetch(self) -> None:
        tools = compute_effective_tools("web", agent_web_search=True)
        self.assertIn("journal_search", tools)
        self.assertIn("calendar_search", tools)
        self.assertIn("task_search", tools)
        self.assertIn("web_search", tools)
        self.assertNotIn("http_fetch", tools)

    def test_workspace_web_default_drops_http_fetch(self) -> None:
        tools = compute_effective_tools("workspace_web", agent_web_search=True)
        self.assertIn("read_file", tools)
        self.assertIn("journal_search", tools)
        self.assertNotIn("http_fetch", tools)

    def test_opt_in_restores_http_fetch(self) -> None:
        tools = compute_effective_tools(
            "web",
            agent_web_search=True,
            allow_http_fetch=True,
        )
        self.assertIn("http_fetch", tools)
        self.assertIn("journal_search", tools)

    def test_project_policy_can_still_disable_fetch(self) -> None:
        tools = compute_effective_tools(
            "web",
            agent_web_search=True,
            allow_http_fetch=True,
            project_disabled=["http_fetch"],
        )
        self.assertNotIn("http_fetch", tools)

    def test_fetch_allowed_when_personal_tools_all_disabled(self) -> None:
        personal = [
            "calendar_search",
            "calendar_get",
            "calendar_add",
            "calendar_edit",
            "calendar_delete",
            "journal_search",
            "journal_get",
            "journal_add",
            "journal_edit",
            "journal_delete",
            "task_search",
            "task_get",
            "task_add",
            "task_edit",
            "task_delete",
        ]
        tools = compute_effective_tools(
            "web",
            agent_web_search=True,
            job_disabled=personal,
        )
        self.assertIn("http_fetch", tools)
        self.assertFalse(any(t.startswith("journal_") for t in tools))

    def test_workspace_never_has_http_fetch(self) -> None:
        tools = compute_effective_tools(
            "workspace",
            agent_web_search=True,
            allow_http_fetch=True,
        )
        self.assertNotIn("http_fetch", tools)

    def test_system_prompt_omits_fetch_when_off(self) -> None:
        tools = compute_effective_tools("web", agent_web_search=True)
        msg = system_message_for("web", None, effective_tool_ids=tools)
        self.assertIn("web_search", msg)
        self.assertIn("do not have http_fetch", msg.lower())


if __name__ == "__main__":
    unittest.main()
