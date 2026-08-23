from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

from ownrun_runner.agent.tools import (
    TOOL_RESULT_MAX,
    Workspace,
    build_tools,
    profile_flags,
    serialize_args,
    system_message_for,
    truncate,
)
from ownrun_runner.config import Config

log = logging.getLogger(__name__)


def _model_info_for(model: str) -> dict[str, Any]:
    try:
        from autogen_ext.models.ollama._model_info import get_info

        info = get_info(model)
        if not info.get("function_calling", False):
            raise RuntimeError(
                f"Model {model!r} does not support tool/function calling. "
                "Pick a tools-capable Ollama model (e.g. llama3.1, llama3.2, qwen2.5)."
            )
        return dict(info)
    except KeyError:
        # Unknown tag — assume tools and let Ollama fail clearly if not.
        return {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        }


def agent_capable_models(models: list[str]) -> list[str]:
    capable: list[str] = []
    try:
        from autogen_ext.models.ollama._model_info import get_info
    except ImportError:
        return list(models)

    for name in models:
        try:
            info = get_info(name)
            if info.get("function_calling", False):
                capable.append(name)
        except KeyError:
            # Unknown to Autogen — still offer; adapter supplies model_info.
            capable.append(name)
        except Exception:
            capable.append(name)
    return capable


def _event_to_trace_steps(event: Any) -> list[dict[str, Any]]:
    from autogen_agentchat.messages import (
        TextMessage,
        ToolCallExecutionEvent,
        ToolCallRequestEvent,
        ToolCallSummaryMessage,
    )

    steps: list[dict[str, Any]] = []
    etype = type(event).__name__

    if isinstance(event, ToolCallRequestEvent):
        for call in event.content:
            name = getattr(call, "name", None) or "unknown"
            args = getattr(call, "arguments", None)
            if isinstance(args, str):
                import json

                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    parsed = {"raw": args}
            else:
                parsed = serialize_args(args)
            steps.append({"kind": "tool_call", "name": name, "args": parsed})
        return steps

    if isinstance(event, ToolCallExecutionEvent):
        for result in event.content:
            name = getattr(result, "name", None) or "unknown"
            content = truncate(str(getattr(result, "content", "")))
            is_error = bool(getattr(result, "is_error", False))
            steps.append(
                {
                    "kind": "tool_result",
                    "name": name,
                    "ok": not is_error,
                    "content": content,
                }
            )
        return steps

    if isinstance(event, ToolCallSummaryMessage):
        # Reflection / final after tools — prefer later TextMessage if present.
        content = truncate(str(event.content))
        if content:
            steps.append(
                {
                    "kind": "message",
                    "source": getattr(event, "source", "ownrun_assistant"),
                    "content": content,
                }
            )
        return steps

    if isinstance(event, TextMessage):
        source = getattr(event, "source", "") or "assistant"
        content = truncate(str(event.content))
        if source == "user":
            steps.append({"kind": "user", "content": content})
        else:
            steps.append({"kind": "message", "source": source, "content": content})
        return steps

    # Ignore streaming chunks and other events.
    if etype in {"ModelClientStreamingChunkEvent", "ThoughtEvent", "StopMessage"}:
        return []
    return []


def _final_text_from_trace(trace: list[dict[str, Any]], task_result: Any) -> str:
    if task_result is not None:
        messages = getattr(task_result, "messages", None) or []
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            source = getattr(msg, "source", "")
            if content and source != "user" and type(msg).__name__ in {
                "TextMessage",
                "ToolCallSummaryMessage",
            }:
                return truncate(str(content), TOOL_RESULT_MAX * 4)
    for step in reversed(trace):
        if step.get("kind") in {"message", "user"} and step.get("kind") == "message":
            return str(step.get("content") or "")
    return ""


def _task_from_request(request: dict[str, Any]) -> str | list[Any]:
    from autogen_agentchat.messages import TextMessage

    raw = request.get("messages")
    if isinstance(raw, list) and raw:
        msgs: list[TextMessage] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", ""))
            content = item.get("content")
            if content is None:
                continue
            if role == "user":
                msgs.append(TextMessage(content=str(content), source="user"))
            elif role == "assistant":
                msgs.append(TextMessage(content=str(content), source="ownrun_assistant"))
        if msgs:
            return msgs

    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Agent job missing prompt")
    return prompt


async def _run_agent_async(
    config: Config,
    request: dict[str, Any],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> dict[str, Any]:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import TaskResult
    from autogen_core import CancellationToken
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    model = request.get("model")
    if not model:
        raise ValueError("Job request missing model")
    profile = str(request.get("profile") or "").strip()
    if profile not in config.agent_profiles:
        raise ValueError(f"Profile {profile!r} is not enabled on this runner")

    include_ws, include_web = profile_flags(profile)
    max_iter = int(request.get("max_tool_iterations") or 8)
    max_iter = max(1, min(max_iter, 20))

    system_extra = request.get("system_prompt")
    system_message = system_message_for(
        profile, str(system_extra) if system_extra else None
    )

    workspace = Workspace(Path(config.agent_workspace))
    tools = build_tools(
        workspace,
        include_workspace=include_ws,
        include_web=include_web,
        fetch_timeout=config.agent_fetch_timeout,
    )

    model_info = _model_info_for(str(model))
    client = OllamaChatCompletionClient(
        model=str(model),
        host=config.ollama_base_url,
        model_info=model_info,
        timeout=config.job_timeout,
    )

    agent = AssistantAgent(
        name="ownrun_assistant",
        model_client=client,
        tools=tools,
        system_message=system_message,
        reflect_on_tool_use=True,
        max_tool_iterations=max_iter,
        model_client_stream=False,
    )

    cancel_token = CancellationToken()
    task = _task_from_request(request)
    trace: list[dict[str, Any]] = []

    start = time.monotonic()
    last_progress = 0.0
    stop_reason = "completed"
    task_result: TaskResult | None = None
    iterations = 0

    try:
        stream = agent.run_stream(task=task, cancellation_token=cancel_token)
        async for event in stream:
            if should_cancel and should_cancel():
                cancel_token.cancel()
                stop_reason = "cancelled"
                break

            elapsed = time.monotonic() - start
            if elapsed >= config.agent_wall_clock:
                cancel_token.cancel()
                stop_reason = "timeout"
                break

            if isinstance(event, TaskResult):
                task_result = event
                reason = getattr(event, "stop_reason", None)
                if stop_reason == "completed" and reason:
                    text_reason = str(reason).lower()
                    if "max" in text_reason:
                        stop_reason = "max_iterations"
                continue

            steps = _event_to_trace_steps(event)
            if steps:
                for step in steps:
                    if step.get("kind") == "tool_call":
                        iterations += 1
                        log.info("Tool call: %s", step.get("name"))
                    trace.append(step)

                now = time.monotonic()
                if on_progress and (now - last_progress >= 2.0):
                    partial = {
                        "text": _final_text_from_trace(trace, task_result),
                        "model": str(model),
                        "profile": profile,
                        "duration_ms": int((now - start) * 1000),
                        "stop_reason": "running",
                        "iterations": iterations,
                        "trace": list(trace),
                    }
                    try:
                        on_progress(partial)
                    except Exception as e:
                        log.warning("Progress upload failed: %s", e)
                    last_progress = now

        if stop_reason == "completed" and iterations >= max_iter:
            stop_reason = "max_iterations"

    except Exception as e:
        if should_cancel and should_cancel():
            stop_reason = "cancelled"
        elif stop_reason in {"cancelled", "timeout"}:
            pass
        elif "cancel" in str(e).lower():
            stop_reason = "cancelled"
        else:
            raise
    finally:
        try:
            await client.close()
        except Exception:
            pass

    if stop_reason == "cancelled":
        raise RuntimeError("cancelled")

    duration_ms = int((time.monotonic() - start) * 1000)
    text = _final_text_from_trace(trace, task_result)
    if not text:
        text = "(no final answer)"

    return {
        "text": text,
        "model": str(model),
        "profile": profile,
        "duration_ms": duration_ms,
        "stop_reason": stop_reason,
        "iterations": iterations,
        "trace": trace,
    }


def run_agent_task(
    config: Config,
    request: dict[str, Any],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        _run_agent_async(
            config,
            request,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
    )
