from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from ownrun_runner.agent.tools import (
    TOOL_RESULT_MAX,
    build_tools,
    compute_effective_tools,
    profile_flags,
    project_workspace,
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


def _plain_user_content(msg: Any) -> str | None:
    """Return user-turn text that Qwen templates count as a real query, or None."""
    role = getattr(msg, "role", None)
    if role != "user":
        return None
    text = str(getattr(msg, "content", None) or "").strip()
    if not text:
        return None
    if text.startswith("<tool_response>") and text.endswith("</tool_response>"):
        return None
    return text


def _has_plain_user_query(messages: Sequence[Any]) -> bool:
    return any(_plain_user_content(m) is not None for m in messages)


def _ensure_user_query(messages: Sequence[Any], fallback: str) -> list[Any]:
    """Keep a plain user turn in the Ollama payload.

    Qwen3.5/3.8 Jinja templates 500 if no non-``<tool_response>`` user turn
    remains. That happens when Autogen omits the user message, or when Ollama
    truncates it to fit ``num_ctx`` after a large tool result.
    """
    msgs = list(messages)
    has_query = _has_plain_user_query(msgs)
    last_role = getattr(msgs[-1], "role", None) if msgs else None
    last_is_tool = last_role == "tool" or (
        bool(msgs) and last_role == "user" and _plain_user_content(msgs[-1]) is None
    )

    if has_query and not last_is_tool:
        return msgs

    from ollama import Message

    if not has_query:
        text = (fallback or "").strip() or "Please continue the task."
        log.warning(
            "Ollama chat payload had no user query; inserting fallback for Qwen templates"
        )
        msgs.append(Message(role="user", content=text))
        return msgs

    # Tool follow-up: a trailing user turn survives truncation of the original.
    msgs.append(Message(role="user", content="Continue from the tool results above."))
    return msgs


def _fallback_user_query(request: dict[str, Any], task: str | list[Any]) -> str:
    prompt = str(request.get("prompt") or "").strip()
    if prompt:
        return prompt
    if isinstance(task, str) and task.strip():
        return task.strip()
    if isinstance(task, list):
        for item in reversed(task):
            content = str(getattr(item, "content", "") or "").strip()
            source = getattr(item, "source", "")
            if source == "user" and content:
                return content
    return "Please continue the task."


def _ownrun_ollama_client(fallback_user_query: str, **kwargs: Any) -> Any:
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    class OwnRunOllamaClient(OllamaChatCompletionClient):
        def __init__(self, **inner: Any) -> None:
            super().__init__(**inner)

        def _process_create_args(self, *args: Any, **kw: Any) -> Any:  # type: ignore[override]
            params = super()._process_create_args(*args, **kw)
            params.messages = _ensure_user_query(params.messages, fallback_user_query)
            return params

    return OwnRunOllamaClient(**kwargs)


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
    job_id: int,
    calendar_client: Any | None,
    on_progress: Callable[[dict[str, Any]], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> dict[str, Any]:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import TaskResult
    from autogen_core import CancellationToken

    model = request.get("model")
    if not model:
        raise ValueError("Job request missing model")
    profile = str(request.get("profile") or "").strip()
    if profile not in config.agent_profiles:
        raise ValueError(f"Profile {profile!r} is not enabled on this runner")

    include_ws, include_web = profile_flags(profile)
    max_iter = int(request.get("max_tool_iterations") or 8)
    max_iter = max(1, min(max_iter, 20))

    project_raw = request.get("project")
    project: str | None = None
    if include_ws:
        if not project_raw or not str(project_raw).strip():
            raise ValueError(
                "Agent jobs with file tools require a project "
                "(subdirectory under OWNRUN_AGENT_WORKSPACE)"
            )
        project = str(project_raw).strip()
    elif project_raw and str(project_raw).strip():
        # Optional on web-only profiles; still validate if provided.
        project = str(project_raw).strip()

    project_disabled: list[str] = []
    if calendar_client is not None:
        try:
            project_disabled = calendar_client.tool_policy(job_id)
        except Exception as e:
            log.warning("Job #%s tool policy fetch failed: %s", job_id, e)

    job_disabled_raw = request.get("disabled_tools")
    job_disabled: list[str] = []
    if isinstance(job_disabled_raw, list):
        job_disabled = [str(x) for x in job_disabled_raw if str(x).strip()]

    effective = compute_effective_tools(
        profile,
        agent_web_search=config.agent_web_search,
        project_disabled=project_disabled,
        job_disabled=job_disabled,
    )

    if "web_search" in effective and not config.ollama_api_key:
        raise ValueError(
            "OLLAMA_API_KEY is required for web search on this job. "
            "Set OLLAMA_API_KEY on the runner or disable web search."
        )

    system_extra = request.get("system_prompt")
    system_message = system_message_for(
        profile,
        str(system_extra) if system_extra else None,
        timezone=config.timezone,
        project=project,
        effective_tool_ids=effective,
    )

    if include_ws:
        assert project is not None
        workspace = project_workspace(Path(config.agent_workspace), project)
        log.info("Agent project sandbox: %s", workspace.root)
    else:
        workspace = None

    tools = build_tools(
        workspace,
        include_workspace=include_ws,
        include_web=include_web,
        fetch_timeout=config.agent_fetch_timeout,
        timezone_name=config.timezone,
        calendar_client=calendar_client,
        job_id=job_id,
        effective_tool_ids=effective,
        ollama_api_key=config.ollama_api_key or None,
    )

    model_info = _model_info_for(str(model))
    options: dict[str, Any] = {"num_predict": config.ollama_num_predict}
    if config.ollama_num_ctx > 0:
        options["num_ctx"] = config.ollama_num_ctx
    ollama_kwargs: dict[str, Any] = {
        "model": str(model),
        "host": config.ollama_base_url,
        "model_info": model_info,
        "timeout": config.job_timeout,
        "options": options,
    }
    if not config.ollama_think:
        # Qwen3.x "thinking" models can burn the output budget before tool-call
        # JSON finishes; Ollama then returns HTTP 500 (invalid tool arguments).
        ollama_kwargs["think"] = False

    task = _task_from_request(request)
    client = _ownrun_ollama_client(
        _fallback_user_query(request, task),
        **ollama_kwargs,
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
    if config.ollama_num_ctx > 0:
        log.info("Agent Ollama num_ctx=%s", config.ollama_num_ctx)
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
                    if project:
                        partial["project"] = project
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

    result: dict[str, Any] = {
        "text": text,
        "model": str(model),
        "profile": profile,
        "duration_ms": duration_ms,
        "stop_reason": stop_reason,
        "iterations": iterations,
        "trace": trace,
    }
    if project:
        result["project"] = project
    return result


def run_agent_task(
    config: Config,
    request: dict[str, Any],
    *,
    job_id: int,
    calendar_client: Any | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        _run_agent_async(
            config,
            request,
            job_id=job_id,
            calendar_client=calendar_client,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
    )
