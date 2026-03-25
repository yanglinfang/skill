"""
Agent execution helpers for PinchBench — ZeroClaw adapter.

Replaces OpenClaw subprocess calls with HTTP requests to ZeroClaw's
gateway webhook endpoint. No external dependencies (stdlib only).

Environment variables:
    ZEROCLAW_GATEWAY_URL    Gateway URL (default: http://localhost:42617)
    ZEROCLAW_WORKSPACE      Workspace path (default: /tmp/pinchbench)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib_tasks import Task


logger = logging.getLogger(__name__)

ZEROCLAW_URL = os.environ.get("ZEROCLAW_GATEWAY_URL", "http://localhost:42617")
ZEROCLAW_WORKSPACE = os.environ.get("ZEROCLAW_WORKSPACE", "/tmp/pinchbench")


class ModelValidationError(Exception):
    """Raised when a model ID is invalid or inaccessible."""

    pass


def slugify_model(model_id: str) -> str:
    return model_id.replace("/", "-").replace(".", "-").lower()


def validate_openrouter_model(model_id: str, timeout_seconds: float = 10.0) -> bool:
    """
    Validate that a model ID exists on OpenRouter.

    Args:
        model_id: Model ID (with or without openrouter/ prefix)
        timeout_seconds: HTTP request timeout

    Returns:
        True if model is valid and accessible

    Raises:
        ModelValidationError: If model doesn't exist or validation fails
    """
    bare_model_id = model_id
    if bare_model_id.startswith("openrouter/"):
        bare_model_id = bare_model_id[len("openrouter/"):]

    if "/" not in bare_model_id:
        logger.info("Skipping model validation for non-OpenRouter model: %s", model_id)
        return True

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set, skipping model validation")
        return True

    logger.info("Validating model: %s", bare_model_id)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://pinchbench.com",
        "X-Title": "PinchBench",
    }

    encoded_model_id = bare_model_id.replace("/", "%2F")
    specific_endpoint = f"https://openrouter.ai/api/v1/models/{encoded_model_id}"
    req = urllib.request.Request(specific_endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            logger.info("Model validated: %s", bare_model_id)
            return True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            logger.warning("OpenRouter API error during validation: %s", exc)
            return True
    except urllib.error.URLError as exc:
        logger.warning("Network error during model validation: %s", exc)
        return True

    # Model not found — fetch catalog for suggestions
    catalog_endpoint = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(catalog_endpoint, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ModelValidationError(f"Model '{bare_model_id}' not found on OpenRouter.")

    models = data.get("data", [])
    model_ids = {
        mid
        for m in models
        if isinstance(m, dict)
        for mid in [m.get("id")]
        if isinstance(mid, str) and mid
    }

    if bare_model_id in model_ids:
        logger.info("Model validated via catalog fallback: %s", bare_model_id)
        return True

    close_matches = []
    bare_lower = bare_model_id.lower()
    for mid in model_ids:
        mid_lower = mid.lower()
        if mid_lower == bare_lower:
            continue
        if bare_lower in mid_lower or mid_lower in bare_lower:
            close_matches.append(mid)

    error_msg = f"Model '{bare_model_id}' not found on OpenRouter."
    if close_matches:
        close_matches_str = ", ".join(sorted(close_matches)[:5])
        error_msg += f" Did you mean: {close_matches_str}?"
    else:
        provider = bare_model_id.split("/")[0] if "/" in bare_model_id else None
        if provider:
            provider_models = [m for m in model_ids if m.startswith(f"{provider}/")]
            if provider_models:
                error_msg += (
                    f" Available {provider} models: {', '.join(sorted(provider_models)[:5])}"
                )

    raise ModelValidationError(error_msg)


# ── ZeroClaw gateway helpers ──────────────────────────────────────


def _zeroclaw_workspace() -> Path:
    """Return the workspace path where ZeroClaw reads/writes files."""
    return Path(ZEROCLAW_WORKSPACE) / "workspace"


def _send_to_zeroclaw(message: str, timeout_seconds: float) -> str:
    """Send a message to ZeroClaw's webhook and return the response text."""
    url = f"{ZEROCLAW_URL}/webhook"
    payload = json.dumps({"message": message}).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[HTTP {e.code}] {body}"
    except urllib.error.URLError as e:
        return f"[Connection Error] {e.reason}"
    except TimeoutError:
        return "[Timeout] Request exceeded time limit"


def _build_transcript(prompt: str, response: str) -> List[Dict[str, Any]]:
    """Build a PinchBench-compatible transcript from a prompt/response pair."""
    return [
        {
            "type": "message",
            "message": {"role": "user", "content": prompt},
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": response,
            },
        },
    ]


# ── Public API (called by benchmark.py) ──────────────────────────


def _get_agent_workspace(agent_id: str) -> Path | None:
    """Return ZeroClaw's workspace path."""
    return _zeroclaw_workspace()


def ensure_agent_exists(agent_id: str, model_id: str, workspace_dir: Path) -> bool:
    """Verify ZeroClaw is running and healthy. No agent creation needed."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(f"{ZEROCLAW_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                logger.info("ZeroClaw healthy at %s", ZEROCLAW_URL)
                return True
            else:
                raise RuntimeError(f"ZeroClaw unhealthy: {data}")
    except Exception as exc:
        raise RuntimeError(
            f"ZeroClaw not running at {ZEROCLAW_URL}: {exc}\n"
            "Start it with: docker compose -f docker-compose.minimal.yml up -d"
        )


def cleanup_agent_sessions(agent_id: str) -> None:
    """No-op — ZeroClaw doesn't persist session transcripts on the host."""
    pass


def prepare_task_workspace(skill_dir: Path, run_id: str, task: Task, agent_id: str) -> Path:
    """
    Prepare workspace for a task by copying fixtures into ZeroClaw's workspace.
    """
    import shutil

    workspace = _zeroclaw_workspace()
    workspace.mkdir(parents=True, exist_ok=True)

    # Remove only task-created files, not ZeroClaw's state (memory DB, sessions, etc.)
    # Preserve directories starting with '.' and known state files.
    for item in list(workspace.iterdir()):
        name = item.name
        # Keep ZeroClaw internal state
        if name.startswith(".") or name in ("memory", "state", "cron", "sessions", "skills"):
            continue
        if name.endswith(".db") or name.endswith(".db-wal") or name.endswith(".db-shm"):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for file_spec in task.workspace_files:
        if "content" in file_spec:
            dest = workspace / file_spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(file_spec["content"])
            continue

        source = skill_dir / "assets" / file_spec["source"]
        dest = workspace / file_spec["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(source.read_bytes())
        except FileNotFoundError:
            logger.error("Workspace file not found: %s", source)
            raise

    # Remove bootstrap files that interfere with benchmark tasks
    for bootstrap_file in ["BOOTSTRAP.md", "SOUL.md", "USER.md", "IDENTITY.md"]:
        bootstrap_path = workspace / bootstrap_file
        if bootstrap_path.exists():
            try:
                bootstrap_path.unlink()
                logger.info("Removed bootstrap file: %s", bootstrap_file)
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", bootstrap_file, exc)

    return workspace


def execute_openclaw_task(
    *,
    task: Task,
    agent_id: str,
    model_id: str,
    run_id: str,
    timeout_multiplier: float,
    skill_dir: Path,
    verbose: bool = False,
) -> Dict[str, Any]:
    logger.info("Agent [%s] starting task: %s", agent_id, task.task_id)
    logger.info("   Task: %s", task.name)
    logger.info("   Category: %s", task.category)
    if verbose:
        logger.info(
            "   Prompt: %s", task.prompt[:500] + "..." if len(task.prompt) > 500 else task.prompt
        )

    start_time = time.time()
    workspace = prepare_task_workspace(skill_dir, run_id, task, agent_id)
    timeout_seconds = task.timeout_seconds * timeout_multiplier
    timed_out = False
    transcript: List[Dict[str, Any]] = []
    response_text = ""

    # Check if this is a multi-session task
    sessions = task.frontmatter.get("sessions", [])
    if sessions:
        logger.info("Multi-session task with %d sessions", len(sessions))
        for i, session_entry in enumerate(sessions, 1):
            if isinstance(session_entry, str):
                session_prompt = session_entry
            elif isinstance(session_entry, dict):
                session_prompt = session_entry.get("prompt") or session_entry.get("message", "")
            else:
                logger.warning("Skipping invalid session entry: %s", session_entry)
                continue

            logger.info("   Session %d/%d", i, len(sessions))
            elapsed = time.time() - start_time
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                timed_out = True
                break

            response_text = _send_to_zeroclaw(session_prompt, remaining)
            transcript.extend(_build_transcript(session_prompt, response_text))

            if response_text.startswith("[HTTP ") or response_text.startswith("[Connection"):
                break
    else:
        response_text = _send_to_zeroclaw(task.prompt, timeout_seconds)
        transcript = _build_transcript(task.prompt, response_text)

    execution_time = time.time() - start_time

    # Determine status
    status = "success"
    if timed_out or response_text.startswith("[Timeout]"):
        status = "timeout"
    elif response_text.startswith("[HTTP ") or response_text.startswith("[Connection"):
        status = "error"

    # Usage — ZeroClaw webhook doesn't return token counts in the response,
    # so we report request count only. Check OpenRouter dashboard for exact costs.
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": len(sessions) if sessions else 1,
    }

    # Verbose logging
    if verbose:
        logger.info("   [VERBOSE] Execution time: %.2fs", execution_time)
        logger.info("   [VERBOSE] Execution status: %s", status)
        logger.info("   [VERBOSE] Workspace: %s", workspace)
        logger.info("   [VERBOSE] Transcript entries: %d", len(transcript))

        for entry in transcript:
            if entry.get("type") == "message":
                msg = entry.get("message", {})
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "assistant":
                    preview = content[:500] + "..." if len(content) > 500 else content
                    logger.info("   [VERBOSE] Agent response: %s", preview)
                elif role == "user":
                    preview = content[:200] + "..." if len(content) > 200 else content
                    logger.info("   [VERBOSE] User message: %s", preview)

        if workspace.exists():
            logger.info("   [VERBOSE] Workspace files after task:")
            for f in sorted(workspace.rglob("*")):
                if f.is_file():
                    try:
                        size = f.stat().st_size
                        logger.info("      %s (%d bytes)", f.relative_to(workspace), size)
                    except OSError:
                        logger.info("      %s", f.relative_to(workspace))

    return {
        "agent_id": agent_id,
        "task_id": task.task_id,
        "status": status,
        "transcript": transcript,
        "usage": usage,
        "workspace": str(workspace),
        "exit_code": 0 if status == "success" else 1,
        "timed_out": timed_out,
        "execution_time": execution_time,
        "stdout": response_text,
        "stderr": "",
    }


def _call_openrouter_direct(prompt: str, model: str, timeout_seconds: float) -> str:
    """Call OpenRouter API directly (no ZeroClaw). Used for LLM judge."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY", "")
    if not api_key:
        return "[Error] No API key set (OPENROUTER_API_KEY or API_KEY)"

    # Strip openrouter/ prefix if present
    bare_model = model
    if bare_model.startswith("openrouter/"):
        bare_model = bare_model[len("openrouter/"):]

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = json.dumps({
        "model": bare_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://pinchbench.com",
        "X-Title": "PinchBench-Judge",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read())
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return "[Error] No choices in response"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"[HTTP {e.code}] {body}"
    except Exception as e:
        return f"[Error] {e}"


def run_openclaw_prompt(
    *,
    agent_id: str,
    prompt: str,
    workspace: Path,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Run a single prompt for helper agents like the judge.

    Calls OpenRouter directly (not through ZeroClaw) since the judge
    doesn't need tools — just text-in, text-out evaluation. The model
    is extracted from the agent_id (e.g. 'bench-judge-anthropic-claude-sonnet-4-6').
    """
    start_time = time.time()
    workspace.mkdir(parents=True, exist_ok=True)

    # Extract model from agent_id: bench-judge-{model_slug}
    # The judge agent_id format is "{prefix}-{slugified_model}"
    # We need the original model ID, which we reconstruct or use a fallback.
    # The grading module passes the model via ensure_agent_exists, but we
    # can't recover it from the slug reliably. Instead, send through ZeroClaw
    # for the default model, or use direct OpenRouter if JUDGE_MODEL is set.
    judge_model = os.environ.get("PINCHBENCH_JUDGE_MODEL", "")
    if judge_model:
        response_text = _call_openrouter_direct(prompt, judge_model, timeout_seconds)
    else:
        # Fallback: use ZeroClaw webhook (uses whatever model is configured)
        response_text = _send_to_zeroclaw(prompt, timeout_seconds)

    transcript = _build_transcript(prompt, response_text)
    execution_time = time.time() - start_time

    status = "success"
    if response_text.startswith("[Timeout]"):
        status = "timeout"
    elif response_text.startswith("[HTTP ") or response_text.startswith("[Connection"):
        status = "error"

    return {
        "agent_id": agent_id,
        "status": status,
        "transcript": transcript,
        "workspace": str(workspace),
        "exit_code": 0 if status == "success" else 1,
        "timed_out": status == "timeout",
        "execution_time": execution_time,
        "stdout": response_text,
        "stderr": "",
    }
