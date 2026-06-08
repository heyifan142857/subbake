"""Intent classification and confidence gating for SubBakeAgent.

The intent classifier determines the user's goal from natural language input,
then either dispatches directly (high confidence) or falls through to the
agent loop. The confidence gate prevents low-confidence mutations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from subbake.models.base_model import MockBackend
from subbake import runtime_options as _runtime_options
from subbake.cancellation import OperationCancelledError
from .loop import AgentLoopState
from .text_helpers import extract_references
from .tool_registry import ALWAYS_AVAILABLE_TOOLS, TOOL_CATEGORIES

# Re-exported from __init__
CONFIDENCE_LOW_THRESHOLD = 0.4
CONFIDENCE_MEDIUM_THRESHOLD = 0.7
CONFIDENCE_MIN_OBSERVATIONS = 2

VALID_INTENT_CATEGORIES = frozenset({
    "translate_file", "translate_series", "edit_subtitle",
    "diagnose", "file_operation", "browse", "profile", "chat",
})


def classify_intent(
    agent: SubBakeAgent,
    line: str,
) -> dict[str, Any] | None:
    """Lightweight intent classification. Returns intent dict or None to skip gate."""
    backend = _runtime_options.build_backend_from_values(agent.values)
    if backend is None:
        return _fallback_intent_classification(agent, line)
    if isinstance(backend, MockBackend):
        return _mock_classify_intent(agent, line)

    context = {
        "message": line,
        "cwd": str(agent.cwd),
        "profile": agent.profile,
        "recent_events": agent.session.events[-4:],
    }
    system_prompt = (
        "You are a classifier for a subtitle translation agent. "
        "Classify the user's request into exactly one category and extract parameters.\n"
        "Categories:\n"
        "- translate_file: Translate a single subtitle file\n"
        "- translate_series: Translate a series/folder of subtitle files\n"
        "- edit_subtitle: Edit/post-process an already-generated subtitle\n"
        "- diagnose: Analyze failure logs or subtitle files\n"
        "- file_operation: Create, append, replace, rename, or delete files\n"
        "- browse: List, search, or read files\n"
        "- profile: Switch or list model profiles\n"
        "- chat: General conversation, no tool needed\n\n"
        "Return JSON only with: category, confidence (0-1), parameters (dict), and reason.\n"
        "Extract file paths, language names, format preferences from natural language."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    try:
        payload, _ = agent._generate_json(backend, messages)
    except OperationCancelledError:
        raise
    except Exception:
        return None

    if not isinstance(payload, dict) or "category" not in payload:
        return None
    category = str(payload.get("category", ""))
    if category not in VALID_INTENT_CATEGORIES:
        return None
    return {
        "category": category,
        "parameters": dict(payload.get("parameters", {})),
        "confidence": float(payload.get("confidence", 0.5)),
        "reason": str(payload.get("reason", "")),
    }


def _mock_classify_intent(
    agent: SubBakeAgent,
    line: str,
) -> dict[str, Any] | None:
    """Keyword-based intent classification for mock backend.
    Only returns classification when confident enough to skip the agent loop.
    Returns None for ambiguous cases, letting existing agent loop handle them."""

    lowered = line.casefold()
    references = extract_references(agent, line)
    has_refs = bool(references)
    has_dir = any(r.is_dir() for r in references)
    has_file = any(r.is_file() for r in references)

    first_ref = str(references[0]) if references else ""

    # Clear translation with directory reference → skip agent loop
    if has_dir and any(w in lowered for w in ("翻译", "translate", "series")):
        return {"category": "translate_series", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "Directory reference"}
    # Edit with reference — must come before translate check to avoid
    # matching "translate" inside "translated" in file paths
    if has_refs and any(w in lowered for w in ("编辑", "修改", "edit", "fix", "change")):
        return {"category": "edit_subtitle", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "Edit reference"}  # fmt: skip
    # Clear translation with file reference → skip agent loop
    if has_file and any(
        lowered.startswith(w) or f" {w}" in lowered
        for w in ("翻译", "translate")
    ):
        return {"category": "translate_file", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "File reference"}
    # Explicit diagnose with reference → skip agent loop
    if has_refs and any(w in lowered for w in ("诊断", "分析", "diagnose", "error", "log", "failure")):
        return {"category": "diagnose", "parameters": {"path": first_ref}, "confidence": 0.9, "reason": "Diagnosis reference"}  # fmt: skip
    # Everything else → let agent loop handle
    return None


def _fallback_intent_classification(
    agent: SubBakeAgent,
    line: str,
) -> dict[str, Any] | None:
    """Fallback when no backend is available. Only fires for clear cases with references."""
    lowered = line.casefold()
    references = extract_references(agent, line)
    has_refs = bool(references)
    if has_refs and any(w in lowered for w in ("翻译", "translate")):
        paths = [str(r) for r in references]
        if len(references) == 1 and references[0].is_dir():
            return {"category": "translate_series", "parameters": {"path": paths[0]}, "confidence": 0.5, "reason": "Fallback: translate dir"}  # fmt: skip
        return {"category": "translate_file", "parameters": {"path": paths[0]}, "confidence": 0.5, "reason": "Fallback: translate ref"}  # fmt: skip
    return None


def intent_to_decision(
    agent: SubBakeAgent,
    intent: dict[str, Any],
    line: str,
    *,
    run_agent_loop: Callable[[str, AgentLoopState | None], dict[str, Any]],
    agent_loop_max_steps: int,
) -> dict[str, Any]:
    """Convert intent classification to an agent decision."""
    category = intent["category"]
    params = intent["parameters"]
    reason = intent.get("reason", "")

    if category == "chat":
        return {"action": "respond", "message": "Hello! I'm SubBake, your subtitle translation assistant. How can I help you today?"}

    allowed_tools: set[str] = set(ALWAYS_AVAILABLE_TOOLS)
    allowed_tools.update(TOOL_CATEGORIES.get(category, []))

    intent_confidence = intent.get("confidence", 0.5)
    if intent_confidence < 0.4:
        return {"action": "ask_user", "message": f"I'm not sure what you want to do. Could you clarify?\n\n({reason})"}

    pre_args = _prepopulate_args_from_intent_parameters(params)

    if intent_confidence >= 0.85 and _has_required_args(category, pre_args):
        return {
            "action": "final_tool_call",
            "tool_name": _category_to_default_tool(category),
            "arguments": pre_args,
            "message": reason or f"Proceeding with {category}.",
            "confidence": intent_confidence,
            "reason": reason,
        }

    state = AgentLoopState(
        original_user_message=line,
        max_steps=agent_loop_max_steps,
        current_mode=agent.session.mode,
        allowed_tools=tuple(sorted(allowed_tools)),
        pre_populated_arguments=pre_args,
        intent_hint={
            "category": category,
            "confidence": intent_confidence,
            "reason": reason,
        },
    )
    return run_agent_loop(line, state=state)


def _prepopulate_args_from_intent_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert intent-extracted parameters into tool argument format."""
    args: dict[str, Any] = {}
    for key in ("path", "target_language", "source_language", "output_format", "pattern", "query", "content", "old", "new", "old_path", "new_path", "instruction", "text"):
        if key in parameters:
            args[key] = parameters[key]
    for key in ("bilingual", "recursive", "overwrite", "dry_run", "fast", "final_review"):
        if key in parameters:
            args[key] = bool(parameters[key])
    return args


def _has_required_args(category: str, args: dict[str, Any]) -> bool:
    if category in {"translate_file", "diagnose"}:
        return "path" in args
    if category == "edit_subtitle":
        return "path" in args and "instruction" in args
    if category == "translate_series":
        return "path" in args
    return False


def _category_to_default_tool(category: str) -> str:
    mapping = {
        "translate_file": "translate_file",
        "translate_series": "translate_series",
        "edit_subtitle": "edit_subtitle",
        "diagnose": "diagnose_path",
        "file_operation": "create_file",
        "browse": "list_files",
        "profile": "list_profiles",
    }
    return mapping.get(category, "list_files")


def apply_confidence_gate(
    decision: dict[str, Any],
    state: AgentLoopState,
) -> dict[str, Any] | None:
    """Check LLM confidence and gate mutating actions. Returns a modified decision or None to proceed.
    Only gates final_tool_call (mutating) — discovery tool_call actions pass through unchanged."""
    action = str(decision.get("action") or "").strip()
    if action not in {"final_tool_call"}:
        return None

    raw_confidence = decision.get("confidence")
    if not isinstance(raw_confidence, (int, float)):
        return None

    confidence = float(raw_confidence)
    reason = str(decision.get("reason") or "")
    num_observations = len(state.observations)

    if confidence < CONFIDENCE_LOW_THRESHOLD:
        return {"action": "respond", "message": "I need more information to proceed confidently. Could you please clarify your request?"}

    if confidence < CONFIDENCE_MEDIUM_THRESHOLD and num_observations < CONFIDENCE_MIN_OBSERVATIONS:
        message = reason or f"I think I should {action} but I am not entirely sure."
        return {"action": "ask_user", "message": f"{message}\n\nCan you confirm this is what you want?"}

    return None
