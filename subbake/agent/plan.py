"""Plan-mode management for SubBakeAgent.

Extracted from ``_core.py``. Handles the /plan lifecycle: entering and
leaving plan mode, storing pending plans, approving, rejecting, and
toggling with Shift+Tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from .trace import (
    _now_iso,
    _slash_command_matches,
    _unique_slash_command_match,
)
from .ui import print_tool_call_preview


def handle_plan_command(agent: SubBakeAgent, rest: str) -> None:
    """Handle the /plan command (on/off)."""
    option = rest.strip().lower()
    if option in {"off", "false", "0", "disable"}:
        agent.session.mode = "chat"
        agent.console.print("[bold green]Plan mode off.[/bold green]")
        agent._record_event("mode", "/plan off", {"mode": "chat"})
        return
    agent.session.mode = "plan"
    agent.console.print("[bold green]Plan mode on.[/bold green] Mutating tools will be proposed, not executed.")
    agent._record_event("mode", "/plan", {"mode": "plan"})


def approve_pending_plan(agent: SubBakeAgent) -> None:
    """Execute all tool calls in the pending plan."""
    plan = agent.session.pending_plan
    if not plan:
        agent.console.print("[bold yellow]No pending plan to approve.[/bold yellow]")
        return
    tool_calls = [
        call
        for call in plan.get("tool_calls") or []
        if isinstance(call, dict)
    ]
    if not tool_calls:
        agent.console.print("[bold yellow]Pending plan has no executable tool calls.[/bold yellow]")
        agent.session.pending_plan = None
        return
    for call in tool_calls:
        result = agent._run_tool_call(
            str(call.get("tool_name") or ""),
            dict(call.get("arguments") or {}),
            original=str(plan.get("original") or "/approve"),
        )
        if result:
            agent.console.print(result)
    agent.session.pending_plan = None
    agent._record_event("approve", "/approve", {"executed": len(tool_calls)})


def reject_pending_plan(agent: SubBakeAgent) -> None:
    """Discard the pending plan."""
    if agent.session.pending_plan is None:
        agent.console.print("[bold yellow]No pending plan to reject.[/bold yellow]")
        return
    agent.session.pending_plan = None
    agent.console.print("[bold green]Pending plan discarded.[/bold green]")
    agent._record_event("reject", "/reject")


def store_plan(agent: SubBakeAgent, decision: dict[str, Any], *, original: str) -> None:
    """Store a proposed plan in the session for approval."""
    tool_calls = [
        call
        for call in decision.get("tool_calls") or []
        if isinstance(call, dict)
    ]
    plan = {
        "message": str(decision.get("message") or "Proposed plan."),
        "tool_calls": tool_calls,
        "created_at": _now_iso(),
        "original": original,
    }
    agent.session.pending_plan = plan
    agent.console.print("<proposed_plan>")
    agent.console.print(plan["message"])
    if tool_calls:
        agent.console.print("─" * 40)
        for call in tool_calls:
            print_tool_call_preview(agent.console, call)
        agent.console.print("─" * 40)
    agent.console.print("</proposed_plan>")
    agent.console.print("Use /approve to execute this plan, or /reject to discard it.")
    agent._record_event("plan", original, {"tool_calls": tool_calls})


def toggle_plan_mode(agent: SubBakeAgent) -> None:
    """Toggle plan mode on/off (triggered by Shift+Tab)."""
    if agent.session.mode == "plan":
        agent.session.mode = "chat"
    else:
        agent.session.mode = "plan"
    agent._record_event("mode", "<shift-tab>", {"mode": agent.session.mode})
    agent.store.save(agent.session)


def build_plan_toggle_key_bindings(agent: SubBakeAgent):
    """Build prompt_toolkit key bindings for Shift+Tab toggle and tab-completion."""
    try:
        from prompt_toolkit.filters import has_completions
        from prompt_toolkit.key_binding import KeyBindings
    except Exception:
        return None

    key_bindings = KeyBindings()

    @key_bindings.add("s-tab")
    def _toggle(event) -> None:
        toggle_plan_mode(agent)
        event.app.exit(result="")

    def _do_complete(event) -> None:
        buffer = event.current_buffer
        text = buffer.document.text_before_cursor
        if text.startswith("/"):
            match = _unique_slash_command_match(text)
            if match is not None:
                buffer.delete_before_cursor(len(text))
                buffer.insert_text(match)
                return
            matches = _slash_command_matches(text)
            if len(matches) > 1 and buffer.complete_state is None:
                buffer.start_completion(select_first=True)
                return
        if buffer.complete_state is None:
            buffer.start_completion(select_first=True)
            return
        completion = buffer.complete_state.current_completion
        if completion is not None:
            buffer.apply_completion(completion)

    key_bindings.add("tab")(_do_complete)

    @key_bindings.add("down", filter=has_completions)
    def _next_completion(event) -> None:
        event.current_buffer.complete_next()

    @key_bindings.add("up", filter=has_completions)
    def _previous_completion(event) -> None:
        event.current_buffer.complete_previous()

    @key_bindings.add("enter", filter=has_completions)
    def _accept_completion(event) -> None:
        buffer = event.current_buffer
        complete_state = buffer.complete_state
        completion = complete_state.current_completion if complete_state is not None else None
        if completion is not None:
            buffer.apply_completion(completion)
        event.app.exit(result=buffer.text)

    return key_bindings
