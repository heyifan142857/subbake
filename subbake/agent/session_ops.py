"""Session management for SubBakeAgent.

Extracted from ``_core.py``. Handles listing, switching, resuming,
and activating agent sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._core import SubBakeAgent

from .profile import values_for_profile
from .session import AgentSession
from .trace import _short_title
from .ui import select_from_list


def print_sessions(agent: SubBakeAgent) -> None:
    """Print the list of recent sessions."""
    sessions = agent.store.list_sessions()[-10:]
    if not sessions:
        agent.console.print("No agent sessions found.")
        return
    agent.console.print("[bold green]Recent sessions:[/bold green]")
    for path in sessions:
        try:
            session = agent.store.load(path)
        except Exception:
            continue
        marker = "*" if session.id == agent.session.id else " "
        title = session_title(agent, session)
        agent.console.print(
            f" {marker} {session.id}  profile={session.profile or 'default'}  {title}"
        )


def handle_session_command(agent: SubBakeAgent, rest: str) -> None:
    """Handle the /session command."""
    key = rest.strip()
    if key:
        switch_session_by_key(agent, key)
        return
    options = session_options(agent, limit=30)
    if not options:
        agent.console.print("No agent sessions found.")
        return
    if agent.interactive:
        selected = select_from_list(agent.console, agent.interactive, "Sessions", options, default=agent.session.id)
        if selected:
            switch_session_by_key(agent, selected)
        return
    print_sessions(agent)


def switch_session_by_key(agent: SubBakeAgent, key: str) -> None:
    """Switch to a session identified by key (id or path)."""
    path = find_session_path(agent, key)
    if path is None:
        raise ValueError(f"Agent session '{key}' was not found.")
    session = agent.store.load(path)
    activate_session(agent, session)
    agent.console.print(f"[bold green]Session switched:[/bold green] {session.id}  {session_title(agent, session)}")


def find_session_path(agent: SubBakeAgent, key: str) -> Path | None:
    """Find the file path for a session given a key."""
    for path in agent.store.list_sessions():
        if key in {path.stem, path.name, str(path)}:
            return path
    candidate = Path(key).expanduser()
    if not candidate.is_absolute():
        candidate = agent.cwd / candidate
    if candidate.exists():
        return candidate
    return None


def activate_session(agent: SubBakeAgent, session: AgentSession) -> None:
    """Activate a session, restoring its config and profile."""
    agent.session = session
    if session.config_path:
        config_path = Path(session.config_path).expanduser()
        if config_path.exists():
            agent.config_path = config_path
            agent.config = agent._load_config(config_path)
    if session.profile is not None:
        if agent.config is not None and session.profile in agent.config.profiles:
            agent.profile = session.profile
            agent.values = values_for_profile(agent, session.profile)
        else:
            agent.console.print(
                f"[bold yellow]Session profile unavailable:[/bold yellow] {session.profile}. "
                f"Keeping {agent.profile or 'default'}."
            )
            agent.session.profile = agent.profile
    elif agent.config is None or not agent.config.profiles:
        agent.profile = None
        agent.values = values_for_profile(agent, None)


def session_options(agent: SubBakeAgent, *, limit: int) -> list[tuple[str, str]]:
    """Build display options for the session picker."""
    options: list[tuple[str, str]] = []
    for path in reversed(agent.store.list_sessions()[-limit:]):
        try:
            session = agent.store.load(path)
        except Exception:
            continue
        marker = "* " if session.id == agent.session.id else ""
        updated = session.updated_at.replace("T", " ")[:19]
        label = f"{marker}{session_title(agent, session)}  ({session.profile or 'default'}, {updated})"
        options.append((session.id, label))
    return options


def session_title(agent: SubBakeAgent, session: AgentSession) -> str:
    """Derive a short display title for a session."""
    for event in session.events:
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("summary"), str) and data["summary"].strip():
            return _short_title(data["summary"])
    for event in session.events:
        if event.get("kind") == "user" and isinstance(event.get("input"), str) and event["input"].strip():
            return _short_title(event["input"])
    return "New session"


def resume_latest_session(agent: SubBakeAgent) -> None:
    """Resume a session — show interactive picker of recent sessions."""
    options = session_options(agent, limit=10)
    if not options:
        agent.console.print("[bold yellow]No previous agent session found.[/bold yellow]")
        return
    if agent.interactive:
        selected = select_from_list(agent.console, agent.interactive, "Resume session", options, default=agent.session.id)
        if selected is None:
            return
        switch_session_by_key(agent, selected)
        agent._record_event("resume", "/resume", {"session_id": selected})
        _show_session_replay(agent)
        return
    # Non-interactive fallback: resume the latest session
    latest = agent.store.latest()
    if latest is None:
        agent.console.print("[bold yellow]No previous agent session found.[/bold yellow]")
        return
    activate_session(agent, latest)
    agent.console.print(f"[bold green]Resumed session:[/bold green] {latest.id}  {session_title(agent, latest)}")
    agent._record_event("resume", "/resume", {"session_id": latest.id})
    _show_session_replay(agent)


MAX_REPLAY_EVENTS = 40


def _show_session_replay(agent: SubBakeAgent) -> None:
    """Display the full conversation history from the session."""
    events = [
        e for e in agent.session.events
        if e.get("kind") in ("user", "assistant")
    ]
    if not events:
        return
    total = len(events)
    # Show last MAX_REPLAY_EVENTS, note count of skipped ones
    show = events[-MAX_REPLAY_EVENTS:]
    skipped = total - len(show)
    agent.console.print("[bold]─── Conversation history ───[/bold]")
    if skipped > 0:
        agent.console.print(
            f"  [dim]… and {skipped} earlier message(s) (use /history to see all)[/dim]"
        )
    for event in show:
        kind = event.get("kind", "")
        text = (event.get("input", "") or "").strip()
        if not text:
            continue
        _print_event_line(agent, kind, text)
    agent.console.print("[bold]─── End ───[/bold]")


def _print_event_line(agent: SubBakeAgent, kind: str, text: str) -> None:
    """Print a single event line with appropriate styling."""
    preview = text if len(text) <= 120 else f"{text[:117]}..."
    if kind == "user":
        agent.console.print(f"  [bold]You:[/bold] {preview}")
    elif kind == "assistant":
        agent.console.print(f"  [dim]Assistant:[/dim] {preview}")


def print_full_history(agent: SubBakeAgent, *, limit: int | None = None) -> None:
    """Print all user/assistant events with index numbers.

    Args:
        agent: The agent instance.
        limit: If set, show only the last N exchanges (pairs). Each exchange
               is one user input followed by one assistant response.
    """
    events = [
        e for e in agent.session.events
        if e.get("kind") in ("user", "assistant")
    ]
    if not events:
        agent.console.print("[bold yellow]No conversation history found.[/bold yellow]")
        return

    if limit is not None and limit > 0:
        # Show last N exchanges (each exchange is a user+assistant pair = 2 events)
        pair_count = limit * 2
        events = events[-pair_count:]

    agent.console.print("[bold]─── Full conversation history ───[/bold]")
    for idx, event in enumerate(events, start=1):
        kind = event.get("kind", "")
        text = (event.get("input", "") or "").strip()
        if not text:
            continue
        if kind == "user":
            agent.console.print(f"  [bold]{idx}. You:[/bold] {text}")
        elif kind == "assistant":
            agent.console.print(f"     [dim]Assistant:[/dim] {text}")
    agent.console.print("[bold]─── End ───[/bold]")
    total_pairs = len(events) // 2
    agent.console.print(f"[dim]{total_pairs} exchange(s) shown ({len(events)} message(s))[/dim]")


def load_or_create_session(agent: SubBakeAgent, *, resume: bool, session_id: str | None = None) -> AgentSession:
    """Load the latest session (if resume=True) or a specific session (if session_id given), else create a new one."""
    if session_id is not None:
        found = agent.store.find_by_id(session_id)
        if found is not None:
            agent.console.print(f"[bold green]Resumed session:[/bold green] {found.id}  {session_title(agent, found)}")
            return found
        agent.console.print(f"[bold yellow]Session '{session_id}' not found. Starting a new one.[/bold yellow]")
    elif resume:
        latest = agent.store.latest()
        if latest is not None:
            agent.console.print(f"[bold green]Resumed session:[/bold green] {latest.id}")
            return latest
        agent.console.print("[bold yellow]No previous agent session found. Starting a new one.[/bold yellow]")
    return agent.store.create(
        cwd=agent.cwd,
        profile=agent.profile,
        config_path=agent.config_path,
    )
