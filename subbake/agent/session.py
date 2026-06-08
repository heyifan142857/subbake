"""Agent session model and persistence.

Extracted from agent.py. Pure data classes — zero agent coupling.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .trace import _now_iso, _verify_write_text

SESSION_VERSION = 1


@dataclass(slots=True)
class AgentSession:
    """A single agent conversation session."""

    id: str
    created_at: str
    updated_at: str
    cwd: str
    profile: str | None = None
    config_path: str | None = None
    mode: str = "chat"
    pending_plan: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SESSION_VERSION,
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cwd": self.cwd,
            "profile": self.profile,
            "config_path": self.config_path,
            "mode": self.mode,
            "pending_plan": self.pending_plan,
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSession":
        return cls(
            id=str(data["id"]),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
            cwd=str(data.get("cwd") or Path.cwd()),
            profile=data.get("profile") if isinstance(data.get("profile"), str) else None,
            config_path=data.get("config_path") if isinstance(data.get("config_path"), str) else None,
            mode=str(data.get("mode") or "chat"),
            pending_plan=data.get("pending_plan") if isinstance(data.get("pending_plan"), dict) else None,
            events=list(data.get("events") or []),
        )


class AgentSessionStore:
    """File-system persistence for agent sessions."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root / ".subbake" / "agent" / "sessions"

    def create(self, *, cwd: Path, profile: str | None, config_path: Path | None) -> AgentSession:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        return AgentSession(
            id=session_id,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            cwd=str(cwd),
            profile=profile,
            config_path=str(config_path) if config_path is not None else None,
            mode="chat",
        )

    def save(self, session: AgentSession) -> Path:
        session.updated_at = _now_iso()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session.id)
        serialized = json.dumps(session.to_dict(), ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding="utf-8")
        _verify_write_text(path, serialized)
        return path

    def latest(self) -> AgentSession | None:
        sessions = self.list_sessions()
        if not sessions:
            return None
        return self.load(sessions[-1])

    def find_by_id(self, session_id: str) -> AgentSession | None:
        """Find a session by its full ID."""
        path = self.path_for(session_id)
        if path.exists():
            return self.load(path)
        return None

    def list_sessions(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("*.json"))

    def load(self, path: Path) -> AgentSession:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != SESSION_VERSION:
            raise ValueError(f"Unsupported agent session version in {path}.")
        return AgentSession.from_dict(data)

    def path_for(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"
