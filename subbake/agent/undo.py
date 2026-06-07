"""Undo system for SubBakeAgent.

These functions manage the undo of file operations performed by the agent.
Each file operation is recorded in the session event log, and the undo
system restores backup copies or reverses the operation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._core import SubBakeAgent


def undo_last_operation(agent: SubBakeAgent) -> None:
    """Undo the most recent file operation by restoring from its backup."""
    undo_target = _latest_file_operation_to_undo(agent)
    if undo_target is None:
        agent.console.print("[bold yellow]Nothing to undo.[/bold yellow]")
        agent._record_event("undo", "/undo", {"result": "nothing_to_undo"})
        return

    undo_targets = _file_operation_group_targets(agent, undo_target)
    undone_targets: list[dict[str, Any]] = []
    for target in undo_targets:
        if not _undo_file_operation(agent, target):
            return
        target["undone"] = True
        undone_targets.append(target)

    action = str(undo_target.get("action") or "")
    path = Path(str(undo_target.get("path") or ""))
    agent._record_event(
        "undo",
        "/undo",
        {"result": "ok", "action": action, "path": str(path), "count": len(undone_targets)},
    )


def _latest_file_operation_to_undo(agent: SubBakeAgent) -> dict[str, Any] | None:
    """Find the most recent file operation that has not yet been undone."""
    for event in reversed(agent.session.events):
        if event.get("kind") != "file_operation":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        if data.get("undone"):
            continue
        return data
    return None


def _file_operation_group_targets(
    agent: SubBakeAgent,
    undo_target: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return all file operations in the same undo group, newest first."""
    group_id = str(undo_target.get("group_id") or "")
    if not group_id:
        return [undo_target]
    targets: list[dict[str, Any]] = []
    for event in agent.session.events:
        if event.get("kind") != "file_operation":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        if data.get("undone"):
            continue
        if data.get("group_id") == group_id:
            targets.append(data)
    return list(reversed(targets))


def _undo_file_operation(agent: SubBakeAgent, undo_target: dict[str, Any]) -> bool:
    """Perform a single undo operation. Returns True on success, False on failure."""
    action = str(undo_target.get("action") or "")
    path = Path(str(undo_target.get("path") or ""))
    new_path_str = str(undo_target.get("new_path") or "")
    backup_str = str(undo_target.get("backup_path") or "")

    if action == "created":
        if path.exists():
            path.unlink()
            agent.console.print(f"[bold green]Undo created:[/bold green] deleted {path}")
        else:
            agent.console.print(
                f"[bold yellow]Created file no longer exists:[/bold yellow] {path}"
            )

    elif action in {"appended", "modified", "renamed"}:
        backup = Path(backup_str) if backup_str else None
        if backup is None or not backup.exists():
            agent.console.print(
                f"[bold yellow]Backup not found for {action}:[/bold yellow] {path}. Cannot undo."
            )
            agent._record_event(
                "undo", "/undo", {"result": "backup_missing", "path": str(path)}
            )
            return False
        if action == "renamed":
            new_path = Path(new_path_str) if new_path_str else None
            if new_path is not None and new_path.exists():
                new_path.unlink()
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, path)
            agent.console.print(f"[bold green]Undo renamed:[/bold green] restored {path}")
        else:
            shutil.copy2(backup, path)
            agent.console.print(f"[bold green]Undo {action}:[/bold green] restored {path}")

    elif action == "deleted":
        backup = Path(backup_str) if backup_str else None
        if backup is None or not backup.exists():
            agent.console.print(
                f"[bold yellow]Backup not found for deleted file:[/bold yellow] {path}. Cannot undo."
            )
            agent._record_event(
                "undo", "/undo", {"result": "backup_missing", "path": str(path)}
            )
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)
        agent.console.print(f"[bold green]Undo deleted:[/bold green] restored {path}")

    else:
        agent.console.print(
            f"[bold yellow]Unknown operation type:[/bold yellow] {action}"
        )
        agent._record_event(
            "undo", "/undo", {"result": "unknown_action", "action": action}
        )
        return False

    return True
