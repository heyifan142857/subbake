from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


PROTECTED_PATH_PARTS = {".git", ".hg", ".svn", ".venv", "venv", ".subbake", "__pycache__"}


@dataclass(slots=True)
class FileOpResult:
    action: str
    path: Path
    backup_path: Path | None = None
    new_path: Path | None = None
    detail: str = ""


class FileOperationGuard:
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def create_file(self, path: Path, content: str) -> FileOpResult:
        safe_path = self._resolve_safe_path(path)
        if safe_path.exists():
            raise ValueError(f"File already exists: {safe_path}")
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return FileOpResult(action="created", path=safe_path)

    def read_file(self, path: Path, *, limit: int = 12000) -> str:
        safe_path = self._require_text_file(path)
        text = safe_path.read_text(encoding="utf-8")
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n...[truncated]"

    def list_files(self, path: Path, *, recursive: bool = False, limit: int = 200) -> list[Path]:
        safe_path = self._resolve_safe_path(path, allow_project_root=True)
        if not safe_path.exists():
            raise FileNotFoundError(f"Path not found: {safe_path}")
        if safe_path.is_file():
            return [safe_path]
        results: list[Path] = []
        for item in self._iter_safe_children(safe_path, recursive=recursive):
            if len(results) >= limit:
                break
            results.append(item)
        return sorted(results)

    def search_files(self, path: Path, pattern: str, *, limit: int = 50) -> list[str]:
        if not pattern:
            raise ValueError("Search pattern cannot be empty.")
        safe_path = self._resolve_safe_path(path, allow_project_root=True)
        if not safe_path.exists():
            raise FileNotFoundError(f"Path not found: {safe_path}")
        files = [safe_path] if safe_path.is_file() else [
            item
            for item in self._iter_safe_children(safe_path, recursive=True)
            if item.is_file()
        ]
        matches: list[str] = []
        expression = re.compile(re.escape(pattern), re.IGNORECASE)
        for file_path in files:
            if len(matches) >= limit:
                break
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if expression.search(line):
                    relative = file_path.relative_to(self.project_root)
                    matches.append(f"{relative}:{line_number}: {line}")
                    if len(matches) >= limit:
                        break
        return matches

    def append_file(self, path: Path, content: str) -> FileOpResult:
        safe_path = self._require_text_file(path)
        backup_path = self._backup_path(safe_path)
        text = safe_path.read_text(encoding="utf-8")
        prefix = "" if not text else "\n"
        safe_path.write_text(text + prefix + content, encoding="utf-8")
        return FileOpResult(action="appended", path=safe_path, backup_path=backup_path)

    def replace_in_file(self, path: Path, old: str, new: str) -> FileOpResult:
        if not old:
            raise ValueError("Replacement source text cannot be empty.")
        safe_path = self._require_text_file(path)
        text = safe_path.read_text(encoding="utf-8")
        if old not in text:
            raise ValueError("Replacement source text was not found.")
        backup_path = self._backup_path(safe_path)
        replaced = text.replace(old, new, 1)
        safe_path.write_text(replaced, encoding="utf-8")
        return FileOpResult(action="modified", path=safe_path, backup_path=backup_path)

    def rename_path(self, old_path: Path, new_path: Path) -> FileOpResult:
        safe_old_path = self._resolve_safe_path(old_path)
        safe_new_path = self._resolve_safe_path(new_path)
        if not safe_old_path.exists():
            raise FileNotFoundError(f"Path not found: {safe_old_path}")
        if safe_new_path.exists():
            raise ValueError(f"Destination already exists: {safe_new_path}")
        safe_new_path.parent.mkdir(parents=True, exist_ok=True)
        safe_old_path.rename(safe_new_path)
        return FileOpResult(action="renamed", path=safe_old_path, new_path=safe_new_path)

    def delete_file(self, path: Path) -> FileOpResult:
        safe_path = self._resolve_safe_path(path)
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {safe_path}")
        if safe_path.is_dir():
            raise ValueError("Agent file deletion only supports files, not directories.")
        backup_path = self._backup_path(safe_path)
        safe_path.unlink()
        return FileOpResult(action="deleted", path=safe_path, backup_path=backup_path)

    def _require_text_file(self, path: Path) -> Path:
        safe_path = self._resolve_safe_path(path)
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {safe_path}")
        if not safe_path.is_file():
            raise ValueError(f"Expected a file: {safe_path}")
        try:
            safe_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Refusing to edit a non-UTF-8 text file: {safe_path}") from exc
        return safe_path

    def _resolve_safe_path(self, path: Path, *, allow_project_root: bool = False) -> Path:
        resolved = path.resolve()
        if resolved == self.project_root and not allow_project_root:
            raise ValueError("Path must point to a file or child path, not the project root.")
        if self.project_root not in (resolved, *resolved.parents):
            raise ValueError(f"Path is outside the project root: {resolved}")
        relative = resolved.relative_to(self.project_root)
        protected = PROTECTED_PATH_PARTS.intersection(relative.parts)
        if protected:
            protected_part = sorted(protected)[0]
            raise ValueError(f"Refusing to operate inside protected path: {protected_part}")
        return resolved

    def _is_protected(self, path: Path) -> bool:
        try:
            self._resolve_safe_path(path)
        except ValueError:
            return True
        return False

    def _iter_safe_children(self, path: Path, *, recursive: bool) -> list[Path]:
        if not recursive:
            return [item for item in path.iterdir() if not self._is_protected(item)]

        results: list[Path] = []
        pending = [path]
        while pending:
            current = pending.pop()
            for item in current.iterdir():
                if self._is_protected(item):
                    continue
                results.append(item)
                if item.is_dir() and not item.is_symlink():
                    pending.append(item)
        return results

    def _backup_path(self, path: Path) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        relative = path.relative_to(self.project_root)
        backup_path = self.project_root / ".subbake" / "agent" / "backups" / timestamp / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            backup_path = backup_path.with_name(f"{backup_path.stem}-{path.stat().st_mtime_ns}{backup_path.suffix}")
        shutil.copy2(path, backup_path)
        return backup_path
