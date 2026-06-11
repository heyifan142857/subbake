"""Runtime cancellation helpers for interactive operations.

During a running interactive command, Esc and Ctrl+C request cancellation and
return to the prompt.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


class OperationCancelledError(RuntimeError):
    """Raised when the user cancels a running operation."""


class _CancellationState:
    """Mutable state shared by the signal handler and Esc listener."""

    __slots__ = (
        "cancelled",
        "original_handler",
        "original_terminal_attrs",
        "stop_event",
        "stdin_fd",
        "thread",
        "write_fd",
    )

    def __init__(self, *, stdin_fd: int | None, write_fd: int) -> None:
        self.cancelled: bool = False
        self.original_handler: signal._HANDLER = signal.SIG_DFL
        self.original_terminal_attrs: list[object] | None = None
        self.stop_event = threading.Event()
        self.stdin_fd = stdin_fd
        self.thread: threading.Thread | None = None
        self.write_fd = write_fd


def install_cancellation_handler(
    *,
    stderr_fileno: int | None = None,
    stdin_fileno: int | None = None,
    enable_escape: bool = True,
) -> tuple[Callable[[], bool], Callable[[], None]]:
    """Install runtime cancellation handling.

    Esc sets the returned cancellation flag when stdin is an interactive TTY.
    Ctrl+C sets the same cancellation flag while this handler is installed.
    """
    write_fd = stderr_fileno if stderr_fileno is not None else sys.__stderr__.fileno()
    read_fd = _interactive_stdin_fileno(stdin_fileno) if enable_escape else None
    state = _CancellationState(stdin_fd=read_fd, write_fd=write_fd)

    def _interrupt(signum: int, frame: object | None) -> None:
        state.cancelled = True
        _signal_safe_newline(state.write_fd)

    state.original_handler = signal.signal(signal.SIGINT, _interrupt)
    if read_fd is not None:
        _start_escape_listener(state)

    def restore() -> None:
        state.stop_event.set()
        if state.thread is not None:
            state.thread.join(timeout=0.25)
        if state.stdin_fd is not None and state.original_terminal_attrs is not None:
            try:
                termios.tcsetattr(state.stdin_fd, termios.TCSADRAIN, state.original_terminal_attrs)
            except termios.error:
                pass
        signal.signal(signal.SIGINT, state.original_handler)
        state.cancelled = False

    def cancel_requested() -> bool:
        return state.cancelled

    return cancel_requested, restore


@contextmanager
def cancellation_scope(
    *,
    stderr_fileno: int | None = None,
    stdin_fileno: int | None = None,
    enable_escape: bool = True,
) -> Iterator[Callable[[], bool]]:
    """Context manager wrapper around install_cancellation_handler."""
    cancel_requested, restore = install_cancellation_handler(
        stderr_fileno=stderr_fileno,
        stdin_fileno=stdin_fileno,
        enable_escape=enable_escape,
    )
    try:
        yield cancel_requested
    finally:
        restore()


def run_interruptibly(
    operation: Callable[[], T],
    *,
    cancel_requested: Callable[[], bool] | None,
    poll_interval: float = 0.05,
) -> T:
    """Run operation while the main thread remains responsive to Esc/Ctrl+C."""
    if cancel_requested is None:
        return operation()

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_queue.put((True, operation()))
        except BaseException as exc:
            result_queue.put((False, exc))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    while True:
        if cancel_requested():
            raise OperationCancelledError("Operation cancelled by user.")
        try:
            ok, value = result_queue.get(timeout=poll_interval)
        except queue.Empty:
            continue
        if ok:
            return value  # type: ignore[return-value]
        raise value  # type: ignore[misc]


def _interactive_stdin_fileno(stdin_fileno: int | None) -> int | None:
    try:
        fd = stdin_fileno if stdin_fileno is not None else sys.__stdin__.fileno()
    except (AttributeError, OSError):
        return None
    return fd if os.isatty(fd) else None


def _start_escape_listener(state: _CancellationState) -> None:
    if state.stdin_fd is None:
        return
    try:
        state.original_terminal_attrs = termios.tcgetattr(state.stdin_fd)
        tty.setcbreak(state.stdin_fd)
    except termios.error:
        state.original_terminal_attrs = None
        return

    def _listen_for_escape() -> None:
        while not state.stop_event.is_set():
            try:
                readable, _, _ = select.select([state.stdin_fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not readable:
                continue
            try:
                raw = os.read(state.stdin_fd, 1)
            except OSError:
                return
            if raw == b"\x1b":
                state.cancelled = True
                _signal_safe_newline(state.write_fd)
                return
            time.sleep(0)

    state.thread = threading.Thread(target=_listen_for_escape, daemon=True)
    state.thread.start()


def _signal_safe_newline(write_fd: int) -> None:
    try:
        os.write(write_fd, b"\n")
    except OSError:
        pass
