"""Compatibility re-export for agent cancellation helpers."""

from __future__ import annotations

from subbake.cancellation import (
    OperationCancelledError,
    cancellation_scope,
    install_cancellation_handler,
    run_interruptibly,
)

__all__ = [
    "OperationCancelledError",
    "cancellation_scope",
    "install_cancellation_handler",
    "run_interruptibly",
]
