"""Tests for runtime cancellation helpers."""

from __future__ import annotations

import os
import signal
import time
import unittest

from subbake.agent.cancellation import (
    OperationCancelledError,
    install_cancellation_handler,
    run_interruptibly,
)


class CancellationTestCase(unittest.TestCase):
    def setUp(self):
        self._original_handler = signal.getsignal(signal.SIGINT)
        self._null_fd = os.open(os.devnull, os.O_WRONLY)

    def tearDown(self):
        signal.signal(signal.SIGINT, self._original_handler)
        os.close(self._null_fd)

    def test_cancel_requested_starts_false(self):
        cancel_requested, restore = install_cancellation_handler(
            stderr_fileno=self._null_fd,
            enable_escape=False,
        )
        try:
            self.assertFalse(cancel_requested())
        finally:
            restore()

    def test_ctrl_c_sets_cancel_requested_while_handler_installed(self):
        cancel_requested, restore = install_cancellation_handler(
            stderr_fileno=self._null_fd,
            enable_escape=False,
        )
        try:
            os.kill(os.getpid(), signal.SIGINT)
            self.assertTrue(cancel_requested())
        finally:
            restore()

    def test_restore_restores_original_handler(self):
        cancel_requested, restore = install_cancellation_handler(
            stderr_fileno=self._null_fd,
            enable_escape=False,
        )
        self.assertFalse(cancel_requested())
        restore()
        with self.assertRaises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)

    def test_escape_sets_cancel_requested(self):
        master_fd, slave_fd = os.openpty()
        try:
            cancel_requested, restore = install_cancellation_handler(
                stderr_fileno=self._null_fd,
                stdin_fileno=slave_fd,
            )
            try:
                os.write(master_fd, b"\x1b")
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not cancel_requested():
                    time.sleep(0.01)
                self.assertTrue(cancel_requested())
            finally:
                restore()
        finally:
            os.close(master_fd)
            os.close(slave_fd)

    def test_run_interruptibly_returns_result(self):
        result = run_interruptibly(lambda: "done", cancel_requested=lambda: False)

        self.assertEqual(result, "done")

    def test_run_interruptibly_propagates_operation_exception(self):
        def fail():
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            run_interruptibly(fail, cancel_requested=lambda: False)

    def test_run_interruptibly_cancels_without_waiting_for_operation(self):
        started_at = time.monotonic()

        with self.assertRaises(OperationCancelledError):
            run_interruptibly(
                lambda: time.sleep(1.0),
                cancel_requested=lambda: True,
                poll_interval=0.01,
            )

        self.assertLess(time.monotonic() - started_at, 0.5)


if __name__ == "__main__":
    unittest.main()
