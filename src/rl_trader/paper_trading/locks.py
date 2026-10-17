"""Exclusive process lock for the paper trader (prevents duplicate cycles)."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from types import TracebackType


class ProcessLockError(RuntimeError):
    """Raised when another paper-trader process holds the lock."""


class ProcessLock:
    """PID-file lock with exclusive ``fcntl`` flock (POSIX)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: object | None = None

    def acquire(self) -> None:
        """Acquire an exclusive non-blocking lock or raise."""
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise ProcessLockError(
                f"paper trader already running (lock={self.path})"
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        atexit.register(self.release)

    def release(self) -> None:
        """Release the lock if held."""
        import fcntl

        fh = self._fh
        if fh is None:
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        finally:
            try:
                fh.close()  # type: ignore[attr-defined]
            finally:
                self._fh = None

    def __enter__(self) -> ProcessLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc, tb
        self.release()
