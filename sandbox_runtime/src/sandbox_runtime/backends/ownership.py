"""Single-host, store-scoped lease. The OS releases locks after process death."""

import hashlib
import os
import threading
from pathlib import Path


def owner_for_store(root: Path) -> str:
    return hashlib.sha256(os.path.normcase(str(root.resolve())).encode("utf-8")).hexdigest()


class RuntimeLease:
    def __init__(self, root: Path):
        self.path = root.resolve() / ".runtime.lock"
        self._file = None
        self._lock = threading.Lock()

    @property
    def held(self) -> bool:
        return self._file is not None

    def acquire(self) -> bool:
        with self._lock:
            if self.held:
                return True
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            try:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
            self._file = handle
            return True

    def release(self) -> None:
        with self._lock:
            if self._file is None:
                return
            handle, self._file = self._file, None
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
