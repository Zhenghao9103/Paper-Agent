"""Bounded tar validation and opaque local storage; never extract untrusted paths."""

import io
import json
import math
import os
import re
import stat
import tarfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from .policy import PolicyViolation, SandboxPolicy
from .schemas import ArtifactInfo

_ID = re.compile(r"^[a-f0-9]{32}$")
_LOCK = threading.RLock()
_TYPES = {".png": "image/png", ".csv": "text/csv", ".json": "application/json"}


def default_artifact_root() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share")))
    return base / "paper-agent-sandbox" / "artifacts"


class ArtifactStore:
    def __init__(self, root: Path, policy: SandboxPolicy):
        if root.is_symlink():
            raise ValueError("Artifact root must not be a symlink")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.policy = policy

    @contextmanager
    def _locked(self):
        """Serialize quota/metadata mutations across independent stdio server processes."""
        with _LOCK:
            lock_path = self.root / ".store.lock"
            if lock_path.is_symlink():
                raise PolicyViolation("artifact_link")
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(lock_path, flags, 0o600), "r+b") as stream:
                if os.fstat(stream.fileno()).st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                if os.name == "nt":
                    import msvcrt

                    def acquire():
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

                    def release():
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    def acquire():
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    def release():
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

                deadline = time.monotonic() + 2
                while True:
                    try:
                        acquire()
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise PolicyViolation("artifact_store_busy") from None
                        time.sleep(0.01)
                try:
                    yield
                finally:
                    release()

    def _paths(self, artifact_id):
        if not _ID.fullmatch(artifact_id):
            raise PolicyViolation("invalid_artifact_id")
        paths = self.root / f"{artifact_id}.bin", self.root / f"{artifact_id}.json"
        if any(p.is_symlink() for p in paths):
            raise PolicyViolation("artifact_link")
        return paths

    def collect(self, data: bytes) -> list[ArtifactInfo]:
        if len(data) > self.policy.max_archive_bytes:
            raise PolicyViolation("archive_size")
        files = []
        total = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as archive:
                members = 0
                for member in archive:
                    members += 1
                    if members > self.policy.max_artifacts + 1:
                        raise PolicyViolation("artifact_count")
                    name = member.name
                    parts = PurePosixPath(name).parts
                    if (
                        name.startswith("/")
                        or "\\" in name
                        or ".." in parts
                        or PureWindowsPath(name).drive
                    ):
                        raise PolicyViolation("artifact_path")
                    if member.isdir() and name.rstrip("/") == "artifacts":
                        continue
                    if (
                        len(parts) != 2
                        or parts[0] != "artifacts"
                        or not member.isfile()
                        or member.issparse()
                        or ":" in name
                    ):
                        raise PolicyViolation("artifact_path_or_type")
                    if member.size < 0 or member.size > self.policy.max_artifact_bytes:
                        raise PolicyViolation("artifact_size")
                    total += member.size
                    if total > self.policy.max_total_artifact_bytes:
                        raise PolicyViolation("artifact_total_size")
                    filename = parts[1]
                    if len(filename.encode()) > 255 or any(f[0] == filename for f in files):
                        raise PolicyViolation("artifact_filename")
                    suffix = PurePosixPath(filename).suffix.lower()
                    if suffix not in _TYPES:
                        raise PolicyViolation("artifact_type")
                    source = archive.extractfile(member)
                    if source is None:
                        raise PolicyViolation("artifact_type")
                    content = source.read(self.policy.max_artifact_bytes + 1)
                    if len(content) != member.size:
                        raise PolicyViolation("artifact_size")
                    if suffix == ".png":
                        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                            raise PolicyViolation("artifact_type")
                    else:
                        text = content.decode("utf-8")
                        if "\x00" in text:
                            raise PolicyViolation("artifact_type")
                        if suffix == ".json":
                            json.loads(text)
                    if len(files) >= self.policy.max_artifacts:
                        raise PolicyViolation("artifact_count")
                    files.append((filename, _TYPES[suffix], content))
        except (tarfile.TarError, UnicodeError, ValueError, RecursionError) as exc:
            if isinstance(exc, PolicyViolation):
                raise
            raise PolicyViolation("invalid_archive") from exc
        result = []
        with self._locked():
            self._cleanup()
            used = sum(
                max(4096, p.stat().st_size)
                for p in self.root.iterdir()
                if _ID.fullmatch(p.stem) and p.suffix in {".bin", ".json"} and not p.is_symlink()
            )
            # Charge a minimum block for both data and sidecar, including empty files.
            incoming = sum(max(4096, len(content)) + 4096 for _, _, content in files)
            if used + incoming > self.policy.artifact_store_bytes:
                raise PolicyViolation("artifact_quota")
            try:
                for filename, mime_type, content in files:
                    info = ArtifactInfo(
                        artifact_id=uuid4().hex,
                        filename=filename,
                        mime_type=mime_type,
                        size_bytes=len(content),
                    )
                    result.append(info)
                    binary, metadata = self._paths(info.artifact_id)
                    with binary.open("xb") as stream:
                        stream.write(content)
                    metadata.write_text(
                        json.dumps(
                            {
                                "info": info.model_dump(),
                                "expires_at": time.time() + self.policy.artifact_ttl_seconds,
                            }
                        ),
                        encoding="utf-8",
                    )
            except Exception:
                for info in result:
                    self._delete(info.artifact_id)
                raise
        return result

    def read(self, artifact_id: str) -> tuple[ArtifactInfo, bytes]:
        with self._locked():
            binary, metadata = self._paths(artifact_id)
            try:
                record, info = self._read_metadata(metadata)
                if record["expires_at"] <= time.time():
                    self._delete(artifact_id)
                    raise FileNotFoundError("artifact_expired")
                content = self._read_regular(binary, self.policy.max_artifact_bytes)
                if len(content) != info.size_bytes or info.artifact_id != artifact_id:
                    raise PolicyViolation("artifact_corrupt")
                return info, content
            except FileNotFoundError:
                raise FileNotFoundError("artifact_missing_or_expired") from None

    @staticmethod
    def _read_regular(path: Path, limit: int) -> bytes:
        def validate(state):
            if stat.S_ISLNK(state.st_mode):
                raise PolicyViolation("artifact_link")
            if not stat.S_ISREG(state.st_mode):
                raise PolicyViolation("artifact_path_or_type")
            if state.st_nlink != 1:
                raise PolicyViolation("artifact_link")
            if state.st_size > limit:
                raise PolicyViolation("artifact_size")

        try:
            validate(path.lstat())
            flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                     | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(os.open(path, flags), "rb") as stream:
                validate(os.fstat(stream.fileno()))
                content = stream.read(limit + 1)
            if len(content) > limit:
                raise PolicyViolation("artifact_size")
            return content
        except FileNotFoundError:
            raise
        except OSError:
            raise PolicyViolation("artifact_unavailable") from None

    def _read_metadata(self, path: Path):
        try:
            record = json.loads(self._read_regular(path, 8192))
            expires = record["expires_at"]
            if type(expires) not in (int, float) or not math.isfinite(expires):
                raise ValueError("invalid_expiry")
            info = ArtifactInfo.model_validate(record["info"])
            return record, info
        except PolicyViolation:
            raise
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise PolicyViolation("artifact_corrupt") from None

    def delete(self, artifact_id: str) -> None:
        with self._locked():
            self._delete(artifact_id)

    def _delete(self, artifact_id: str) -> None:
        for path in self._paths(artifact_id):
            path.unlink(missing_ok=True)

    def cleanup(self, *, now: float | None = None) -> None:
        with self._locked():
            self._cleanup(now=now)

    def _cleanup(self, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for path in self.root.glob("*.json"):
            if not _ID.fullmatch(path.stem) or path.is_symlink():
                continue
            try:
                record, _ = self._read_metadata(path)
                if record["expires_at"] <= now:
                    self._delete(path.stem)
            except (ValueError, KeyError, TypeError, FileNotFoundError):
                if path.exists() and path.stat().st_mtime < now - self.policy.artifact_ttl_seconds:
                    self._delete(path.stem)
        for path in self.root.glob("*.bin"):
            if (
                _ID.fullmatch(path.stem)
                and not path.is_symlink()
                and not (self.root / f"{path.stem}.json").exists()
                and path.stat().st_mtime < now - self.policy.artifact_ttl_seconds
            ):
                self._delete(path.stem)
