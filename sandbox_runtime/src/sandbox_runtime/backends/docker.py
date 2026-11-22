"""Docker authority boundary. No generated code runs on the host."""

import base64
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from ..policy import PolicyViolation, SandboxPolicy
from ..schemas import ExecutePythonRequest
from .base import BackendFailure, BackendOutput, ExecutionCancelled
from .ownership import RuntimeLease, owner_for_store

logger = logging.getLogger(__name__)
LABEL = "paper-agent.sandbox-runtime"
KEEPER_GRACE_SECONDS = 5.0
_PROBE = (
    "import json,os,pathlib; "
    "s=dict(line.split(':',1) for line in open('/proc/self/status') if ':' in line); "
    "p=pathlib.Path('/sys/fs/cgroup/memory.events'); "
    "e=dict(line.split() for line in p.read_text().splitlines()) if p.exists() else {}; "
    "print(json.dumps({'uid':os.getuid(),'oom_kill':int(e['oom_kill']) if 'oom_kill' in e "
    "else None,**{k:s.get(k,'').strip() for k in "
    "('NoNewPrivs','Seccomp','CapEff','CapPrm','CapBnd','CapAmb')}}))"
)


def _tmpfs(policy):
    return {
        "/workspace": (
            f"rw,nosuid,nodev,size={policy.workspace_bytes},uid=10001,gid=10001,mode=700"
        ),
        "/tmp": f"rw,nosuid,nodev,noexec,size={policy.tmp_bytes},mode=1777",
    }


def _verify_config(container, policy):
    """Validate daemon inspect, rather than trusting create arguments were applied."""
    attrs = container.attrs
    config, host = attrs.get("Config", {}), attrs.get("HostConfig", {})
    options = set(host.get("SecurityOpt") or [])
    nnp = options & {"no-new-privileges", "no-new-privileges:true", "no-new-privileges=true"}
    if (
        config.get("User") != "10001:10001"
        or config.get("WorkingDir") != "/workspace"
        or config.get("NetworkDisabled") is not True
        or config.get("Entrypoint")
        or host.get("Privileged") is not False
        or host.get("ReadonlyRootfs") is not True
        or "ALL" not in {str(cap).upper() for cap in host.get("CapDrop") or []}
        or host.get("CapAdd")
        or not nnp
        or any(option.startswith("seccomp=") for option in options)
        or host.get("NetworkMode") != "none"
        or host.get("Memory") != policy.memory_bytes
        or host.get("MemorySwap") != policy.memory_bytes
        or host.get("NanoCpus") != int(policy.cpus * 1_000_000_000)
        or host.get("PidsLimit") != policy.pids
        or any(host.get(key) for key in (
            "Binds", "Mounts", "VolumesFrom", "Devices", "DeviceRequests", "PortBindings"
        ))
        or any(
            host.get(key, "") not in ("", "private")
            for key in ("PidMode", "IpcMode", "UTSMode", "UsernsMode")
        )
        or host.get("Tmpfs") != _tmpfs(policy)
        or any(
            mount.get("Type") != "tmpfs" or mount.get("Destination") not in _tmpfs(policy)
            for mount in attrs.get("Mounts", [])
        )
    ):
        raise BackendFailure("isolation_unavailable")


def _verify_probe(container):
    result = container.exec_run(
        ["python", "-I", "-B", "-c", _PROBE], user="10001:10001", workdir="/workspace",
        stdout=True, stderr=True, demux=True,
    )
    try:
        stdout, stderr = result.output
        if result.exit_code != 0 or stderr or not stdout or len(stdout) > 4096:
            raise ValueError()
        state = json.loads(stdout)
        valid = (
            state["uid"] == 10001 and state["NoNewPrivs"] == "1" and state["Seccomp"] == "2"
            and all(int(state[name], 16) == 0 for name in ("CapEff", "CapPrm", "CapBnd", "CapAmb"))
        )
    except (ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise BackendFailure("isolation_unavailable")
    oom_kill = state.get("oom_kill")
    if oom_kill is not None and (type(oom_kill) is not int or oom_kill < 0):
        raise BackendFailure("isolation_unavailable")
    return oom_kill


def _request_arguments(request: ExecutePythonRequest) -> list[str]:
    data = json.dumps(request.model_dump(), ensure_ascii=False, allow_nan=False).encode()
    encoded = base64.b64encode(data).decode("ascii")
    # Linux limits each argv element independently. Inputs remain data for a fixed runner.
    return [encoded[index:index + 32768] for index in range(0, len(encoded), 32768)]


class _OutputReader:
    def __init__(self, stream, limit, *, demux=True):
        self.stream, self.limit = stream, limit
        self.demux = demux
        self.buffers = [bytearray(), bytearray()]
        self.truncated = [False, False]
        self.failed = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.read, daemon=True)

    def read(self):
        try:
            for chunk in self.stream:
                pair = chunk if self.demux else (chunk, None)
                for index, chunk in enumerate(pair):
                    if chunk:
                        remaining = self.limit - len(self.buffers[index])
                        self.buffers[index].extend(chunk[:remaining])
                        self.truncated[index] |= len(chunk) > remaining
                if any(self.truncated):
                    break
        except Exception:
            self.failed = True
        finally:
            self.done.set()

    def text(self, index):
        return self.buffers[index].decode("utf-8", errors="replace")


class DockerSandboxBackend:
    def __init__(self, client=None):
        self.client = client
        self._active = {}
        self._active_events = {}
        self._active_lock = threading.RLock()
        self._readers = []
        self._run_lock = threading.Lock()
        self.instance_id = uuid4().hex
        self.owner_id = None
        self._lease = None

    def configure_owner(self, artifact_root: Path) -> None:
        owner = owner_for_store(Path(artifact_root))
        if self.owner_id == owner:
            return
        if self._active or (self._lease is not None and self._lease.held):
            raise BackendFailure("runtime_busy")
        self.owner_id = owner
        self._lease = RuntimeLease(Path(artifact_root))

    def _ensure_owner(self):
        if self._lease is None:
            from ..artifacts import default_artifact_root

            self.configure_owner(default_artifact_root())

    def _remove(self, execution_id, container):
        from docker.errors import APIError, NotFound
        from requests.exceptions import Timeout as RequestTimeout

        # Shutdown/cancellation cleanup has a separate, short bounded grace period.
        # Cancellation and the worker's finally may both arrive here for the same ID.
        with self._active_lock:
            container.client.api.timeout = 0.5
            for attempt in range(2):
                try:
                    container.remove(force=True)
                    try:
                        container.client.api.inspect_container(container.id)
                    except NotFound:
                        break
                    if attempt:
                        raise BackendFailure("cleanup_failed")
                except NotFound:
                    break
                except RequestTimeout:
                    # The daemon may have removed it after the HTTP response timed out.
                    if attempt:
                        raise
                except APIError as exc:
                    if exc.status_code != 409:
                        raise
                    # An independent daemon-side removal can still be in progress.
                    try:
                        container.client.api.inspect_container(container.id)
                    except NotFound:
                        break
                    if attempt:
                        raise
                    time.sleep(0.02)
            self._active.pop(execution_id, None)
            self._active_events.pop(execution_id, None)

    def cancel(self, cancelled):
        with self._active_lock:
            for execution_id, container in list(self._active.items()):
                if self._active_events.get(execution_id) is cancelled:
                    try:
                        self._remove(execution_id, container)
                    except Exception:
                        logger.error("cancel_cleanup_failed execution_id=%s", execution_id)

    def close(self) -> None:
        with self._active_lock:
            for execution_id, container in list(self._active.items()):
                try:
                    self._remove(execution_id, container)
                    if self.client is None:
                        container.client.close()
                except Exception:
                    logger.error("shutdown_cleanup_failed execution_id=%s", execution_id)
            if not self._active and not self._run_lock.locked() and self._lease is not None:
                self._lease.release()

    def _recover(self, client, check_deadline):
        report = {"removed": 0, "failed": 0, "skipped": 0, "busy": False}
        check_deadline()
        for container in client.containers.list(
            all=True, filters={"label": [f"{LABEL}=1", f"owner_id={self.owner_id}"]}, sparse=True,
        ):
            check_deadline()
            # Docker SDK's .labels property rejects sparse list objects; use list Labels.
            labels = (
                container.attrs["Labels"] if "Labels" in container.attrs else container.labels
            ) or {}
            execution_id = labels.get("execution_id", "")
            if (
                labels.get(LABEL) != "1" or labels.get("owner_id") != self.owner_id
                or not re.fullmatch(r"[a-f0-9]{32}", labels.get("instance_id", ""))
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", execution_id)
                or execution_id in self._active
                or any(container.id == active.id for active in self._active.values())
            ):
                report["skipped"] += 1
                continue
            try:
                self._remove(execution_id, container)
                report["removed"] += 1
                logger.info("recovered_container execution_id=%s", execution_id)
            except Exception:
                report["failed"] += 1
                logger.error("stale_cleanup_failed execution_id=%s", execution_id)
        logger.info(
            "recovery_result removed=%s failed=%s skipped=%s",
            report["removed"], report["failed"], report["skipped"],
        )
        return report

    def reap_stale(self, policy: SandboxPolicy):
        import docker

        self._ensure_owner()
        busy = {"removed": 0, "failed": 0, "skipped": 0, "busy": True}
        if not self._run_lock.acquire(blocking=False):
            return busy
        client = None
        try:
            if self._active or not self._lease.acquire():
                return busy
            client = self.client or docker.from_env(timeout=2)
            deadline = time.monotonic() + 5

            def check_deadline():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendFailure("recovery_timeout")
                client.api.timeout = min(2.0, remaining)

            return self._recover(client, check_deadline)
        finally:
            if client is not None and self.client is None:
                client.close()
            if not self._active:
                self._lease.release()
            self._run_lock.release()

    def run(
        self,
        request: ExecutePythonRequest,
        *,
        execution_id: str,
        policy: SandboxPolicy,
        deadline: float,
        cancelled: threading.Event | None = None,
    ) -> BackendOutput:
        import docker
        from docker.types import LogConfig
        from requests.exceptions import Timeout as RequestTimeout

        self._ensure_owner()
        if not self._run_lock.acquire(blocking=False):
            raise BackendFailure("runtime_busy")
        owned_client = self.client is None
        client = self.client
        container = None
        reader = None
        archive_reader = None
        output = None

        def check_deadline():
            if cancelled is not None and cancelled.is_set():
                raise ExecutionCancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline_exceeded")
            if client is not None:
                client.api.timeout = max(0.01, min(2.0, deadline - time.monotonic()))

        try:
            check_deadline()
            if not self._lease.acquire():
                raise BackendFailure("runtime_busy")
            self._readers = [r for r in self._readers if r.thread.is_alive()]
            if self._active or self._readers:
                raise BackendFailure("cleanup_pending")
            client = self.client or docker.from_env(timeout=min(2.0, policy.timeout_seconds))
            check_deadline()
            if self._recover(client, check_deadline)["failed"]:
                raise BackendFailure("cleanup_pending")
            check_deadline()
            info = client.info()
            if info.get("OSType") != "linux" or not any(
                "name=seccomp" in option.split(",") and "profile=builtin" in option.split(",")
                for option in info.get("SecurityOptions", [])
            ):
                raise BackendFailure("isolation_unavailable")
            check_deadline()
            image = client.images.get(policy.image)
            check_deadline()
            container = client.containers.create(
                image.id,
                command=[
                    "python", "-I", "-B", "/opt/sandbox/keeper.py",
                    str(policy.timeout_seconds + KEEPER_GRACE_SECONDS),
                ],
                user="10001:10001",
                working_dir="/workspace",
                network_disabled=True,
                network_mode="none",
                read_only=True,
                privileged=False,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit=policy.memory_bytes,
                memswap_limit=policy.memory_bytes,
                nano_cpus=int(policy.cpus * 1_000_000_000),
                pids_limit=policy.pids,
                tmpfs=_tmpfs(policy),
                environment={
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": "/tmp/matplotlib",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
                tty=False,
                labels={
                    LABEL: "1", "execution_id": execution_id, "created_at": str(time.time()),
                    "owner_id": self.owner_id, "instance_id": self.instance_id,
                },
                log_config=LogConfig(
                    type="local", config={"max-size": "1m", "max-file": "1", "compress": "false"}
                ),
            )
            with self._active_lock:
                self._active[execution_id] = container
                self._active_events[execution_id] = cancelled
            check_deadline()
            container.start()
            if time.monotonic() >= deadline:
                output = BackendOutput(exit_code=None, timed_out=True)
                return output
            check_deadline()
            container.reload()
            _verify_config(container, policy)
            check_deadline()
            oom_before = _verify_probe(container)
            check_deadline()
            execution = client.api.exec_create(
                container.id,
                cmd=["python", "-I", "-B", "/opt/sandbox/runner.py", *_request_arguments(request)],
                user="10001:10001",
                workdir="/workspace",
                stdout=True,
                stderr=True,
            )
            check_deadline()
            stream = client.api.exec_start(execution["Id"], stream=True, demux=True)
            reader = _OutputReader(stream, policy.max_log_bytes)
            self._readers.append(reader)
            reader.thread.start()
            timed_out = False
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise ExecutionCancelled()
                if time.monotonic() >= deadline:
                    timed_out = True
                    client.api.timeout = 0.5
                    try:
                        container.kill()
                    except Exception:
                        logger.warning("timeout_kill_failed execution_id=%s", execution_id)
                    break
                check_deadline()
                state = client.api.exec_inspect(execution["Id"])
                if not state["Running"]:
                    break
                if any(reader.truncated):
                    client.api.timeout = 0.5
                    container.kill()
                    break
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            reader.done.wait(max(0, min(0.05, deadline - time.monotonic())))
            if not timed_out and not any(reader.truncated) and reader.failed:
                raise BackendFailure("output_collection_failed")
            if not timed_out:
                check_deadline()
                container.reload()
            oom_killed = bool(container.attrs.get("State", {}).get("OOMKilled", False))
            if (
                not timed_out and not any(reader.truncated) and state.get("ExitCode") == 137
                and container.attrs.get("State", {}).get("Running") and oom_before is not None
            ):
                check_deadline()
                try:
                    oom_after = _verify_probe(container)
                    oom_killed |= oom_after is not None and oom_after > oom_before
                except Exception:
                    # Resource exhaustion can prevent runc from spawning a diagnostic exec.
                    # Initial isolation was already verified; preserve the observed code exit.
                    logger.warning("oom_diagnostic_unavailable execution_id=%s", execution_id)
            archive_data = None
            if not timed_out and not any(reader.truncated) and state.get("ExitCode") == 0:
                check_deadline()
                collection = client.api.exec_create(
                    container.id,
                    cmd=["python", "-I", "-B", "/opt/sandbox/collect.py"],
                    user="10001:10001", workdir="/workspace", stdout=True, stderr=True,
                )
                check_deadline()
                chunks = client.api.exec_start(collection["Id"], stream=True, demux=True)
                archive_reader = _OutputReader(chunks, policy.max_archive_bytes)
                self._readers.append(archive_reader)
                archive_reader.thread.start()
                while not archive_reader.done.wait(min(0.02, max(0, deadline - time.monotonic()))):
                    check_deadline()
                check_deadline()
                if archive_reader.truncated[0]:
                    raise PolicyViolation("archive_size")
                collection_code = client.api.exec_inspect(collection["Id"]).get("ExitCode")
                if (
                    collection_code == 2
                    and archive_reader.text(1).strip() == "artifact_path_or_type"
                ):
                    raise PolicyViolation("artifact_path_or_type")
                if archive_reader.failed or archive_reader.buffers[1] or collection_code != 0:
                    raise BackendFailure("archive_collection_failed")
                archive_data = bytes(archive_reader.buffers[0])
            if not timed_out and not any(reader.truncated) and not reader.done.is_set():
                # A child can inherit stdio after the main exec has exited. Collect tmpfs
                # artifacts first, then stop the entire container to close those writers.
                check_deadline()
                container.kill()
                reader.done.wait(max(0, min(0.5, deadline - time.monotonic())))
                if not reader.done.is_set():
                    raise BackendFailure("output_collection_failed")
            output = BackendOutput(
                exit_code=None if timed_out else state.get("ExitCode"),
                stdout=reader.text(0),
                stderr=reader.text(1),
                timed_out=timed_out,
                oom_killed=oom_killed,
                stdout_truncated=reader.truncated[0],
                stderr_truncated=reader.truncated[1],
                artifact_archive=archive_data,
            )
            return output
        except docker.errors.ImageNotFound:
            raise BackendFailure("image_unavailable") from None
        except RequestTimeout:
            if cancelled is not None and cancelled.is_set():
                raise ExecutionCancelled() from None
            if time.monotonic() >= deadline:
                raise TimeoutError("deadline_exceeded") from None
            raise BackendFailure("docker_timeout") from None
        except docker.errors.APIError:
            if cancelled is not None and cancelled.is_set():
                raise ExecutionCancelled() from None
            raise BackendFailure("docker_api_error") from None
        finally:
            original_error = sys.exc_info()[1]
            cleanup_started = time.monotonic()
            cleanup_status, cleanup_category = "not_required", None
            try:
                if container is not None:
                    try:
                        self._remove(execution_id, container)
                        cleanup_status = "succeeded"
                    except Exception as exc:
                        cleanup_status, cleanup_category = "failed", "cleanup_failed"
                        logger.error(
                            "cleanup_failed execution_id=%s error_type=%s",
                            execution_id, type(exc).__name__,
                        )
            finally:
                for current_reader in (reader, archive_reader):
                    if current_reader is None:
                        continue
                    close = getattr(current_reader.stream, "close", None)
                    if close:
                        try:
                            close()
                        except Exception:
                            logger.warning("stream_close_failed execution_id=%s", execution_id)
                    if current_reader.thread.ident is not None:
                        current_reader.thread.join(timeout=0.05)
                if owned_client and client is not None and execution_id not in self._active:
                    try:
                        client.close()
                    except Exception:
                        logger.warning("client_close_failed execution_id=%s", execution_id)
                if not self._active:
                    try:
                        self._lease.release()
                    except Exception:
                        logger.error("lease_release_failed execution_id=%s", execution_id)
                self._run_lock.release()
                target = output if output is not None else original_error
                if target is not None:
                    target.cleanup_status = cleanup_status
                    target.cleanup_error_category = cleanup_category
                    target.cleanup_duration_ms = (time.monotonic() - cleanup_started) * 1000
