"""A tiny background job runner with progress reporting.

Import and training both run for a long time, so the HTTP layer starts them and
polls. Deliberately in-process and single-worker: two concurrent imports would
contend for the same GPU, and serialising them is the correct behaviour, not a
limitation to work around.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"          # running | done | error | cancelled
    progress: float = 0.0            # 0..1
    message: str = ""
    result: Any = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event)

    def set(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message
            self.log.append(message)
            # Unbounded logs are a slow memory leak on a long training run.
            del self.log[:-500]

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "log": self.log[-200:],
        }


class JobCancelled(Exception):
    pass


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        with self._lock:
            if any(j.status == "running" and j.kind == kind for j in self._jobs.values()):
                raise RuntimeError(f"a {kind} job is already running")
            job = Job(id=uuid.uuid4().hex[:12], kind=kind)
            self._jobs[job.id] = job

        def run() -> None:
            try:
                job.result = fn(job)
                job.status = "done" if not job.cancelled else "cancelled"
                if job.status == "done":
                    job.set(1.0, "finished")
            except JobCancelled:
                job.status = "cancelled"
                job.set(message="cancelled")
            except Exception as exc:                     # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.set(message=job.error)
                job.log.append(traceback.format_exc()[-2000:])

        threading.Thread(target=run, name=f"job-{kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self, kind: str) -> Job | None:
        for job in self._jobs.values():
            if job.kind == kind and job.status == "running":
                return job
        return None

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == "running":
            job._cancel.set()
            return True
        return False


RUNNER = JobRunner()
