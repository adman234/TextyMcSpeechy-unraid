"""Subprocess helper that does not throw away the error message.

`subprocess.run(..., capture_output=True, check=True)` raises a
CalledProcessError carrying only an exit status, so a failing ffmpeg or shell
script surfaces in the UI as "returned non-zero exit status 1" with the actual
reason discarded. Every external command here goes through run() instead.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class CommandFailed(RuntimeError):
    def __init__(self, cmd: list[str], code: int, output: str):
        self.cmd, self.code, self.output = cmd, code, output
        shown = output.strip()[-1500:] or "(no output)"
        super().__init__(f"`{' '.join(cmd[:4])}` failed (exit {code}):\n{shown}")


def run(cmd: list[str], cwd: str | Path | None = None,
        stdin: str | None = None, env: dict | None = None) -> str:
    proc = subprocess.run(
        cmd, cwd=cwd, input=stdin, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if proc.returncode != 0:
        raise CommandFailed(cmd, proc.returncode, proc.stdout or "")
    return proc.stdout or ""
