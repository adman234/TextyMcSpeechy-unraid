"""Project state, stored as one JSON file per project.

A project is one source recording and the clips derived from it. Plain JSON on
the appdata mount rather than a database: it is a few hundred KB, the user can
read and back it up, and there is nothing here worth a schema migration.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from . import config

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_name(name: str) -> str:
    """Names become directory names and Piper voice names, so keep them tame."""
    name = (name or "").strip()
    if not SAFE_NAME.match(name):
        raise ValueError(
            "Name must be 1-64 characters: letters, digits, underscore or "
            "hyphen, starting with a letter or digit."
        )
    return name


def project_path(name: str) -> Path:
    return config.PROJECT_DIR / validate_name(name)


def manifest_path(name: str) -> Path:
    return project_path(name) / "project.json"


def load(name: str) -> dict:
    path = manifest_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"no project named {name!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def save(project: dict) -> None:
    path = manifest_path(project["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: a crash mid-write must not cost the user their proofreading.
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     encoding="utf-8") as fh:
        json.dump(project, fh, indent=1)
        tmp = Path(fh.name)
    tmp.replace(path)


def create(name: str, **fields) -> dict:
    name = validate_name(name)
    if manifest_path(name).exists():
        raise FileExistsError(f"project {name!r} already exists")
    project = {
        "name": name,
        "voice_name": fields.get("voice_name") or name,
        "dataset_name": fields.get("dataset_name") or name,
        "description": fields.get("description", ""),
        "espeak_language": fields.get("espeak_language", "en-us"),
        "piper_prefix": fields.get("piper_prefix", "en_US"),
        "voice_type": fields.get("voice_type", "M"),
        "quality": fields.get("quality", "medium"),
        "batch_size": fields.get("batch_size", 8),
        "num_workers": fields.get("num_workers", 8),
        "from_scratch": False,
        "status": "empty",
        "source": None,
        "duration": 0.0,
        "speakers": 0,
        "warnings": [],
        "clips": [],
    }
    save(project)
    return project


def listing() -> list[dict]:
    out = []
    if not config.PROJECT_DIR.is_dir():
        return out
    for path in sorted(config.PROJECT_DIR.iterdir()):
        manifest = path / "project.json"
        if not manifest.is_file():
            continue
        try:
            p = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        clips = p.get("clips", [])
        out.append({
            "name": p["name"],
            "status": p.get("status", "empty"),
            "clips": len(clips),
            "included": sum(1 for c in clips if c.get("include")),
            "duration": p.get("duration", 0.0),
        })
    return out


def delete(name: str) -> None:
    shutil.rmtree(project_path(name), ignore_errors=True)


def stats(project: dict) -> dict:
    """Numbers the UI shows and the user actually decides on."""
    clips = project.get("clips", [])
    included = [c for c in clips if c.get("include")]
    secs = sum(c["end"] - c["start"] for c in included)
    return {
        "total": len(clips),
        "included": len(included),
        "included_seconds": round(secs, 1),
        "included_minutes": round(secs / 60, 1),
        "speakers": project.get("speakers", 0),
        "low_confidence": sum(1 for c in included if c.get("confidence", 1) < 0.65),
    }
