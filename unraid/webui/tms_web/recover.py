"""Rebuild project clips from an exported dataset.

Exporting writes the included clips and their corrected transcripts to
DATASETS/<name>/, outside the project. That copy is the reason a lost project
is recoverable at all: it holds the expensive part -- the text a human fixed by
hand -- even when the project manifest that produced it is gone.

    python3 -m tms_web.recover <dataset> <project>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import config, projects
from .proc import run as sh


def _duration(path: Path) -> float:
    try:
        return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]).strip())
    except Exception:                                     # noqa: BLE001
        return 0.0


def recover(dataset: str, project_name: str) -> dict:
    src = config.DATASETS_DIR / dataset
    meta = src / "metadata.csv"
    wavs = src / "wav_22050"
    if not meta.is_file() or not wavs.is_dir():
        raise SystemExit(
            f"{src} does not look like an exported dataset "
            "(expected metadata.csv and wav_22050/).")

    try:
        project = projects.load(project_name)
    except FileNotFoundError:
        project = projects.create(project_name)

    existing = {c["id"] for c in project["clips"]}
    prefix = f"r{len(project['sources']) + 1}"
    speaker = max((c.get("speaker", 0) for c in project["clips"]), default=-1) + 1

    clip_dir = projects.project_path(project_name) / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    added, skipped, offset = [], 0, 0.0
    for line in meta.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        stem, text = line.split("|", 1)
        wav = wavs / f"{stem}.wav"
        if not wav.is_file():
            skipped += 1
            continue
        clip_id = f"{prefix}{stem}"
        if clip_id in existing:
            skipped += 1
            continue
        shutil.copy2(wav, clip_dir / f"{clip_id}.wav")
        length = _duration(wav)
        added.append({
            "id": clip_id, "file": f"{clip_id}.wav",
            "text": text.strip(),
            # Offsets are synthetic: the original timeline is gone, but clip
            # length is what the UI actually shows and what stats sum.
            "start": offset, "end": offset + length,
            "confidence": 1.0, "speaker": speaker,
            "include": True, "edited": True, "source": prefix,
        })
        offset += length

    if not added:
        raise SystemExit("nothing to recover -- every clip is already in the project.")

    project["clips"].extend(added)
    project["sources"].append({
        "file": f"(recovered from DATASETS/{dataset})", "original": f"DATASETS/{dataset}",
        "prefix": prefix, "imported": True, "clips": len(added),
    })
    project["speakers"] = speaker + 1
    project["duration"] = project.get("duration", 0.0) + offset
    project["status"] = "review"
    projects.save(project)
    return {"added": len(added), "skipped": skipped,
            "minutes": round(offset / 60, 1), "project": project_name}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    result = recover(argv[0], argv[1])
    print(f"Recovered {result['added']} clips "
          f"({result['minutes']} min) into project '{result['project']}'."
          + (f" Skipped {result['skipped']}." if result["skipped"] else ""))
    print("They are tagged as a separate group in Review so you can tell them "
          "apart from anything else in the project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
