"""Drive the dojo without its interactive console.

`piper_training.sh` is already headless: it reads everything it needs from a
handful of small state files that `link_dataset.sh` normally writes by asking
the user questions. So rather than automate a tmux UI, we write those files
directly and call the training script. Same code path the interactive flow
uses, no fork to maintain.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from . import config
from .jobs import Job
from .proc import run as sh

QUALITY_CODE = {"x-low": "L", "medium": "M", "high": "H"}
# Which prepared audio directory each quality trains from, matching the keys
# create_dataset.sh writes into dataset.conf.
QUALITY_RATE = {"x-low": 16000, "medium": 22050, "high": 22050}


def dataset_dir(name: str) -> Path:
    return config.DATASETS_DIR / name


def dojo_dir(voice: str) -> Path:
    return config.DOJO_DIR / f"{voice}_dojo"


def _resample(src: Path, dst: Path, rate: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh(["ffmpeg", "-nostdin", "-y", "-i", str(src),
        "-ac", "1", "-ar", str(rate), str(dst)])


def _create_dojo(voice: str) -> Path:
    """Create <voice>_dojo from DOJO_CONTENTS.

    Deliberately NOT upstream's newdojo.sh. That script ends with
    `chown -R 1000:1000`, which assumes the container runs as UID 1000. This
    image runs as Unraid's 99:100 by default, a non-root user cannot chown to
    a different UID, and the script's `set -e` then aborts having already
    created and populated the directory -- so every retry also fails, with
    "directory already exists". We already run as the user we want files owned
    by, so the chown was redundant here anyway.

    Idempotent on purpose: it must be safe to re-run after a failed attempt.
    """
    dojo = dojo_dir(voice)
    contents = config.DOJO_DIR / "DOJO_CONTENTS"
    if not contents.is_dir():
        raise RuntimeError(
            f"{contents} is missing. The dojo is not seeded -- restart the container.")
    dojo.mkdir(parents=True, exist_ok=True)
    for item in contents.iterdir():
        dest = dojo / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest, symlinks=True)
        else:
            shutil.copy2(item, dest)          # copy2 keeps the executable bit
    return dojo


def export_dataset(project: dict, clip_dir: Path, job: Job) -> Path:
    """Write the included clips out as a dojo dataset.

    Produces the same layout create_dataset.sh does (wav_22050 + wav_16000 +
    metadata.csv + dataset.conf) so the rest of the dojo cannot tell the
    difference, but without its interactive prompts.
    """
    name = project["dataset_name"]
    out = dataset_dir(name)
    for sub in ("wav_22050", "wav_16000"):
        shutil.rmtree(out / sub, ignore_errors=True)

    included = [c for c in project["clips"] if c.get("include")]
    if not included:
        raise RuntimeError("no clips are included -- nothing to train on")

    rows = []
    for n, clip in enumerate(included, 1):
        job.raise_if_cancelled()
        src = clip_dir / clip["file"]
        _resample(src, out / "wav_22050" / clip["file"], 22050)
        _resample(src, out / "wav_16000" / clip["file"], 16000)
        # metadata.csv keys on the stem, with no extension.
        rows.append(f"{Path(clip['file']).stem}|{clip['text'].strip()}")
        if n % 20 == 0 or n == len(included):
            job.set(0.05 + 0.75 * (n / len(included)), f"exporting clips {n}/{len(included)}")

    (out / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    conf = "\n".join([
        f'NAME="{name}"',
        f'DESCRIPTION="{project.get("description", "Created with the TextyMcSpeechy web UI")}"',
        f'DEFAULT_VOICE_TYPE="{project.get("voice_type", "M")}"',
        'LOW_AUDIO="wav_16000"',
        'MEDIUM_AUDIO="wav_22050"',
        'HIGH_AUDIO="wav_22050"',
        f'ESPEAK_LANGUAGE_IDENTIFIER={project.get("espeak_language", "en-us")}',
        f'PIPER_FILENAME_PREFIX={project.get("piper_prefix", "en_US")}',
    ]) + "\n"
    (out / "dataset.conf").write_text(conf, encoding="utf-8")
    job.set(0.82, f"dataset written to DATASETS/{name} ({len(included)} clips)")
    return out


def find_pretrained(voice_type: str, quality: str) -> Path | None:
    """Highest-epoch pretrained checkpoint for this voice type and quality."""
    folder = config.CHECKPOINTS_DIR / "default" / f"{voice_type}_voice" / quality
    ckpts = sorted(folder.glob("*.ckpt")) if folder.is_dir() else []
    if not ckpts:
        return None

    return max(ckpts, key=_epoch_of)


def _epoch_of(path: Path) -> int:
    m = re.search(r"epoch=(\d+)", path.name)
    return int(m.group(1)) if m else -1


def find_resume_checkpoint(voice: str) -> Path | None:
    """The newest checkpoint from this dojo's OWN previous training, if any.

    Only checkpoints whose filename carries val_mel= or val_mos= qualify.
    piper_fit.py decides resume-vs-restart purely by that pattern: anything
    else -- last.ckpt, or a pretrained checkpoint named epoch=N-step=M -- takes
    its "legacy" branch, which loads the weights but restarts the epoch counter
    and the optimizer at zero. Handing it the wrong file does not error, it
    just silently throws away your training progress.
    """
    dojo = dojo_dir(voice)
    found: list[Path] = []
    for folder in (dojo / "voice_checkpoints",
                   dojo / "training_folder" / "lightning_logs"):
        if folder.is_dir():
            found += [p for p in folder.rglob("*.ckpt")
                      if re.search(r"val_(?:mel|mos)=", p.name)]
    return max(found, key=_epoch_of) if found else None


def prepare_dojo(project: dict, job: Job) -> Path:
    """Create the dojo and write the state files training reads.

    Mirrors link_dataset.sh: relative symlinks into DATASETS, plus the four
    dotfiles that carry the answers it would otherwise prompt for.
    """
    voice = project["voice_name"]
    quality = project.get("quality", "medium")
    dojo = dojo_dir(voice)

    job.set(0.85, f"preparing {voice}_dojo")
    _create_dojo(voice)

    target = dojo / "target_voice_dataset"
    target.mkdir(parents=True, exist_ok=True)
    (dojo / "training_folder").mkdir(exist_ok=True)

    audio_key = "wav_16000" if quality == "x-low" else "wav_22050"
    dataset = project["dataset_name"]
    # Relative, like the dojo's own links, so they resolve the same way.
    links = {
        target / "wav": f"../../DATASETS/{dataset}/{audio_key}",
        target / "metadata.csv": f"../../DATASETS/{dataset}/metadata.csv",
        target / "dataset.conf": f"../../DATASETS/{dataset}/dataset.conf",
    }
    for link, dest in links.items():
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(dest)

    (target / ".QUALITY").write_text(QUALITY_CODE[quality] + "\n")
    (target / ".SCRATCH").write_text(
        "true\n" if project.get("from_scratch") else "false\n")
    (dojo / "scripts" / ".SAMPLING_RATE").write_text(f"{QUALITY_RATE[quality]}\n")
    (dojo / "scripts" / ".MAX_WORKERS").write_text(
        f"{project.get('num_workers', 8)}\n")

    settings = dojo / "scripts" / "SETTINGS.txt"
    if settings.exists() and project.get("batch_size"):
        text = settings.read_text()
        text = re.sub(r"^PIPER_BATCH_SIZE=.*$",
                      f"PIPER_BATCH_SIZE={int(project['batch_size'])}",
                      text, count=1, flags=re.M)
        settings.write_text(text)

    job.set(0.95, f"{voice}_dojo ready")
    return dojo


def start_training(project: dict, job: Job) -> None:
    """Resume this dojo's training, or start a fine-tune from a pretrained one."""
    voice = project["voice_name"]
    dojo = dojo_dir(voice)
    quality = project.get("quality", "medium")

    ckpt = ""
    # Continue this dojo's own run unless the user explicitly asked to start
    # over. Without this a restart -- deliberate, or just a container bounce --
    # silently threw away every epoch trained so far.
    if not project.get("from_scratch") and not project.get("restart"):
        resume = find_resume_checkpoint(voice)
        if resume:
            job.set(message=f"resuming from epoch {_epoch_of(resume)} ({resume.name})")
            _run_training(dojo, str(resume.resolve()), job)
            return

    if not project.get("from_scratch"):
        found = find_pretrained(project.get("voice_type", "M"), quality)
        if not found:
            # Deliberately fatal. Quietly falling back to from-scratch training
            # burns tens of hours and needs far more data than a fine-tune, so
            # the user must choose it rather than arrive at it by accident.
            voice_type = project.get("voice_type", "M")
            raise RuntimeError(
                f"No pretrained {quality} checkpoint for a {voice_type}_voice.\n"
                f"Expected a .ckpt in PRETRAINED_CHECKPOINTS/default/"
                f"{voice_type}_voice/{quality}/.\n\n"
                "Download a set from the container console, picking the language "
                "closest to your speaker's accent:\n"
                "    tms checkpoints en-gb     (British, and the better match for "
                "Australian, Irish and other non-rhotic accents)\n"
                "    tms checkpoints en-us     (American)\n\n"
                "Or tick 'train from scratch' if you really mean it -- that needs "
                "hours of audio and days of training, not minutes."
            )
        ckpt = str(found.resolve())
        job.set(message=f"fine-tuning from {found.name}")

    _run_training(dojo, ckpt, job)


def _run_training(dojo: Path, ckpt: str, job: Job) -> None:
    """Run piper_training.sh, streaming its output into the job log."""
    cmd = ["bash", "utils/piper_training.sh"] + ([ckpt] if ckpt else [])
    job.set(0.0, "starting training")
    proc = subprocess.Popen(
        cmd, cwd=dojo / "scripts", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    job.result = {"pid": proc.pid}

    epoch_re = re.compile(r"[Ee]poch\s+(\d+)")
    try:
        for line in proc.stdout:                          # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                job.set(message=line)
            m = epoch_re.search(line)
            if m:
                job.result = {"pid": proc.pid, "epoch": int(m.group(1))}
            if job.cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                job.raise_if_cancelled()
    finally:
        if proc.poll() is None:
            proc.terminate()

    code = proc.wait()
    if code != 0 and not job.cancelled:
        raise RuntimeError(f"training exited with code {code} -- see the log above")


def list_checkpoints(voice: str) -> list[dict]:
    """Checkpoints available to sample, newest epoch first."""
    dojo = dojo_dir(voice)
    found: list[dict] = []
    for folder in (dojo / "training_folder" / "lightning_logs",
                   dojo / "voice_checkpoints"):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.ckpt"):
            m = re.search(r"epoch=(\d+)", path.name)
            found.append({
                "path": str(path),
                "name": path.name,
                "epoch": int(m.group(1)) if m else -1,
                "size_mb": round(path.stat().st_size / 1e6, 1),
            })
    found.sort(key=lambda c: c["epoch"], reverse=True)
    return found


def sample_voice(voice: str, checkpoint: str, text: str, out_wav: Path) -> Path:
    """Export a checkpoint to onnx (cached) and speak `text` with it.

    This is the point of the whole exercise: hearing a checkpoint is how you
    decide training is done, and it is far more informative than the loss curve.
    """
    dojo = dojo_dir(voice)
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    onnx_dir = dojo / "tts_voices" / "_webui_samples"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx = onnx_dir / f"{ckpt.stem}.onnx"

    if not onnx.exists():
        sh(["python3", str(dojo / "scripts" / "utils" / "export_onnx.py"),
            "--checkpoint", str(ckpt), "--output-file", str(onnx)],
           cwd="/app/piper")
        cfg = dojo / "training_folder" / "config.json"
        if cfg.exists():
            shutil.copy(cfg, onnx.with_suffix(".onnx.json"))

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sh(["python3", "-m", "piper", "-m", str(onnx), "-f", str(out_wav)], stdin=text)
    return out_wav
