"""HTTP API for the TextyMcSpeechy web UI."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, dojo, importer, projects
from .jobs import RUNNER, Job

config.ensure_dirs()
app = FastAPI(title="TextyMcSpeechy")
STATIC = Path(__file__).resolve().parent.parent / "static"


def _project_or_404(name: str) -> dict:
    try:
        return projects.load(name)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


def _clip(project: dict, clip_id: str) -> dict:
    for c in project["clips"]:
        if c["id"] == clip_id:
            return c
    raise HTTPException(404, f"no clip {clip_id}")


# --- projects ---------------------------------------------------------------

class NewProject(BaseModel):
    name: str
    voice_type: str = "M"
    quality: str = "medium"
    espeak_language: str = "en-us"
    piper_prefix: str = "en_US"
    description: str = ""


@app.get("/api/projects")
def api_list():
    return projects.listing()


@app.post("/api/projects")
def api_create(body: NewProject):
    try:
        return projects.create(**body.model_dump())
    except (ValueError, FileExistsError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/projects/{name}")
def api_get(name: str):
    project = _project_or_404(name)
    return {**project, "stats": projects.stats(project)}


@app.delete("/api/projects/{name}")
def api_delete(name: str):
    _project_or_404(name)
    projects.delete(name)
    return {"ok": True}


class Settings(BaseModel):
    voice_type: str | None = None
    quality: str | None = None
    espeak_language: str | None = None
    piper_prefix: str | None = None
    batch_size: int | None = None
    from_scratch: bool | None = None


@app.patch("/api/projects/{name}/settings")
def api_settings(name: str, body: Settings):
    project = _project_or_404(name)
    project.update({k: v for k, v in body.model_dump().items() if v is not None})
    projects.save(project)
    return {"ok": True}


# --- import -----------------------------------------------------------------

@app.post("/api/projects/{name}/upload")
async def api_upload(name: str, file: UploadFile = File(...)):
    project = _project_or_404(name)
    dest_dir = projects.project_path(name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"source{Path(file.filename or 'audio').suffix or '.audio'}"
    # Streamed, not read into memory: these are hour-plus recordings.
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    project["source"] = dest.name
    project["status"] = "uploaded"
    projects.save(project)
    return {"ok": True, "file": dest.name,
            "size_mb": round(dest.stat().st_size / 1e6, 1)}


@app.post("/api/projects/{name}/import")
def api_import(name: str):
    project = _project_or_404(name)
    if not project.get("source"):
        raise HTTPException(400, "upload an audio file first")
    source = projects.project_path(name) / project["source"]

    def work(job: Job):
        result = importer.run_import(projects.project_path(name), source, job)
        fresh = projects.load(name)
        fresh.update({**result, "status": "review"})
        projects.save(fresh)
        return {"clips": len(result["clips"])}

    try:
        return RUNNER.start("import", work).as_dict()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


# --- review -----------------------------------------------------------------

class ClipEdit(BaseModel):
    text: str | None = None
    include: bool | None = None
    speaker: int | None = None


@app.patch("/api/projects/{name}/clips/{clip_id}")
def api_edit_clip(name: str, clip_id: str, body: ClipEdit):
    project = _project_or_404(name)
    clip = _clip(project, clip_id)
    if body.text is not None and body.text != clip["text"]:
        clip["text"] = body.text
        clip["edited"] = True
    if body.include is not None:
        clip["include"] = body.include
    if body.speaker is not None:
        clip["speaker"] = body.speaker
    projects.save(project)
    return {"ok": True, "stats": projects.stats(project)}


class Bulk(BaseModel):
    action: str          # include_only_speaker | include_all | exclude_all | exclude_short
    speaker: int | None = None


@app.post("/api/projects/{name}/clips/bulk")
def api_bulk(name: str, body: Bulk):
    project = _project_or_404(name)
    for clip in project["clips"]:
        if body.action == "include_all":
            clip["include"] = True
        elif body.action == "exclude_all":
            clip["include"] = False
        elif body.action == "include_only_speaker":
            clip["include"] = clip.get("speaker", 0) == body.speaker
        elif body.action == "exclude_short":
            if clip["end"] - clip["start"] < config.MIN_CLIP_SECONDS * 1.5:
                clip["include"] = False
        else:
            raise HTTPException(400, f"unknown action {body.action!r}")
    projects.save(project)
    return {"ok": True, "stats": projects.stats(project)}


@app.get("/api/projects/{name}/clips/{clip_id}/audio")
def api_clip_audio(name: str, clip_id: str):
    project = _project_or_404(name)
    clip = _clip(project, clip_id)
    path = projects.project_path(name) / "clips" / clip["file"]
    if not path.is_file():
        raise HTTPException(404, "clip audio missing")
    return FileResponse(path, media_type="audio/wav")


# --- training ---------------------------------------------------------------

@app.post("/api/projects/{name}/train")
def api_train(name: str):
    project = _project_or_404(name)
    if not any(c.get("include") for c in project.get("clips", [])):
        raise HTTPException(400, "no clips are included")

    def work(job: Job):
        fresh = projects.load(name)
        dojo.export_dataset(fresh, projects.project_path(name) / "clips", job)
        dojo.prepare_dojo(fresh, job)
        fresh["status"] = "training"
        projects.save(fresh)
        dojo.start_training(fresh, job)
        done = projects.load(name)
        done["status"] = "trained"
        projects.save(done)

    try:
        return RUNNER.start("train", work).as_dict()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/projects/{name}/checkpoints")
def api_checkpoints(name: str):
    project = _project_or_404(name)
    return dojo.list_checkpoints(project["voice_name"])


class Sample(BaseModel):
    checkpoint: str
    text: str = "The quick brown fox jumps over the lazy dog."


@app.post("/api/projects/{name}/sample")
def api_sample(name: str, body: Sample):
    project = _project_or_404(name)
    out = projects.project_path(name) / "samples" / f"{Path(body.checkpoint).stem}.wav"
    try:
        dojo.sample_voice(project["voice_name"], body.checkpoint, body.text, out)
    except Exception as exc:                              # noqa: BLE001
        raise HTTPException(500, f"could not render sample: {exc}") from exc
    return {"url": f"/api/projects/{name}/sample/{out.name}"}


@app.get("/api/projects/{name}/sample/{filename}")
def api_sample_audio(name: str, filename: str):
    path = projects.project_path(name) / "samples" / Path(filename).name
    if not path.is_file():
        raise HTTPException(404, "sample not found")
    return FileResponse(path, media_type="audio/wav")


# --- jobs -------------------------------------------------------------------

@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = RUNNER.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    payload = job.as_dict()
    if job.kind == "train" and isinstance(job.result, dict):
        payload["epoch"] = job.result.get("epoch")
    return payload


@app.get("/api/jobs")
def api_jobs():
    return {kind: (job.as_dict() if (job := RUNNER.active(kind)) else None)
            for kind in ("import", "train")}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    if not RUNNER.cancel(job_id):
        raise HTTPException(404, "job is not running")
    return {"ok": True}


@app.get("/api/health")
def api_health():
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        arches = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
        supported = (f"sm_{cap[0]}{cap[1]}" in arches) if cap else False
    except Exception:                                     # noqa: BLE001
        gpu, supported = None, False
    usage = shutil.disk_usage(config.DOJO_DIR)
    return {
        "gpu": gpu,
        "gpu_supported": supported,
        "free_gb": round(usage.free / 1e9, 1),
        # Any checkpoint at all, not just the M/medium pair: the banner should
        # not nag someone who downloaded a different voice type or quality.
        "has_checkpoints": any(config.CHECKPOINTS_DIR.glob("default/*_voice/*/*.ckpt")),
    }


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
