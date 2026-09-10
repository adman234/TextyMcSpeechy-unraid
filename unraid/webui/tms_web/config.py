"""Filesystem layout for the web UI.

Everything the user creates lives under the appdata mount so it survives image
updates, exactly like the rest of the dojo.
"""
from __future__ import annotations

import os
from pathlib import Path

DOJO_DIR = Path(os.environ.get("TMS_DOJO_DIR", "/app/tts_dojo"))

# Web UI working area. Kept beside the dojo rather than inside DATASETS so a
# half-reviewed import never looks like a finished dataset to the dojo scripts.
WEB_DIR = DOJO_DIR / "WEBUI"
UPLOAD_DIR = WEB_DIR / "uploads"
PROJECT_DIR = WEB_DIR / "projects"

DATASETS_DIR = DOJO_DIR / "DATASETS"
CHECKPOINTS_DIR = DOJO_DIR / "PRETRAINED_CHECKPOINTS"

# Piper medium quality trains at 22050 Hz; whisper wants 16 kHz mono.
CLIP_RATE = 22050
ASR_RATE = 16000

# Clip length bounds. Upstream recommends 1-15s; anything outside that is more
# likely a segmentation mistake than a usable utterance.
MIN_CLIP_SECONDS = 1.2
MAX_CLIP_SECONDS = 15.0
# Where to prefer splitting: a pause at least this long reads as a sentence end.
SPLIT_PAUSE_SECONDS = 0.45
TARGET_CLIP_SECONDS = 8.0

WHISPER_MODEL = os.environ.get("TMS_WHISPER_MODEL", "large-v3")
WHISPER_COMPUTE = os.environ.get("TMS_WHISPER_COMPUTE", "float16")
MODEL_CACHE = Path(os.environ.get("TMS_MODEL_CACHE", str(WEB_DIR / "models")))


def ensure_dirs() -> None:
    for d in (WEB_DIR, UPLOAD_DIR, PROJECT_DIR, MODEL_CACHE):
        d.mkdir(parents=True, exist_ok=True)
