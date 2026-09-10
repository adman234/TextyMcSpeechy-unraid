"""Turn one long recording into reviewable clips.

Three stages: transcribe with word timestamps, group words into utterance-sized
clips at natural pauses, then cluster voice embeddings so a podcast's other
speakers can be excluded in one click instead of five hundred.

The clustering stage is optional by design. It needs a model download, and if
that is unavailable the import still succeeds with every clip in one speaker
group -- a worse experience, not a broken one.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config
from .jobs import Job


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def probe_bandwidth_hz(path: Path) -> int | None:
    """Source sample rate, as a proxy for how much high end actually exists.

    A recording that arrives at 8 kHz cannot contain anything above 4 kHz, and
    a voice trained on it will be permanently dull. Worth warning about before
    the user spends an evening proofreading it.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return int(out.stdout.strip())
    except (TypeError, ValueError):
        return None


def _decode(src: Path, dst: Path, rate: int) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(src),
         "-ac", "1", "-ar", str(rate), "-vn", str(dst)],
        check=True, capture_output=True,
    )


def transcribe(audio16k: Path, job: Job) -> list[dict]:
    """Word-level transcription. Words, not segments, so we control splitting."""
    from faster_whisper import WhisperModel

    job.set(0.10, f"loading whisper model ({config.WHISPER_MODEL})")
    try:
        model = WhisperModel(
            config.WHISPER_MODEL, device="cuda",
            compute_type=config.WHISPER_COMPUTE,
            download_root=str(config.MODEL_CACHE),
        )
    except Exception as exc:                              # noqa: BLE001
        # A CPU fallback takes hours on a 2-hour file, but finishing slowly
        # beats failing outright, and the message says which happened.
        job.set(message=f"GPU unavailable for ASR ({exc}); falling back to CPU")
        model = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type="int8",
            download_root=str(config.MODEL_CACHE),
        )

    job.set(0.15, "transcribing")
    segments, info = model.transcribe(
        str(audio16k), word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )

    words: list[dict] = []
    total = max(info.duration or 1.0, 1.0)
    for seg in segments:
        job.raise_if_cancelled()
        for w in (seg.words or []):
            words.append({"start": w.start, "end": w.end,
                          "text": w.word.strip(), "prob": w.probability})
        job.set(0.15 + 0.45 * min(seg.end / total, 1.0),
                f"transcribing {seg.end / 60:.1f} / {total / 60:.1f} min")
    return words


def group_words(words: list[dict]) -> list[dict]:
    """Group words into clips, preferring to break at real pauses.

    Piper wants utterance-length clips. Splitting purely on a length target
    chops words in half; splitting only on long pauses yields clips far past
    the 15s ceiling. So: break on a pause once the clip is long enough to be
    worth keeping, and force a break at the hard maximum.
    """
    clips: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if not cur:
            return
        start, end = cur[0]["start"], cur[-1]["end"]
        if end - start >= config.MIN_CLIP_SECONDS:
            text = " ".join(w["text"] for w in cur).strip()
            if text:
                clips.append({
                    "start": start, "end": end, "text": text,
                    # Mean word confidence is a good proxy for "check this one".
                    "confidence": sum(w["prob"] for w in cur) / len(cur),
                })
        cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        span = w["end"] - cur[0]["start"]
        gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 999.0

        if span >= config.MAX_CLIP_SECONDS:
            flush()
        elif span >= config.TARGET_CLIP_SECONDS and gap >= config.SPLIT_PAUSE_SECONDS / 2:
            flush()
        elif span >= config.MIN_CLIP_SECONDS and gap >= config.SPLIT_PAUSE_SECONDS:
            flush()
    flush()
    return clips


def cut_clips(audio22k: Path, clips: list[dict], out_dir: Path, job: Job) -> None:
    """Write one wav per clip, with a little room tone kept at each edge.

    Trimming hard to the word boundary teaches the model clicks, and clipping
    the first phoneme is a common cause of a voice that swallows consonants.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = 0.06
    for n, clip in enumerate(clips, 1):
        job.raise_if_cancelled()
        start = max(0.0, clip["start"] - pad)
        dur = (clip["end"] - clip["start"]) + pad * 2
        name = f"clip{n:04d}"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
             "-i", str(audio22k), "-ac", "1", "-ar", str(config.CLIP_RATE),
             str(out_dir / f"{name}.wav")],
            check=True, capture_output=True,
        )
        clip["id"] = name
        clip["file"] = f"{name}.wav"
        if n % 10 == 0 or n == len(clips):
            job.set(0.62 + 0.23 * (n / max(len(clips), 1)),
                    f"cutting clips {n}/{len(clips)}")


def cluster_speakers(clip_dir: Path, clips: list[dict], job: Job) -> int:
    """Group clips by who is speaking.

    Unsupervised clustering of voice embeddings, deliberately not pyannote:
    its diarization models are gated behind a Hugging Face licence acceptance,
    which would turn "install the container" into "go make an account". This is
    less precise on overlapping speech, but needs no account and is more than
    good enough to bulk-exclude a podcast host.

    Returns the number of speakers found; 1 means clustering was unavailable,
    in which case the user excludes clips by hand.
    """
    try:
        import numpy as np
        import torch
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:                              # noqa: BLE001
        job.set(message=f"speaker grouping unavailable ({exc}); all clips in one group")
        for clip in clips:
            clip["speaker"] = 0
        return 1

    try:
        job.set(0.86, "loading speaker model")
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(config.MODEL_CACHE / "ecapa"),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        import torchaudio
        vectors = []
        for n, clip in enumerate(clips, 1):
            job.raise_if_cancelled()
            wav, sr = torchaudio.load(str(clip_dir / clip["file"]))
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.no_grad():
                emb = encoder.encode_batch(wav).squeeze().cpu().numpy()
            vectors.append(emb / (np.linalg.norm(emb) + 1e-9))
            if n % 25 == 0:
                job.set(0.86 + 0.12 * (n / len(clips)), f"grouping speakers {n}/{len(clips)}")

        if len(vectors) < 2:
            for clip in clips:
                clip["speaker"] = 0
            return 1

        # Cosine distance with a fixed cut. Tuned so a two-host podcast splits
        # cleanly; the user can always merge groups in the UI.
        matrix = linkage(pdist(np.vstack(vectors), metric="cosine"), method="average")
        labels = fcluster(matrix, t=0.55, criterion="distance")
        for clip, label in zip(clips, labels):
            clip["speaker"] = int(label) - 1
        return int(labels.max())
    except Exception as exc:                              # noqa: BLE001
        job.set(message=f"speaker grouping failed ({exc}); all clips in one group")
        for clip in clips:
            clip["speaker"] = 0
        return 1


def run_import(project_dir: Path, source: Path, job: Job) -> dict:
    """Full pipeline. Returns the project manifest."""
    work = project_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    clip_dir = project_dir / "clips"

    job.set(0.02, "decoding audio")
    a16 = work / "asr.wav"
    a22 = work / "master.wav"
    _decode(source, a16, config.ASR_RATE)
    _decode(source, a22, config.CLIP_RATE)

    duration = probe_duration(a22)
    source_rate = probe_bandwidth_hz(source)

    words = transcribe(a16, job)
    if not words:
        raise RuntimeError("no speech found in this file")

    job.set(0.60, "grouping words into clips")
    clips = group_words(words)
    if not clips:
        raise RuntimeError("transcription produced no usable clips")

    cut_clips(a22, clips, clip_dir, job)
    speakers = cluster_speakers(clip_dir, clips, job)

    for clip in clips:
        clip.setdefault("speaker", 0)
        clip["include"] = True
        clip["edited"] = False

    # The master decode is large and only needed during import.
    for leftover in (a16, a22):
        leftover.unlink(missing_ok=True)

    warnings = []
    if source_rate and source_rate < 22050:
        warnings.append(
            f"Source is {source_rate} Hz, so it has no audio above {source_rate // 2} Hz. "
            "Piper trains at 22050 Hz and the trained voice will sound dull. "
            "This cannot be fixed by training and is not a bug."
        )
    return {
        "duration": duration,
        "source_rate": source_rate,
        "speakers": speakers,
        "clips": clips,
        "warnings": warnings,
    }
