"""Tests for creating a dojo and wiring it up for training.

These cover the failure that shipped: upstream's newdojo.sh ends with
`chown -R 1000:1000`, which a non-root user cannot do, and its `set -e` aborted
*after* the directory was created -- so the retry failed too, with a different
error. Both halves are regression-tested here: it must work as a non-1000 user,
and it must be safe to run twice.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tms_web import config, dojo                 # noqa: E402


class FakeJob:
    def set(self, *a, **k): pass
    def raise_if_cancelled(self): pass
    @property
    def cancelled(self): return False


def scaffold(root: Path) -> None:
    """A miniature DOJO_CONTENTS, shaped like the real one."""
    contents = root / "DOJO_CONTENTS"
    (contents / "scripts" / "utils").mkdir(parents=True)
    (contents / "target_voice_dataset").mkdir()
    (contents / "training_folder").mkdir()
    (contents / "voice_checkpoints").mkdir()
    run = contents / "run_training.sh"
    run.write_text("#!/bin/bash\necho hi\n")
    run.chmod(0o755)
    deep = contents / "scripts" / "utils" / "piper_training.sh"
    deep.write_text("#!/bin/bash\necho train\n")
    deep.chmod(0o755)
    (contents / "scripts" / "SETTINGS.txt").write_text(
        "PIPER_BATCH_SIZE=5\nVALIDATION_SPLIT=0.1\n")
    (contents / "scripts" / ".colors").write_text("RED=x\n")


def with_temp_dojo(fn):
    def wrapper():
        tmp = Path(tempfile.mkdtemp())
        original = config.DOJO_DIR
        try:
            config.DOJO_DIR = tmp
            scaffold(tmp)
            fn(tmp)
        finally:
            config.DOJO_DIR = original
            shutil.rmtree(tmp, ignore_errors=True)
    wrapper.__name__ = fn.__name__
    return wrapper


@with_temp_dojo
def test_creates_a_populated_dojo(tmp):
    d = dojo._create_dojo("joey")
    assert d == tmp / "joey_dojo"
    assert (d / "run_training.sh").is_file()
    assert (d / "scripts" / "utils" / "piper_training.sh").is_file()
    assert (d / "target_voice_dataset").is_dir()


@with_temp_dojo
def test_preserves_executable_bits(tmp):
    d = dojo._create_dojo("joey")
    for rel in ("run_training.sh", "scripts/utils/piper_training.sh"):
        assert os.access(d / rel, os.X_OK), f"{rel} lost its executable bit"


@with_temp_dojo
def test_copies_dotfiles_the_scripts_depend_on(tmp):
    # run_training.sh sources scripts/.colors and exits if it is missing.
    d = dojo._create_dojo("joey")
    assert (d / "scripts" / ".colors").is_file()


@with_temp_dojo
def test_is_idempotent(tmp):
    """The retry case. The original bug left a populated dir, so re-running
    upstream's script failed with 'directory already exists' forever."""
    dojo._create_dojo("joey")
    d = dojo._create_dojo("joey")          # must not raise
    assert (d / "run_training.sh").is_file()


@with_temp_dojo
def test_does_not_clobber_user_edits(tmp):
    """SETTINGS.txt is the user's after the first run -- re-running training
    must not silently reset their batch size."""
    d = dojo._create_dojo("joey")
    settings = d / "scripts" / "SETTINGS.txt"
    settings.write_text("PIPER_BATCH_SIZE=16\nVALIDATION_SPLIT=0.1\n")
    dojo._create_dojo("joey")
    assert "PIPER_BATCH_SIZE=16" in settings.read_text(), "user settings overwritten"


@with_temp_dojo
def test_missing_contents_says_what_is_wrong(tmp):
    shutil.rmtree(tmp / "DOJO_CONTENTS")
    try:
        dojo._create_dojo("joey")
    except RuntimeError as exc:
        assert "not seeded" in str(exc), f"unhelpful message: {exc}"
    else:
        raise AssertionError("a missing DOJO_CONTENTS must raise")


@with_temp_dojo
def test_prepare_dojo_writes_the_state_training_reads(tmp):
    """piper_training.sh exits if any of these are missing."""
    project = {
        "voice_name": "joey", "dataset_name": "joey", "quality": "medium",
        "batch_size": 12, "num_workers": 8, "from_scratch": False,
    }
    d = dojo.prepare_dojo(project, FakeJob())
    assert (d / "target_voice_dataset" / ".QUALITY").read_text().strip() == "M"
    assert (d / "target_voice_dataset" / ".SCRATCH").read_text().strip() == "false"
    assert (d / "scripts" / ".SAMPLING_RATE").read_text().strip() == "22050"
    assert (d / "scripts" / ".MAX_WORKERS").read_text().strip() == "8"
    assert "PIPER_BATCH_SIZE=12" in (d / "scripts" / "SETTINGS.txt").read_text()
    for link in ("wav", "metadata.csv", "dataset.conf"):
        assert (d / "target_voice_dataset" / link).is_symlink(), f"{link} not linked"
    # Relative, like the dojo's own links, so they resolve inside the container.
    target = os.readlink(d / "target_voice_dataset" / "wav")
    assert target == "../../DATASETS/joey/wav_22050", target


@with_temp_dojo
def test_prepare_dojo_is_rerunnable(tmp):
    """Pressing train twice must not fail on already-existing symlinks."""
    project = {"voice_name": "joey", "dataset_name": "joey", "quality": "medium",
               "batch_size": 8, "num_workers": 8, "from_scratch": False}
    dojo.prepare_dojo(project, FakeJob())
    dojo.prepare_dojo(project, FakeJob())


@with_temp_dojo
def test_x_low_quality_links_the_16k_audio(tmp):
    project = {"voice_name": "joey", "dataset_name": "joey", "quality": "x-low",
               "batch_size": 8, "num_workers": 8, "from_scratch": False}
    d = dojo.prepare_dojo(project, FakeJob())
    assert os.readlink(d / "target_voice_dataset" / "wav").endswith("wav_16000")
    assert (d / "scripts" / ".SAMPLING_RATE").read_text().strip() == "16000"
    assert (d / "target_voice_dataset" / ".QUALITY").read_text().strip() == "L"


@with_temp_dojo
def test_missing_checkpoint_refuses_rather_than_training_from_scratch(tmp):
    """Silently falling back wasted hours before the user could find out.

    From-scratch training needs orders of magnitude more audio and time than a
    fine-tune, so it has to be chosen, not stumbled into.
    """
    project = {"voice_name": "joey", "dataset_name": "joey", "quality": "medium",
               "voice_type": "M", "batch_size": 8, "num_workers": 8,
               "from_scratch": False}
    dojo.prepare_dojo(project, FakeJob())
    try:
        dojo.start_training(project, FakeJob())
    except RuntimeError as exc:
        msg = str(exc)
        assert "No pretrained" in msg, msg
        assert "tms checkpoints" in msg, "must say how to fix it"
        assert "en-gb" in msg, "must not only suggest the American pack"
    else:
        raise AssertionError("training must refuse without a checkpoint")


@with_temp_dojo
def test_finds_the_highest_epoch_checkpoint(tmp):
    folder = tmp / "PRETRAINED_CHECKPOINTS" / "default" / "M_voice" / "medium"
    folder.mkdir(parents=True)
    original = config.CHECKPOINTS_DIR
    try:
        config.CHECKPOINTS_DIR = tmp / "PRETRAINED_CHECKPOINTS"
        for name in ("epoch=9-step=1.ckpt", "epoch=2307-step=558536.ckpt",
                     "epoch=100-step=99.ckpt"):
            (folder / name).write_text("x")
        found = dojo.find_pretrained("M", "medium")
        assert found is not None and "2307" in found.name, \
            f"picked {found} -- epochs must compare numerically, not as strings"
    finally:
        config.CHECKPOINTS_DIR = original


def _ckpt(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / name
    f.write_text("x")
    return f


@with_temp_dojo
def test_resumes_from_the_dojos_own_latest_checkpoint(tmp):
    d = dojo._create_dojo("joey")
    logs = d / "training_folder" / "lightning_logs" / "version_0" / "checkpoints"
    _ckpt(logs, "epoch=12-val_mel=0.51.ckpt")
    _ckpt(logs, "epoch=52-val_mel=0.42.ckpt")
    _ckpt(logs, "epoch=30-val_mos=0.60.ckpt")
    found = dojo.find_resume_checkpoint("joey")
    assert found is not None and "epoch=52" in found.name, found


@with_temp_dojo
def test_ignores_last_ckpt_which_would_silently_restart(tmp):
    """last.ckpt has no val_mel=/val_mos= in its name, so piper_fit.py takes
    its legacy branch and resets the epoch counter to zero. Picking it would
    look like a resume and quietly throw the run away."""
    d = dojo._create_dojo("joey")
    logs = d / "training_folder" / "lightning_logs" / "version_0" / "checkpoints"
    _ckpt(logs, "last.ckpt")
    assert dojo.find_resume_checkpoint("joey") is None


@with_temp_dojo
def test_ignores_legacy_step_named_checkpoints(tmp):
    """A pretrained checkpoint (epoch=N-step=M) also hits the legacy branch."""
    d = dojo._create_dojo("joey")
    _ckpt(d / "voice_checkpoints", "epoch=6339-step=1647790.ckpt")
    assert dojo.find_resume_checkpoint("joey") is None


@with_temp_dojo
def test_no_checkpoints_means_no_resume(tmp):
    dojo._create_dojo("joey")
    assert dojo.find_resume_checkpoint("joey") is None


@with_temp_dojo
def test_resume_checkpoint_is_found_in_voice_checkpoints_too(tmp):
    d = dojo._create_dojo("joey")
    _ckpt(d / "voice_checkpoints", "epoch=7-val_mos=0.3.ckpt")
    found = dojo.find_resume_checkpoint("joey")
    assert found is not None and "epoch=7" in found.name


@with_temp_dojo
def test_export_names_and_configures_the_voice(tmp):
    """Piper identifies a voice by filename convention, and the training
    config is not a voice config until three fields are rewritten."""
    import json as _json
    project = {"voice_name": "joey", "dataset_name": "joey", "quality": "medium",
               "voice_type": "M", "piper_prefix": "en_AU", "espeak_language": "en",
               "batch_size": 8, "num_workers": 8, "from_scratch": False}
    d = dojo.prepare_dojo(project, FakeJob())
    (d / "training_folder").mkdir(exist_ok=True)
    (d / "training_folder" / "config.json").write_text(_json.dumps(
        {"audio": {"sample_rate": 22050}, "espeak": {"voice": "en"},
         "num_symbols": 256}))
    ckpt = _ckpt(d / "voice_checkpoints", "epoch=900-val_mel=0.31.ckpt")

    calls = []
    original = dojo._checkpoint_to_onnx

    def fake(dojo_path, src, onnx):
        calls.append((src, onnx))
        onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.write_bytes(b"onnx-bytes")

    try:
        dojo._checkpoint_to_onnx = fake
        result = dojo.export_voice(project, str(ckpt))
    finally:
        dojo._checkpoint_to_onnx = original

    assert result["name"] == "en_AU-joey-medium", result["name"]
    assert result["epoch"] == 900
    onnx = Path(result["onnx"])
    assert onnx.name == "en_AU-joey-medium.onnx"
    assert onnx.parent.name == "en_AU-joey-medium", "voice needs its own folder"

    cfg = _json.loads(Path(result["config"]).read_text())
    assert cfg["audio"]["quality"] == "medium", "clients read quality from here"
    assert cfg["dataset"] == "en_AU-joey-medium"
    assert cfg["audio"]["sample_rate"] == 22050, "existing fields must survive"

    # language.code is the BCP 47 code from the filename, NOT the espeak
    # identifier. Setting it to the espeak code makes Home Assistant file the
    # voice under bare "English" rather than the locale.
    assert cfg["language"]["code"] == "en_AU", \
        f"language.code must match the filename prefix, got {cfg['language']['code']}"
    assert cfg["language"]["family"] == "en"
    assert cfg["language"]["region"] == "AU"

    # espeak.voice is the espeak identifier and drives pronunciation: it must
    # keep the value training wrote, not be overwritten with the locale.
    assert cfg["espeak"]["voice"] == "en", "phonemizer config must survive"


@with_temp_dojo
def test_export_quality_label_matches_the_filename(tmp):
    """audio.quality must use the same spelling as the filename: x_low, not x-low."""
    import json as _json
    project = {"voice_name": "joey", "dataset_name": "joey", "quality": "x-low",
               "voice_type": "M", "piper_prefix": "en_GB", "espeak_language": "en",
               "batch_size": 8, "num_workers": 8, "from_scratch": False}
    d = dojo.prepare_dojo(project, FakeJob())
    (d / "training_folder").mkdir(exist_ok=True)
    (d / "training_folder" / "config.json").write_text(_json.dumps({"audio": {}}))
    ckpt = _ckpt(d / "voice_checkpoints", "epoch=5-val_mel=0.4.ckpt")

    original = dojo._checkpoint_to_onnx
    try:
        dojo._checkpoint_to_onnx = lambda a, b, c: (
            c.parent.mkdir(parents=True, exist_ok=True), c.write_bytes(b"x"))
        result = dojo.export_voice(project, str(ckpt))
    finally:
        dojo._checkpoint_to_onnx = original

    assert result["name"] == "en_GB-joey-x_low", result["name"]
    cfg = _json.loads(Path(result["config"]).read_text())
    assert cfg["audio"]["quality"] == "x_low", \
        f"quality must match the filename spelling, got {cfg['audio']['quality']}"


@with_temp_dojo
def test_export_without_training_config_says_why(tmp):
    project = {"voice_name": "joey", "dataset_name": "joey", "quality": "medium",
               "voice_type": "M", "batch_size": 8, "num_workers": 8,
               "from_scratch": False}
    d = dojo.prepare_dojo(project, FakeJob())
    ckpt = _ckpt(d / "voice_checkpoints", "epoch=1-val_mel=0.9.ckpt")
    original = dojo._checkpoint_to_onnx
    try:
        dojo._checkpoint_to_onnx = lambda a, b, c: c.parent.mkdir(parents=True, exist_ok=True) or c.write_bytes(b"x")
        dojo.export_voice(project, str(ckpt))
    except RuntimeError as exc:
        assert "config.json" in str(exc) and "train" in str(exc), exc
    else:
        raise AssertionError("must explain the missing training config")
    finally:
        dojo._checkpoint_to_onnx = original


def run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:                          # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n  {len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
