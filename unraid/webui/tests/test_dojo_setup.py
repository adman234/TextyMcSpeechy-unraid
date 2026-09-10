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
