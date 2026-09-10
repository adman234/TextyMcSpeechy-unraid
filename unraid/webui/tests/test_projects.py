"""Tests for project state, especially the parts that can destroy user work.

Correcting transcripts is the most expensive thing a user does here -- hours of
listening and typing. An earlier version replaced the whole clip list on every
import, so uploading a second recording silently wiped it. These tests exist so
that cannot come back.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tms_web import config, projects              # noqa: E402


def with_temp_projects(fn):
    def wrapper():
        tmp = Path(tempfile.mkdtemp())
        original = config.PROJECT_DIR
        try:
            config.PROJECT_DIR = tmp
            fn(tmp)
        finally:
            config.PROJECT_DIR = original
            shutil.rmtree(tmp, ignore_errors=True)
    wrapper.__name__ = fn.__name__
    return wrapper


@with_temp_projects
def test_new_project_starts_with_no_sources(tmp):
    p = projects.create("joey")
    assert p["sources"] == []
    assert p["clips"] == []


@with_temp_projects
def test_roundtrip_preserves_clip_edits(tmp):
    p = projects.create("joey")
    p["clips"] = [{"id": "s1c0001", "text": "corrected by hand",
                   "include": True, "edited": True, "start": 0, "end": 2,
                   "speaker": 0, "file": "s1c0001.wav"}]
    projects.save(p)
    back = projects.load("joey")
    assert back["clips"][0]["text"] == "corrected by hand"
    assert back["clips"][0]["edited"] is True


@with_temp_projects
def test_legacy_single_source_project_is_migrated(tmp):
    """Projects created before multi-source support must keep working."""
    p = projects.create("joey")
    del p["sources"]
    p["source"] = "source.mp3"
    p["clips"] = [{"id": "clip0001", "text": "hi", "include": True,
                   "start": 0, "end": 2, "speaker": 0, "file": "clip0001.wav"}]
    projects.save(p)

    back = projects.load("joey")
    assert "source" not in back, "legacy key should be folded away"
    assert len(back["sources"]) == 1
    assert back["sources"][0]["file"] == "source.mp3"
    assert back["sources"][0]["imported"] is True
    assert back["clips"][0]["source"] == "s1", "existing clips need a source tag"
    assert back["clips"][0]["text"] == "hi", "migration must not touch the text"


@with_temp_projects
def test_legacy_project_with_no_source_migrates_cleanly(tmp):
    p = projects.create("joey")
    del p["sources"]
    projects.save(p)
    assert projects.load("joey")["sources"] == []


@with_temp_projects
def test_stats_count_only_included_clips(tmp):
    p = projects.create("joey")
    p["clips"] = [
        {"id": "a", "start": 0, "end": 4, "include": True, "confidence": 0.9},
        {"id": "b", "start": 4, "end": 10, "include": False, "confidence": 0.9},
        {"id": "c", "start": 10, "end": 16, "include": True, "confidence": 0.4},
    ]
    s = projects.stats(p)
    assert s["total"] == 3
    assert s["included"] == 2
    assert s["included_seconds"] == 10.0, "excluded clips must not count"
    assert s["low_confidence"] == 1


@with_temp_projects
def test_save_is_atomic_enough_to_survive_a_reread(tmp):
    p = projects.create("joey")
    for i in range(20):
        p["clips"] = [{"id": f"c{i}", "text": f"take {i}", "include": True,
                       "start": 0, "end": 1}]
        projects.save(p)
        assert projects.load("joey")["clips"][0]["text"] == f"take {i}"


@with_temp_projects
def test_names_that_would_escape_the_project_directory_are_rejected(tmp):
    for bad in ("../etc", "a/b", "", ".", "-leading", "x" * 100):
        try:
            projects.validate_name(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted dangerous name {bad!r}")


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
