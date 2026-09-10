"""Tests for clip grouping.

Grouping decides what the model actually trains on. Get it wrong and nothing
errors -- you just get a worse voice hours later, which is the most expensive
kind of bug here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tms_web import config                        # noqa: E402
from tms_web.importer import group_words          # noqa: E402


def words(spec):
    """Build a word list from (start, end, text) triples."""
    return [{"start": s, "end": e, "text": t, "prob": 0.9} for s, e, t in spec]


def evenly(count, dur=0.4, gap=0.05, start=0.0):
    out, t = [], start
    for i in range(count):
        out.append((t, t + dur, f"w{i}"))
        t += dur + gap
    return out


def test_splits_on_a_real_pause():
    # Two utterances either side of a six-second silence. Both are comfortably
    # longer than the minimum, so both must survive as separate clips.
    w = words(evenly(6) + evenly(6, start=9.0))
    clips = group_words(w)
    assert len(clips) == 2, f"expected a clip either side of the pause, got {len(clips)}"
    assert clips[0]["end"] < 9.0, "first clip must end before the silence"
    assert clips[1]["start"] >= 9.0, "second clip must start after the silence"


def test_short_trailing_fragment_is_dropped_not_merged():
    """A stray half-word after a long gap is noise, not the tail of the clip.

    Merging it back would splice two utterances across six seconds of silence,
    which trains the model on a pause that is not in the transcript.
    """
    w = words(evenly(6) + [(9.0, 9.4, "oh"), (9.5, 9.9, "hm")])
    clips = group_words(w)
    assert len(clips) == 1, "the sub-minimum fragment must be discarded"
    assert clips[0]["end"] < 9.0, "it must not be glued onto the previous clip"


def test_never_exceeds_the_hard_maximum():
    # 60 words with no pause at all: must still be chopped up.
    clips = group_words(words(evenly(60, dur=0.5, gap=0.01)))
    assert clips, "a long unbroken monologue must still yield clips"
    for c in clips:
        assert c["end"] - c["start"] <= config.MAX_CLIP_SECONDS + 0.6, \
            f"clip of {c['end'] - c['start']:.1f}s exceeds the Piper ceiling"


def test_drops_clips_that_are_too_short():
    clips = group_words(words([(0.0, 0.2, "hi")]))
    assert clips == [], "a 0.2s fragment is not a trainable utterance"


def test_keeps_text_and_confidence():
    clips = group_words(words(evenly(4)))
    assert clips
    assert clips[0]["text"] == "w0 w1 w2 w3"
    assert 0.0 < clips[0]["confidence"] <= 1.0


def test_every_word_survives_in_order():
    """No word may be silently dropped or reordered."""
    spec = evenly(40, dur=0.45, gap=0.06)
    clips = group_words(words(spec))
    joined = " ".join(c["text"] for c in clips).split()
    expected = [t for _, _, t in spec]
    assert joined == expected[:len(joined)], "words were reordered or lost"
    assert len(joined) >= len(expected) - 4, "too many words dropped"


def test_clips_are_in_order_and_do_not_overlap():
    clips = group_words(words(evenly(30)))
    for a, b in zip(clips, clips[1:]):
        assert a["end"] <= b["start"], "clips must not overlap"


def test_empty_input_is_not_a_crash():
    assert group_words([]) == []


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
