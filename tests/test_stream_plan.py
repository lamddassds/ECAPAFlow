"""Streaming chunk-scheduler tests — runnable standalone:

    python tests/test_stream_plan.py

Covers the pure planning pieces of the adaptive streaming path: sentence
splitting, runway-based chunk allowance, sentence merging, and the
persistent fixed+variable timing history. Importing app is safe here: the
engine loads lazily via the FastAPI lifespan, which never runs on plain
import.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
from app import (  # noqa: E402
    CHUNK_CHARS, MIN_CHUNK_AUDIO_S, RUNWAY_SAFETY, RtfHistory,
    _split_sentences, _stream_allowance, _take_chunk,
)


def test_split_sentences():
    # plain sentences stay separate (the atoms of the stream scheduler)
    s = _split_sentences("One. Two! Three?")
    assert s == ["One.", "Two!", "Three?"], s
    # run-ons longer than the limit are hard-split at a space
    long = "word " * 100  # 500 chars, no sentence punctuation
    parts = _split_sentences(long.strip(), limit=120, natural=False)
    assert all(len(p) <= 120 for p in parts), [len(p) for p in parts]
    assert " ".join(parts).split() == long.split()
    # the natural path splits the same run-on but marks every non-final
    # piece as a continuation (trailing comma) so the model does not put a
    # sentence-final fall on a sentence that has not ended
    nat = _split_sentences(long.strip(), limit=120, natural=True)
    assert all(len(p) <= 120 for p in nat), [len(p) for p in nat]
    assert all(p.endswith(",") for p in nat[:-1]), nat
    assert " ".join(nat).replace(",", "").split() == long.split()
    # empty input
    assert _split_sentences("") == []


def test_split_sentences_prefers_clause_boundaries():
    # a run-on with a comma inside the window is cut AT the comma, not at
    # whatever space happens to sit at the character limit
    s = ("alpha beta gamma delta epsilon, zeta eta theta iota kappa lambda "
         "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega")
    parts = _split_sentences(s, limit=70, natural=True)
    assert parts[0] == "alpha beta gamma delta epsilon,", parts[0]


def test_auto_steps_prefers_measured_stream_rtf():
    """Auto quality must believe a measured whole-stream RTF over the
    steady-state model — the model is structurally optimistic and once put
    the app on a rung that streamed at 1.10."""
    with tempfile.TemporaryDirectory() as d:
        hist = RtfHistory(Path(d) / "h.json")
        saved = app._rtf_history
        app._rtf_history = hist
        try:
            # nothing measured: the penalty keeps Auto off unproven top rungs
            low = app._auto_steps()
            assert low in app.AUTO_STEPS_LADDER
            # a rung that really streamed too slow is never chosen…
            hist.note_stream(48, 1.10, audio_s=30.0)
            hist.note_stream(32, 0.75, audio_s=30.0)
            assert app._auto_steps() == 32
            # …and a rung that turns out fine is chosen even if it is the top
            hist.note_stream(48, 0.40, audio_s=30.0)
            hist.note_stream(48, 0.40, audio_s=30.0)
            hist.note_stream(48, 0.40, audio_s=30.0)
            assert app._auto_steps() == 48
            # short runs are not representative and must be ignored
            hist2 = RtfHistory(Path(d) / "h2.json")
            hist2.note_stream(32, 9.9, audio_s=1.0)
            assert hist2.stream_rtf(32) is None
        finally:
            app._rtf_history = saved


def test_stream_rtf_survives_a_model_reseed():
    with tempfile.TemporaryDirectory() as d:
        hist = RtfHistory(Path(d) / "h.json")
        hist.note_stream(16, 0.42, audio_s=20.0)
        hist.update(16, chars=200, audio_s=12.0, infer_s=4.0)  # reseeds the row
        assert hist.stream_rtf(16) == 0.42


def test_take_chunk_respects_allowance():
    sents = [(0, "AAAA.", "en"), (0, "BBBB.", "en"), (0, "CCCC.", "en")]
    # allowance of 1 char -> exactly one sentence (never zero)
    assert _take_chunk(sents, 0, 1) == 1
    # allowance fits two sentences joined with a space (5+1+5=11)
    assert _take_chunk(sents, 0, 11) == 2
    # huge allowance -> all three
    assert _take_chunk(sents, 0, 1000) == 3


def test_take_chunk_stops_at_paragraph():
    sents = [(0, "One.", "en"), (0, "Two.", "en"), (1, "Next para.", "en")]
    assert _take_chunk(sents, 0, 1000) == 2  # never merges across paragraphs
    assert _take_chunk(sents, 2, 1000) == 3


def test_stream_allowance():
    fixed_s, rate_rtf, cps = 0.5, 0.15, 14.0
    # Deep in a deficit — even the fixed per-call cost doesn't fit inside
    # the safe runway (e.g. Quality mode outrunning realtime) — go BIG, not
    # small: the fixed cost is paid regardless of chunk size, so shrinking
    # chunks while behind just pays that cost over and over for almost no
    # audio each time (Finding 9). Never the old "shrink toward a floor".
    assert _stream_allowance(0.0, fixed_s, rate_rtf, cps, CHUNK_CHARS) == CHUNK_CHARS
    assert _stream_allowance(-3.0, fixed_s, rate_rtf, cps, CHUNK_CHARS) == CHUNK_CHARS
    # Once safe runway covers the fixed cost, chunk size grows monotonically.
    a = _stream_allowance(2.0, fixed_s, rate_rtf, cps, CHUNK_CHARS)
    b = _stream_allowance(4.0, fixed_s, rate_rtf, cps, CHUNK_CHARS)
    assert a < b < CHUNK_CHARS, (a, b)
    expected_b = int(min(
        CHUNK_CHARS,
        max(MIN_CHUNK_AUDIO_S, (RUNWAY_SAFETY * 4.0 - fixed_s) / rate_rtf) * cps))
    assert b == expected_b, (b, expected_b)
    # Huge runway is capped at the engine's one-pass limit.
    assert _stream_allowance(1000.0, fixed_s, rate_rtf, cps, CHUNK_CHARS) == CHUNK_CHARS
    # Pathological rate estimate never divides by ~zero.
    assert _stream_allowance(5.0, 0.1, 0.0, cps, CHUNK_CHARS) >= 1
    # Never below the MIN_CHUNK_AUDIO_S floor once there IS spare runway.
    just_over = fixed_s / RUNWAY_SAFETY + 0.01
    small = _stream_allowance(just_over, fixed_s, rate_rtf, cps, CHUNK_CHARS)
    assert small == int(MIN_CHUNK_AUDIO_S * cps), small


def test_falls_behind_requests_larger_not_smaller_chunks():
    """Finding 9 regression: on this fixed-cost-dominated hardware, the OLD
    floor-based logic requested the SMALLEST possible chunk once behind —
    the worst possible choice, since the fixed per-call cost is paid
    regardless of chunk size. Once truly behind, the scheduler must request
    the LARGEST allowed chunk instead, to amortize that unavoidable cost."""
    fixed_s, rate_rtf, cps = 2.0, 0.1, 14.0
    for runway in (0.0, -1.0, -10.0):
        allowed = _stream_allowance(runway, fixed_s, rate_rtf, cps, CHUNK_CHARS)
        assert allowed == CHUNK_CHARS, (runway, allowed)


def test_chunks_grow_with_runway_fixed_cost_model():
    """The user-visible behaviour, simulated under a FIXED-COST-DOMINATED
    engine (this hardware's reality: infer_time = fixed_s + rate_rtf *
    audio_s, cost nearly independent of chunk size) rather than the old
    pure-ratio assumption a plain rtf*duration model got wrong. Chunk 1 is
    still exactly one sentence (fastest first audio); later chunks grow as
    the runway builds."""
    sents = [(0, "This is a sentence of text.", "en")] * 30
    fixed_s, rate_rtf, cps = 0.5, 0.1, 14.0  # net RTF < 1: keeps up long-run
    audio_s, clock = 0.0, 0.0
    sizes = []
    i = 0
    first = True
    while i < len(sents):
        if first:
            allowed = 1
            first = False
        else:
            allowed = _stream_allowance(audio_s - clock, fixed_s, rate_rtf, cps, CHUNK_CHARS)
        j = _take_chunk(sents, i, allowed)
        chars = sum(len(s) for _, s, _ in sents[i:j]) + (j - i - 1)
        chunk_audio = chars / cps
        clock += fixed_s + chunk_audio * rate_rtf   # fixed+variable cost model
        audio_s += chunk_audio
        sizes.append(j - i)
        i = j
    assert sizes[0] == 1, sizes                      # instant first audio
    assert sizes[1] >= sizes[0], sizes               # growing…
    assert max(sizes) > 3, sizes                     # …to real batches
    assert sum(sizes) == len(sents)                  # nothing lost


def test_rtf_history():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rtf_history.json"
        h = RtfHistory(path)
        # empty -> conservative defaults, conservative hedge (too few samples)
        fixed_s, rate_rtf, cps, safety = h.estimates(16)
        assert (fixed_s, rate_rtf, cps) == (app.DEFAULT_FIXED_S, app.DEFAULT_RATE_RTF, app.DEFAULT_CPS)
        assert safety == 1.15

        # a small chunk (audio_s < SMALL_AUDIO_S) teaches the FIXED term
        h.update(16, chars=20, audio_s=1.0, infer_s=1.2)
        fixed_after, rate_after, _, _ = h.estimates(16)
        assert fixed_after != app.DEFAULT_FIXED_S

        # a large chunk teaches the RATE term
        h.update(16, chars=280, audio_s=20.0, infer_s=fixed_after + 20.0 * 0.3)
        fixed2, rate2, cps2, _ = h.estimates(16)
        assert rate2 != rate_after

        # unmeasured steps scale linearly from the nearest measured value
        # (both the fixed dispatch cost and the marginal compute scale with
        # diffusion step count)
        fixed_32, rate_32, cps_32, _ = h.estimates(32)
        assert abs(fixed_32 - fixed2 * 2) < 1e-6
        assert abs(rate_32 - rate2 * 2) < 1e-6
        assert cps_32 == cps2

        # persisted: a fresh instance reads the same numbers
        h2 = RtfHistory(path)
        assert h2.estimates(16)[:3] == h.estimates(16)[:3]

        # garbage on disk never breaks startup
        path.write_text("not json")
        h3 = RtfHistory(path)
        assert h3.estimates(16)[:3] == (app.DEFAULT_FIXED_S, app.DEFAULT_RATE_RTF, app.DEFAULT_CPS)

        # invalid measurements are ignored
        h3.update(16, chars=10, audio_s=0.0, infer_s=-1.0)
        assert h3.estimates(16)[:3] == (app.DEFAULT_FIXED_S, app.DEFAULT_RATE_RTF, app.DEFAULT_CPS)

        # old-schema rows ({"rtf":.., "cps":.., "n":..}, pre-2026-07-05) are
        # treated as absent rather than crashing on the missing keys
        path.write_text(json.dumps({"16": {"rtf": 0.6, "cps": 15.0, "n": 2}}))
        h4 = RtfHistory(path)
        assert h4.estimates(16)[:3] == (app.DEFAULT_FIXED_S, app.DEFAULT_RATE_RTF, app.DEFAULT_CPS)
        h4.update(16, chars=20, audio_s=1.0, infer_s=1.2)  # must not KeyError


def test_safety_percentile_hedges_after_slow_chunks():
    """NetEQ reads a high percentile off a delay histogram rather than an
    EMA-of-the-mean to size its safety margin — mirrored here: a handful of
    much-slower-than-predicted chunks should push the p90 safety multiplier
    above 1.0, so future chunk-size decisions hedge against a repeat."""
    with tempfile.TemporaryDirectory() as td:
        h = RtfHistory(Path(td) / "rtf_history.json")
        for _ in range(3):
            h.update(16, chars=200, audio_s=14.0, infer_s=2.0)  # roughly on-model
        for _ in range(3):
            h.update(16, chars=200, audio_s=14.0, infer_s=6.0)  # thermal-throttle-style outlier
        _, _, _, safety = h.estimates(16)
        assert safety > 1.0, safety


TESTS = [
    test_split_sentences,
    test_take_chunk_respects_allowance,
    test_take_chunk_stops_at_paragraph,
    test_stream_allowance,
    test_falls_behind_requests_larger_not_smaller_chunks,
    test_chunks_grow_with_runway_fixed_cost_model,
    test_rtf_history,
    test_safety_percentile_hedges_after_slow_chunks,
    test_split_sentences_prefers_clause_boundaries,
    test_auto_steps_prefers_measured_stream_rtf,
    test_stream_rtf_survives_a_model_reseed,
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed = 1
    sys.exit(failed)
