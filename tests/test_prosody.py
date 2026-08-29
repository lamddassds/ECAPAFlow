"""Natural-speech layer tests — runnable standalone:

    python tests/test_prosody.py

Covers prosody.py (pause table, breath debt, determinism, edge trimming,
fades, room tone, breath, level matching, peak guard) and mastering.py (the
streaming master chain: state continuity across blocks, gain sanity, no
NaNs, reverb tail). Pure numpy/scipy — no engine, no GPU, no model load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mastering  # noqa: E402
import prosody  # noqa: E402

SR = 44100


def _speech(seconds=1.0, sr=SR, level=0.25, lead_silence=0.15, tail_silence=0.35):
    """Fake speech: a couple of formant-ish tones with an amplitude envelope,
    wrapped in the ragged silence a real engine chunk carries."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    sig = (np.sin(2 * np.pi * 140 * t) * 0.6
           + np.sin(2 * np.pi * 620 * t) * 0.3
           + np.sin(2 * np.pi * 2400 * t) * 0.1)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    sig = (sig * env * level).astype(np.float32)
    pad_a = np.zeros(int(sr * lead_silence), dtype=np.float32)
    pad_b = np.zeros(int(sr * tail_silence), dtype=np.float32)
    return np.concatenate([pad_a, sig, pad_b])


# ------------------------------------------------------------------ pauses
def test_tail_punct():
    assert prosody.tail_punct("Hello there.") == "."
    assert prosody.tail_punct("Really?  ") == "?"
    assert prosody.tail_punct('"Stop," he said.') == "."
    assert prosody.tail_punct('He said "stop."') == "."   # looks past the quote
    assert prosody.tail_punct("Wait...") == "…"
    assert prosody.tail_punct("mid sentence") == "e"      # hard split, no mark
    assert prosody.tail_punct("") == ""


def test_pause_scales_with_punctuation():
    # jitter off: the ORDER is a property of the table, the wobble is not
    cfg = prosody.ProsodyConfig(base_gap_s=0.30, pause_jitter=0.0)
    comma = prosody.pause_seconds("a clause,", cfg, para_break=False)
    period = prosody.pause_seconds("a sentence.", cfg, para_break=False)
    question = prosody.pause_seconds("a question?", cfg, para_break=False)
    ellipsis = prosody.pause_seconds("trailing off…", cfg, para_break=False)
    assert comma < period < question < ellipsis, (comma, period, question, ellipsis)
    # anchored to the published ms ranges at the default 0.30 s slider
    assert 0.13 <= comma <= 0.21, comma
    assert 0.26 <= period <= 0.35, period
    assert 0.40 <= ellipsis <= 0.60, ellipsis


def test_pause_respects_slider_and_bounds():
    quiet = prosody.ProsodyConfig(base_gap_s=0.0)
    assert prosody.pause_seconds("x.", quiet, para_break=False) <= prosody.MIN_PAUSE_S
    loud = prosody.ProsodyConfig(base_gap_s=1.0)
    assert prosody.pause_seconds("x…", loud, para_break=False) <= prosody.MAX_PAUSE_S
    # disabled = old flat behaviour, exactly the slider value
    off = prosody.ProsodyConfig(enabled=False, base_gap_s=0.42, para_gap_s=0.9)
    assert prosody.pause_seconds("x,", off, para_break=False) == 0.42
    assert prosody.pause_seconds("x,", off, para_break=True) == 0.9


def test_breath_debt_lengthens_the_next_rest():
    cfg = prosody.ProsodyConfig(base_gap_s=0.30)
    fresh = prosody.pause_seconds("done.", cfg, para_break=False, chars_since_rest=0)
    tired = prosody.pause_seconds("done.", cfg, para_break=False, chars_since_rest=900)
    assert tired > fresh * 1.2, (fresh, tired)
    # and it saturates rather than growing forever
    exhausted = prosody.pause_seconds("done.", cfg, para_break=False,
                                      chars_since_rest=100000)
    assert exhausted <= fresh * (1.0 + prosody.BREATH_DEBT_MAX) * 1.01


def test_pauses_are_varied_but_deterministic():
    cfg = prosody.ProsodyConfig(base_gap_s=0.30)
    texts = [f"sentence number {i} here." for i in range(24)]
    a = [prosody.pause_seconds(t, cfg, para_break=False) for t in texts]
    b = [prosody.pause_seconds(t, cfg, para_break=False) for t in texts]
    assert a == b, "same script must render identically twice"
    assert len(set(round(x, 4) for x in a)) > 8, "pauses should not be metronomic"
    assert max(a) / min(a) < 1.35, "…but the wobble must stay subtle"


def test_rate_jitter_bounds():
    cfg = prosody.ProsodyConfig()
    vals = [prosody.rate_scale(f"a reasonably long sentence number {i}.", cfg)
            for i in range(50)]
    assert all(0.94 - 1e-9 <= v <= 1.06 + 1e-9 for v in vals), (min(vals), max(vals))
    assert (prosody.rate_scale("line 0", cfg)
            == prosody.rate_scale("line 0", cfg)), "must be deterministic"
    off = prosody.ProsodyConfig(enabled=False)
    assert prosody.rate_scale("anything", off) == 1.0


def test_rate_follows_discourse_position():
    cfg = prosody.ProsodyConfig(rate_jitter=0.0)
    body = "This is a sentence of perfectly ordinary length for a paragraph."
    mid = prosody.rate_scale(body, cfg)
    first = prosody.rate_scale(body, cfg, first_of_para=True)
    last = prosody.rate_scale(body, cfg, last_of_para=True)
    # a paragraph slows into its ending, and opens a touch under its body
    assert last < first < mid, (last, first, mid)
    # questions and very short lines run slower than the same text otherwise
    q = prosody.rate_scale("Is that what you meant by it, really?", cfg)
    s = prosody.rate_scale(body.replace("?", "."), cfg)
    assert q < s, (q, s)
    assert prosody.rate_scale("Right.", cfg) < mid


def test_clause_split_and_continuation():
    s = "one two three, four five six seven eight nine ten"
    cut = prosody.clause_split_point(s, 30)
    assert s[:cut] == "one two three,", s[:cut]
    # no clause mark available → falls back to a space, never mid-word
    plain = "aaaa bbbb cccc dddd eeee ffff gggg"
    cut2 = prosody.clause_split_point(plain, 20)
    assert plain[cut2] == " " or cut2 == 20, repr(plain[:cut2])
    assert prosody.continuation_text("half a thought") == "half a thought,"
    assert prosody.continuation_text("already punctuated,") == "already punctuated,"
    assert prosody.continuation_text("done.") == "done."


# --------------------------------------------------------------------- DSP
def test_trim_edges_removes_padding_but_keeps_speech():
    wav = _speech(seconds=1.0, lead_silence=0.30, tail_silence=0.50)
    out = prosody.trim_edges(wav, SR)
    assert out.size < wav.size
    assert out.size >= int(SR * 0.9), out.size / SR      # speech survives
    assert out.size <= int(SR * 1.2), out.size / SR      # padding is gone


def test_trim_never_eats_a_quiet_chunk():
    quiet = (_speech(seconds=0.8, level=0.004, lead_silence=0.0, tail_silence=0.0))
    out = prosody.trim_edges(quiet, SR)
    assert out.size >= quiet.size * 0.5


def test_fades_kill_the_seam_click():
    wav = np.ones(SR // 2, dtype=np.float32) * 0.5
    out = prosody.apply_fades(wav, SR)
    assert abs(float(out[0])) < 1e-3 and abs(float(out[-1])) < 1e-3
    assert abs(float(out[out.size // 2]) - 0.5) < 1e-6   # middle untouched


def test_gap_audio_is_silent_by_default():
    """Default gaps are pure silence: Lauro heard both room tone and breath as
    rustling, and that verdict outranks the theory behind them."""
    cfg = prosody.ProsodyConfig()
    gap = prosody.gap_audio(0.5, SR, cfg, floor=0.0008, speech_level=0.2, key="k")
    assert abs(gap.size - int(SR * 0.5)) <= 1
    assert float(np.abs(gap).max()) == 0.0


def test_gap_audio_room_tone_and_breath_when_asked():
    cfg = prosody.ProsodyConfig(room_tone=True, breath=True)
    gap = prosody.gap_audio(0.5, SR, cfg, floor=0.0008, speech_level=0.2, key="k")
    assert abs(gap.size - int(SR * 0.5)) <= 1
    assert float(np.sqrt(np.mean(gap ** 2))) < 0.2 * 0.12, "gap too loud"
    assert float(np.abs(gap).max()) < 0.2 * 0.5, "a gap must never rival the speech"
    assert float(np.sqrt(np.mean(gap ** 2))) > 0.0, "…but it is not digital zero"
    # short gaps get no breath, long ones do
    short = prosody.gap_audio(0.12, SR, cfg, floor=0.0008, speech_level=0.2, key="k")
    assert float(np.abs(short).max()) < float(np.abs(gap).max())


def test_gap_audio_disabled_is_pure_silence():
    off = prosody.ProsodyConfig(enabled=False)
    gap = prosody.gap_audio(0.3, SR, off, floor=0.001, speech_level=0.3, key="k")
    assert float(np.abs(gap).max()) == 0.0


def test_breath_is_band_limited_and_bounded():
    b = prosody.breath(SR, 0.22, 0.01, "key")
    assert b.size == int(SR * 0.22)
    assert float(np.abs(b).max()) < 0.06
    # energy should sit in the 400-2400 Hz region, not in the top octave
    spec = np.abs(np.fft.rfft(b))
    freqs = np.fft.rfftfreq(b.size, 1.0 / SR)
    band = spec[(freqs > 400) & (freqs < 2400)].sum()
    top = spec[freqs > 6000].sum()
    assert band > top * 2, (band, top)


def test_loudness_matcher_corrects_drift_slowly():
    lm = prosody.Loudness(max_step_db=1.5)
    base = _speech(level=0.25)
    lm.apply(base, SR)                       # first chunk sets the target
    hot = _speech(level=0.25 * 4)            # +12 dB outlier
    out = lm.apply(hot, SR)
    ratio = float(np.abs(out).max() / np.abs(hot).max())
    assert 0.8 < ratio < 1.0, ratio          # pulled down, but ≤1.5 dB at once
    assert ratio >= 10 ** (-1.5 / 20) - 1e-6


def test_peak_guard_never_clips_and_never_pumps_up():
    pg = prosody.PeakGuard(target_peak=0.85)
    loud = np.ones(100, dtype=np.float32) * 1.4
    out = pg.apply(loud)
    assert float(np.abs(out).max()) <= 0.851
    g = pg.gain
    quiet = np.ones(100, dtype=np.float32) * 0.1
    pg.apply(quiet)
    assert pg.gain == g, "gain must never go back up mid-stream"


def test_shape_chunk_is_a_noop_when_disabled():
    wav = _speech()
    off = prosody.ProsodyConfig(enabled=False)
    assert np.array_equal(prosody.shape_chunk(wav, SR, off), wav)


# --------------------------------------------------------------- mastering
def test_master_chain_is_stable_and_quiet_enough():
    m = mastering.Master(SR)
    wav = _speech(seconds=2.0, level=0.2)
    out = m.process(wav)
    assert out.shape == wav.shape
    assert np.all(np.isfinite(out))
    assert float(np.abs(out).max()) < 4.0


def test_master_block_split_matches_whole():
    """Streaming correctness: processing one buffer in pieces must give
    (nearly) the same samples as processing it in one go — otherwise every
    chunk boundary is a filter-state discontinuity, i.e. a click."""
    # auto-level off: it is a deliberately time-varying gain, so it would
    # mask exactly the discontinuity this test is looking for
    cfg = mastering.MasterConfig(level_max_db=0.0)
    wav = _speech(seconds=1.5, level=0.2)
    whole = mastering.Master(SR, cfg).process(wav)
    m = mastering.Master(SR, cfg)
    parts = [m.process(wav[i:i + 7000]) for i in range(0, wav.size, 7000)]
    pieced = np.concatenate(parts)
    assert pieced.shape == whole.shape
    err = float(np.abs(pieced - whole).max()) / max(float(np.abs(whole).max()), 1e-9)
    assert err < 0.02, err


def test_master_disabled_is_bit_identical():
    cfg = mastering.MasterConfig(enabled=False)
    m = mastering.Master(SR, cfg)
    wav = _speech()
    assert np.array_equal(m.process(wav), wav)
    assert m.flush().size == 0


def test_master_reverb_tail_exists_and_ends():
    m = mastering.Master(SR)
    m.process(_speech(seconds=0.5, level=0.2))
    tail = m.flush()
    assert 0 < tail.size <= int(SR * 0.06)
    assert float(np.abs(tail).max()) < 0.5
    assert m.flush().size == 0


def test_master_autolevel_is_clamped_and_rate_limited():
    cfg = mastering.MasterConfig(level_max_db=3.0, level_rate_db_s=0.9,
                                 reverb=False, comp=False)
    m = mastering.Master(SR, cfg)
    tiny = _speech(seconds=1.0, level=0.0005)     # far under target
    m.process(tiny)
    first_db = 20 * np.log10(m.level_gain)
    # ~1.5 s of audio in that block, so ≤1.35 dB of movement is allowed
    assert abs(first_db) <= cfg.level_rate_db_s * 1.6, "must not jump"
    for _ in range(40):
        m.process(tiny)
    assert m.level_gain <= 10 ** (3.0 / 20) + 1e-6
    assert m.level_gain >= 10 ** (-3.0 / 20) - 1e-6


def test_k_weighted_loudness_is_sane():
    """A -20 dBFS RMS pink-ish speech signal should measure in the -20 LUFS
    region, and a signal 6 dB louder should measure ~6 LU louder."""
    m1, m2 = mastering.Master(SR), mastering.Master(SR)
    quiet = _speech(seconds=3.0, level=0.1, lead_silence=0.0, tail_silence=0.0)
    m1.process(quiet)
    m2.process(quiet * 2.0)
    a, b = m1.measured_lufs(), m2.measured_lufs()
    assert a is not None and b is not None
    assert -40 < a < -5, a
    assert abs((b - a) - 6.02) < 0.3, (a, b)


def test_deesser_leaves_the_low_band_alone():
    """Split-band, not broadband ducking: an /s/-like burst must not move the
    level of the vowel underneath it."""
    cfg = mastering.MasterConfig(comp=False, reverb=False, deess=True,
                                 shelf_db=0.0, hp_hz=20.0, level_max_db=0.0)
    m = mastering.Master(SR, cfg)
    n = SR
    t = np.arange(n) / SR
    vowel = (np.sin(2 * np.pi * 220 * t) * 0.25).astype(np.float32)
    hiss = (np.random.default_rng(0).standard_normal(n) * 0.25).astype(np.float32)
    hiss[: n // 2] = 0.0                      # sibilance only in the 2nd half
    out = m.process(vowel + hiss)
    lo_in = float(np.sqrt(np.mean(vowel[n // 2:] ** 2)))
    sos = signal.butter(2, 1000, btype="lowpass", fs=SR, output="sos")
    lo_out = float(np.sqrt(np.mean(signal.sosfilt(sos, out)[n // 2:] ** 2)))
    assert abs(20 * np.log10(lo_out / lo_in)) < 1.0, 20 * np.log10(lo_out / lo_in)


def test_master_handles_silence_without_nan():
    m = mastering.Master(SR)
    out = m.process(np.zeros(SR // 2, dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert m.process(np.zeros(0, dtype=np.float32)).size == 0


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
