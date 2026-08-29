"""ECAPAFlow — Supertonic Voice Builder with real ECAPA smart-init cloning.

The experiment: LoudFlow's production voice-clone path (supertonic3_test/
clone.py, paid_parity* modes) uses ECAPA smart-init + 120-1500 gradient
iterations (8 min - 2 hrs). ECAPAFlow runs the smart-init step ALONE — no
gradient training — and targets under 1 second per clone.

Run: python app.py   (opens http://localhost:7873 after 2s)
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import json
import re
import socket
import struct
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ecapa as ecapa_mod  # noqa: E402
import mastering  # noqa: E402
import prosody  # noqa: E402
import voiceforge  # noqa: E402
import voices as voices_mod  # noqa: E402
from engine import engine  # noqa: E402
from normalizer_quality import normalize as normalize_text  # noqa: E402

# The 12 languages the Voice Builder UI exposes (Supertonic 3 supports 31;
# these are the ones in the design's language menu). "Auto" detects the
# language per paragraph from the text itself.
UI_LANGUAGES = [
    {"name": "Auto", "code": "auto"},
    {"name": "English", "code": "en"}, {"name": "Korean", "code": "ko"},
    {"name": "Spanish", "code": "es"}, {"name": "Portuguese", "code": "pt"},
    {"name": "French", "code": "fr"}, {"name": "Japanese", "code": "ja"},
    {"name": "German", "code": "de"}, {"name": "Arabic", "code": "ar"},
    {"name": "Italian", "code": "it"}, {"name": "Dutch", "code": "nl"},
    {"name": "Russian", "code": "ru"}, {"name": "Turkish", "code": "tr"},
]
SUPERTONIC_LANGS = {
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
}

# Highest ECAPA-steps value that still audibly improves quality — calibrated
# by benchmark_steps.py on real references (see README / benchmark results):
# the target embedding converges toward the dense gold standard up to ~24
# windows (at 16 the pixal blend still carried a wrong 29%-weight third voice;
# at 24 it matches gold; 24→32 shifts weights by <0.5%). Quality over speed
# per the brief — lower the slider to ≤12 for sub-second clones on CPU.
DEFAULT_ECAPA_STEPS = 24
MAX_ECAPA_STEPS = 32

# Hard cap on the script length (frontend textarea + backend truncation).
MAX_TEXT_CHARS = 100000

# Streaming synthesis: the response is a WAV whose PCM is sent chunk by chunk
# as the engine produces it, so playback starts while the rest still renders.
# The script is split into paragraphs at line breaks; a paragraph break
# inserts a longer pause than the regular sentence gap. Within a paragraph
# the text is chunked at sentence boundaries. CHUNK_CHARS matches the pip
# package's internal 300-char limit so one engine call = one model pass =
# one immediately streamable piece (measured 2026-07-02: per-call cost is
# dominated by a fixed overhead, so bigger chunks up to 300 are free runway).
CHUNK_CHARS = 300
PARAGRAPH_PAUSE_S = 0.6

# NO step ramp: every chunk renders at the full slider quality. A ramp
# (first chunks at 5/10 steps) was tried on 2026-07-02 — it bought a ~2s
# first-audio time but the reduced-step chunks were audibly worse, so it
# was removed on request. Quality over speed; streaming still hides most
# of the wait (TTFA ≈ 5s at 16 steps on the laptop, then gapless).
#
# What IS allowed to shrink for latency is the CHUNK SIZE, not the step
# count: in streaming mode the first engine call is a single sentence
# (minimum time-to-first-audio at full quality), then each next chunk is
# sized so its predicted inference time fits inside the playback runway
# already buffered on the client — chunks grow 1 sentence → 2 → 3 → …
# up to CHUNK_CHARS as the runway builds (RTF < 1 means every second of
# audio buys more than a second of buffer). The prediction is seeded from
# the persisted per-steps RTF/chars-per-second averages and refined after
# every chunk; the whole calculation is a few float ops (microseconds).
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…。！？])\s+|\n+")
_PARA_SPLIT_RE = re.compile(r"\s*\n\s*")

RUNWAY_SAFETY = 0.75     # spend at most 75% of the buffered runway per chunk
MIN_CHUNK_AUDIO_S = 1.0  # even when barely past the fixed-cost break-even
                         # point, request at least this much resulting audio
SMALL_AUDIO_S = 2.0      # chunks shorter than this are treated as a "fixed
                         # cost" calibration point, not a "variable rate" one
ERR_HIST_LEN = 20        # rolling window of predicted-vs-actual ratios used
                         # for the p90 safety margin (NetEQ-style)
SAFETY_PERCENTILE = 0.90
DEFAULT_FIXED_S = 1.0    # conservative seeds until the history has real data
DEFAULT_RATE_RTF = 0.15
DEFAULT_CPS = 14.0       # chars of normalized text per second of speech

# Quality-mode presets surfaced as buttons in the UI. Speed keeps the voice
# decent at the best RTF; Medium is the calibrated sweet spot (16 steps per
# Lauro's ear — 12 sounded worse); Quality ignores RTF entirely.
QUALITY_MODES = [
    {"key": "speed", "label": "Speed", "steps": 8},
    {"key": "medium", "label": "Medium", "steps": 16},
    {"key": "quality", "label": "Quality", "steps": 32},
    # steps=0 is the sentinel for "resolve at request time from the measured
    # RTF history" — the highest quality this machine still renders faster
    # than realtime (see _auto_steps).
    {"key": "auto", "label": "Auto", "steps": 0},
]

# Every voice gets a synthesized self-introduction (the small preview button
# on its tab): generated once with the voice's own style, cached to disk,
# regenerated when the voice is renamed (the name is part of the filename).
PREVIEWS_DIR = HERE / "data" / "cache" / "previews"
PREVIEW_TEXTS = [
    "Hi, I'm {name}. This is my voice — clear, natural, and ready to read your script.",
    "Hey there, {name} here. Give me any text and I'll bring it to life.",
    "Hello! My name is {name}, and this is exactly how I sound.",
]


class RtfHistory:
    """Persistent per-steps timing model: infer_time ≈ fixed_s + rate_rtf *
    audio_s, instead of a single infer/audio ratio.

    Measured 2026-07-05: on this class of hardware (DirectML iGPU) a
    diffusion model's per-call cost is dominated by a near-constant per-step
    GPU dispatch overhead that is almost INDEPENDENT of chunk size — a
    3-character chunk and a 160-character chunk cost almost the same. A
    single "rtf = infer/audio" ratio has no fixed term, so it badly
    under-predicts short/early chunks (worst exactly when TTFA and stall
    risk matter most). Splitting the model into a fixed term (per call) and
    a rate term (per second of resulting audio) fixes that.

    Updated once per CHUNK (not once per whole stream) so the estimate
    sharpens within a single run and doesn't blend a 1-sentence first chunk
    with 300-char later chunks into one polluted average.

    Also tracks a rolling window of (actual/predicted) ratios per steps
    value and exposes a high-percentile safety multiplier from it — mirrors
    WebRTC NetEQ's jitter-buffer sizing, which reads a p95 off a delay
    histogram rather than an EMA-of-the-mean, to hedge against a
    slower-than-usual chunk (thermal throttling, OS scheduling jitter)
    without permanently over-provisioning every chunk.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data = loaded
        except Exception:
            self._data = {}

    @staticmethod
    def _valid(row: Optional[dict]) -> bool:
        return bool(row) and "fixed_s" in row and "rate_rtf" in row and "cps" in row

    def _safety_mult(self, row: dict) -> float:
        errs = row.get("err_hist") or []
        if len(errs) < 4:
            return 1.15  # conservative hedge until there's enough data to read a percentile
        s = sorted(errs)
        idx = min(len(s) - 1, int(SAFETY_PERCENTILE * len(s)))
        return max(1.0, float(s[idx]))

    def estimates(self, steps: int) -> tuple[float, float, float, float]:
        """(fixed_s, rate_rtf, cps, safety_mult) for this steps value;
        nearest measured steps scaled linearly (diffusion compute is ~linear
        in steps, both the fixed dispatch cost and the marginal compute)
        when unmeasured."""
        with self._lock:
            row = self._data.get(str(int(steps)))
            if self._valid(row):
                return (float(row["fixed_s"]), float(row["rate_rtf"]),
                        float(row["cps"]), self._safety_mult(row))
            rows = [(int(k), v) for k, v in self._data.items()
                    if str(k).isdigit() and self._valid(v)]
            if rows:
                k, v = min(rows, key=lambda kv: abs(kv[0] - int(steps)))
                scale = int(steps) / max(k, 1)
                return (float(v["fixed_s"]) * scale, float(v["rate_rtf"]) * scale,
                        float(v["cps"]), self._safety_mult(v))
        return DEFAULT_FIXED_S, DEFAULT_RATE_RTF, DEFAULT_CPS, 1.15

    def update(self, steps: int, chars: int, audio_s: float, infer_s: float) -> None:
        if not (audio_s > 0 and infer_s > 0 and chars > 0):
            return
        with self._lock:
            key = str(int(steps))
            row = self._data.get(key)
            if not self._valid(row):
                # Reseed the timing model, but carry over any whole-stream
                # measurement already recorded for this step count — that is
                # what the Auto quality mode reads, and it must survive a
                # reseed of the per-chunk model.
                row = {"fixed_s": DEFAULT_FIXED_S, "rate_rtf": DEFAULT_RATE_RTF,
                       "cps": DEFAULT_CPS, "n": 0, "err_hist": [],
                       **{k: v for k, v in (row or {}).items() if k == "stream_rtf"}}
            predicted = row["fixed_s"] + row["rate_rtf"] * audio_s
            err_hist = list(row.get("err_hist") or [])
            if predicted > 0:
                err_hist = (err_hist + [infer_s / predicted])[-ERR_HIST_LEN:]
            a = 0.35  # more responsive than a once-per-stream EMA: chunks are frequent
            if audio_s < SMALL_AUDIO_S:
                implied_fixed = max(0.02, infer_s - row["rate_rtf"] * audio_s)
                row["fixed_s"] = round((1 - a) * float(row["fixed_s"]) + a * implied_fixed, 4)
            else:
                implied_rate = max(0.0, (infer_s - row["fixed_s"]) / audio_s)
                row["rate_rtf"] = round((1 - a) * float(row["rate_rtf"]) + a * implied_rate, 4)
            row["cps"] = round((1 - a) * float(row["cps"]) + a * (chars / audio_s), 2)
            row["n"] = int(row["n"]) + 1
            row["err_hist"] = err_hist
            self._data[key] = row
            self._save_locked()

    def _save_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._data), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            pass  # history is an optimization, never a failure

    def note_stream(self, steps: int, rtf: float, audio_s: float) -> None:
        """Record the END-TO-END RTF of a finished stream.

        The fixed+rate model above describes a single steady-state chunk and
        is what the chunk scheduler needs. It is NOT what "will this script
        outrun realtime" depends on: a real stream also pays the per-call
        fixed cost on every short early chunk, so its true RTF is always
        worse than the steady-state figure. The Auto quality mode picks a
        step count off THIS number instead — measured, whole-stream, honest.
        """
        if audio_s < 3.0 or rtf <= 0:
            return                      # too short to be representative
        with self._lock:
            row = self._data.setdefault(str(int(steps)), {})
            prev = row.get("stream_rtf")
            row["stream_rtf"] = round(rtf if prev is None else 0.6 * float(prev) + 0.4 * rtf, 4)
            self._save_locked()

    def stream_rtf(self, steps: int) -> Optional[float]:
        with self._lock:
            row = self._data.get(str(int(steps))) or {}
            v = row.get("stream_rtf")
            return float(v) if v else None

    def snapshot(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}


_rtf_history = RtfHistory(HERE / "data" / "cache" / "rtf_history.json")

# "Auto" quality: the highest step count this machine can still render FASTER
# than realtime, read off the measured history instead of guessed per machine.
# The laptop (DirectML iGPU) lands on 24-32; the RTX desktop will land at the
# top of the ladder; a slow CPU box degrades gracefully instead of stalling
# mid-playback. The target leaves headroom below RTF 1.0 for thermal jitter —
# a stream that finishes at exactly 1.0 is a stream that stutters.
AUTO_STEPS_LADDER = (48, 32, 24, 16, 12, 8)
AUTO_RTF_TARGET = 0.80
# A rung nobody has actually streamed yet is judged by the steady-state
# model, which is structurally optimistic (it amortises the per-call fixed
# cost over a full-size chunk; a real stream pays it on every short early
# chunk too). Measured 2026-07-27: the model said 0.80 for a rung that then
# streamed at 1.10 — hence this penalty on unproven rungs. Once a rung has
# been streamed for real, its own measurement is used and the penalty goes.
AUTO_UNMEASURED_PENALTY = 1.45


def _predicted_rtf(steps: int) -> float:
    """Steady-state RTF at this step count for a full-size chunk, from the
    persisted fixed+rate timing model."""
    fixed_s, rate_rtf, cps, _safety = _rtf_history.estimates(int(steps))
    chunk_audio_s = max(0.5, CHUNK_CHARS / max(cps, 1.0))
    return (fixed_s + rate_rtf * chunk_audio_s) / chunk_audio_s


def _expected_stream_rtf(steps: int) -> float:
    """Best available estimate of the END-TO-END RTF at this step count.

    Three sources, best first: this rung's own measurement; the nearest
    measured rung scaled linearly in steps (diffusion compute is ~linear in
    steps — the same assumption the chunk model makes) with a small hedge;
    and only if nothing has ever been streamed, the steady-state model with
    the larger penalty.
    """
    own = _rtf_history.stream_rtf(int(steps))
    if own is not None:
        return own
    measured = [(s, _rtf_history.stream_rtf(s)) for s in AUTO_STEPS_LADDER]
    measured = [(s, r) for s, r in measured if r is not None]
    if measured:
        near_s, near_r = min(measured, key=lambda kv: abs(kv[0] - steps))
        return near_r * (steps / max(near_s, 1)) * 1.15
    return _predicted_rtf(steps) * AUTO_UNMEASURED_PENALTY


def _auto_steps(target: float = AUTO_RTF_TARGET) -> int:
    """Highest ladder rung expected to stay under `target`. Every finished
    stream feeds its real RTF back (see note_stream), so the choice
    self-corrects from measurement instead of trusting a model that has never
    been checked against a whole stream."""
    for steps in AUTO_STEPS_LADDER:
        if _expected_stream_rtf(steps) <= target:
            return steps
    return AUTO_STEPS_LADDER[-1]


def _stream_allowance(runway_s: float, fixed_s: float, rate_rtf: float,
                      cps_est: float, cap: int) -> int:
    """Max chunk size (chars) whose predicted inference time fits inside the
    playback runway the client has already buffered, under a fixed+variable
    cost model. When even the per-call FIXED overhead doesn't fit the safe
    runway (already behind — e.g. Quality mode outrunning realtime), request
    the LARGEST allowed chunk instead of the smallest: the fixed cost is
    paid regardless of chunk size, so once behind, the only lever left is
    maximizing audio delivered per call instead of paying that fixed cost
    over and over for almost no audio each time."""
    safe_budget_s = max(0.0, RUNWAY_SAFETY * runway_s)
    if safe_budget_s <= fixed_s:
        return cap
    allowed_audio_s = max(MIN_CHUNK_AUDIO_S, (safe_budget_s - fixed_s) / max(rate_rtf, 0.02))
    chars = allowed_audio_s * cps_est
    return int(min(cap, max(1.0, chars)))


def _take_chunk(sents: list[tuple[int, str, str]], i: int,
                allowed_chars: int) -> int:
    """Merge sentences from index i (same paragraph only) while the joined
    text stays within allowed_chars. Always takes at least one sentence.
    Returns the exclusive end index."""
    pi = sents[i][0]
    total = len(sents[i][1])
    j = i + 1
    while (j < len(sents) and sents[j][0] == pi
           and total + 1 + len(sents[j][1]) <= allowed_chars):
        total += 1 + len(sents[j][1])
        j += 1
    return j


def _preview_path(voice: dict) -> Path:
    stem = voice.get("builtin_code") or voice["id"]
    slug = re.sub(r"[^\w]+", "-", voice["name"].lower()).strip("-")[:24] or "voice"
    return PREVIEWS_DIR / f"{stem}__{slug}.wav"


def _ensure_preview(voice: dict) -> Path:
    """Synthesized self-introduction in the voice's own style — built-ins,
    cloned and imported voices alike — generated once and cached to disk.
    The current name is baked into the filename, so a rename regenerates."""
    stem = voice.get("builtin_code") or voice["id"]
    p = _preview_path(voice)
    if p.exists():
        return p
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="engine still loading")
    style = _resolve_style(voice)
    if style is None:
        raise HTTPException(
            status_code=404, detail="no style for this voice yet — clone it first")
    text = PREVIEW_TEXTS[sum(stem.encode()) % len(PREVIEW_TEXTS)].format(
        name=voice["name"])
    wav, _d, _t = engine.synthesize(text, style, 16, 1.05, "en", 0.3)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    # Drop stale previews for this voice (old names, pre-rename cache files).
    for old in PREVIEWS_DIR.glob(f"{stem}__*.wav"):
        try:
            old.unlink(missing_ok=True)
        except Exception:
            pass
    legacy = PREVIEWS_DIR / f"{stem}.wav"
    try:
        legacy.unlink(missing_ok=True)
    except Exception:
        pass
    sf.write(str(p), _soft_limit(wav), engine.sample_rate)
    if not voice.get("duration_s"):
        voices_mod.update_voice(
            voice["id"], duration_s=round(len(wav) / engine.sample_rate, 2))
    return p


def _prewarm():
    # Build the built-in voice embedding table once engine + encoder are
    # up, so the FIRST user clone already hits the <1s path.
    while not engine.loaded:
        if not engine.loading and engine.status.startswith("Load failed"):
            return
        time.sleep(0.5)
    try:
        ecapa_mod.load_encoder()
        ecapa_mod.warmup_encoder()
        ecapa_mod.build_voice_embedding_cache(engine)
    except Exception as e:
        engine.log_error("prewarm", e)
    # Pre-generate the self-introduction preview for every voice that has a
    # style (built-ins, cloned, imported) so the first preview click is
    # instant. One-time cost (~30s background); cached as WAV afterwards.
    for v in voices_mod.list_voices():
        try:
            if v.get("builtin_code") or v.get("cloned"):
                _ensure_preview(v)
        except Exception as e:
            engine.log_error("previews", e)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    voices_mod.ensure_builtin_voices()
    engine.start_background_load()
    ecapa_mod.start_background_load()
    threading.Thread(target=_prewarm, name="ecapaflow-prewarm", daemon=True).start()
    yield


app = FastAPI(title="ECAPAFlow", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


# ---- helpers ----
def _soft_limit(wav: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    if wav.size == 0:
        return wav
    peak = float(np.abs(wav).max())
    if peak > target_peak:
        return wav * (target_peak / peak)
    return wav


def _wav_header_streaming(sr: int) -> bytes:
    """44-byte PCM16-mono WAV header with maxed-out sizes — the total length
    is unknown while streaming. The frontend strips these 44 bytes anyway."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
        b"data", 0xFFFFFFFF - 44,
    )


def _pcm16(wav: np.ndarray) -> bytes:
    wav = _soft_limit(wav, target_peak=0.85)
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _split_sentences(text: str, limit: int = CHUNK_CHARS,
                     natural: bool = True) -> list[str]:
    """Split text into sentences, hard-splitting run-ons longer than limit.
    These are the atoms the stream loop merges into engine chunks
    (adaptively when streaming, up to the cap otherwise).

    A run-on is cut at a CLAUSE boundary (comma/semicolon/colon/dash) when
    one is available inside the window, not at whatever space happens to sit
    at position `limit` — the model shapes a phrase far better than it
    shapes half a noun phrase. The left half then gets a trailing comma so
    it is rendered with a continuation contour instead of a sentence-final
    fall on a sentence that has not actually ended (see prosody.py).
    """
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    pieces: list[str] = []
    for s in parts:
        # The continuation comma costs one character, so the natural path
        # cuts one earlier — a piece must never exceed the engine's own
        # internal chunk limit or one engine call stops being one model pass.
        eff = limit - 1 if natural else limit
        while len(s) > limit:
            if natural:
                cut = prosody.clause_split_point(s, eff)
            else:
                cut = s.rfind(" ", limit // 2, limit)
                if cut == -1:
                    cut = limit
            left = s[:cut].strip()
            if natural:
                left = prosody.continuation_text(left)
            pieces.append(left)
            s = s[cut:].strip()
        if s:
            pieces.append(s)
    return pieces


def _split_paragraphs(text: str) -> list[str]:
    """Paragraphs are separated by line breaks in the script textarea. Split
    BEFORE normalization — the normalizer collapses whitespace runs."""
    return [p for p in (s.strip() for s in _PARA_SPLIT_RE.split(text)) if p]


# Stopword votes for the Latin-script languages of the UI menu; non-Latin
# scripts (ko/ja/ar/ru/el) are decided by Unicode ranges before voting.
_LANG_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit",
           "für", "auf", "dem", "den", "ich", "sie", "auch", "aber", "wir",
           "im", "von", "des", "sich", "als", "wurde", "bis", "schon",
           "noch", "nur", "wenn", "dann", "alles", "keine", "werden",
           "kann", "nach", "über", "sehr", "mehr", "zum", "zur", "seit",
           # conversational/short-text words — a stopword-free casual
           # greeting ("Hallo, wie geht es dir heute?") scored zero German
           # hits before this, misdetecting as "na" (unknown) and defeating
           # Auto Voice for exactly the kind of text it's meant for.
           "wie", "was", "wer", "wo", "warum", "heute", "gut", "bitte",
           "danke", "hallo", "geht", "guten"},
    "en": {"the", "and", "is", "are", "was", "of", "to", "in", "that", "it",
           "for", "with", "you", "not", "this", "on", "as", "at", "be",
           "have", "from", "were", "after", "before", "they", "their",
           "there", "his", "her", "had", "but", "all", "when", "then",
           "more", "about", "into", "just", "right", "one"},
    "es": {"el", "los", "las", "es", "y", "que", "en", "un", "una", "por",
           "con", "para", "no", "se", "su", "al", "lo", "como", "más"},
    "fr": {"le", "les", "et", "est", "des", "que", "dans", "un", "une",
           "pour", "avec", "ne", "pas", "ce", "il", "au", "sur", "qui"},
    "it": {"il", "le", "è", "di", "che", "un", "una", "per", "con", "non",
           "si", "sono", "del", "della", "da", "come", "più"},
    "pt": {"os", "as", "é", "de", "que", "em", "um", "uma", "para", "com",
           "não", "se", "do", "da", "no", "na", "por", "mais"},
    "nl": {"de", "het", "een", "en", "is", "van", "dat", "niet", "met",
           "voor", "op", "ik", "je", "zijn", "aan", "ook", "maar", "dit"},
    "tr": {"ve", "bir", "bu", "için", "ile", "değil", "ama", "çok", "daha",
           "olarak", "gibi", "ne", "var", "yok", "bu", "da"},
}


def _detect_lang(text: str) -> str:
    """Best-effort language detection for the Auto mode. Non-Latin scripts
    decide immediately; Latin-script languages vote via stopwords. Returns
    'na' (the engine's own auto mode) when unsure."""
    for ch in text:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7AF:
            return "ko"
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF:
            return "ja"
        if 0x0600 <= o <= 0x06FF:
            return "ar"
        if 0x0400 <= o <= 0x04FF:
            return "ru"
        if 0x0370 <= o <= 0x03FF:
            return "el"
    words = re.findall(r"[a-zà-öø-ÿčćšžğışę']+", text.lower())
    if not words:
        return "na"
    scores = sorted(
        ((sum(1 for w in words if w in sw), code)
         for code, sw in _LANG_STOPWORDS.items()),
        reverse=True)
    best_hits, best_code = scores[0]
    if best_hits < 2 or best_hits == scores[1][0]:
        return "na"
    return best_code


# Per-synthesis stats keyed by synthesis id; the frontend fetches
# /api/synthesize/stats/{sid} after the response body is read.
_SYNTH_STATS: OrderedDict[str, dict] = OrderedDict()


def _remember_synth_stats(sid: str, stats: dict) -> None:
    _SYNTH_STATS[sid] = stats
    while len(_SYNTH_STATS) > 32:
        _SYNTH_STATS.popitem(last=False)


def _read_audio_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        import librosa
        return float(librosa.get_duration(path=str(path)))


def _voice_status(v: dict) -> str:
    """One word for what the UI must show, and for whether this voice may be
    used at all.

    * ready     — a style exists on disk; synthesis is allowed
    * training  — a clone job is running; synthesis is refused
    * new       — uploaded/recorded but never cloned; synthesis is refused
    * failed    — the last clone failed; synthesis is refused
    * no-ref    — imported as a 1:1 style with no reference audio; ready

    The rule Lauro asked for is enforced here and nowhere else: a voice you
    can hear is a voice that finished cloning.
    """
    if v.get("builtin_code"):
        return "ready"
    job = voiceforge.get_job(v["id"])
    if job is not None and job.progress.phase not in ("done", "failed", "cancelled"):
        return "training"
    sp = voices_mod.style_path(v)
    if v.get("cloned") and sp and sp.exists():
        return "ready"
    st = v.get("clone_status")
    if st in ("failed", "training"):
        return "failed" if st == "failed" else "new"
    if not v.get("ref_filename"):
        return "no-ref"
    return "new"


def _clone_voice_now(voice: dict, ecapa_steps: int) -> dict:
    """Run smart-init on a registry voice, persist the style, update the
    record. Returns {voice, clone} (clone = smart-init result summary).
    Kicks off async verification (synthesize sample + ECAPA-compare)."""
    if voice.get("builtin_code"):
        raise HTTPException(status_code=400, detail="built-in voice — nothing to clone")
    ref = voices_mod.ref_path(voice)
    if not ref.exists():
        raise HTTPException(status_code=404, detail=f"reference audio missing: {ref.name}")

    if not voice.get("ref_filename") or not ref.is_file():
        raise HTTPException(
            status_code=400,
            detail="this voice has no reference audio to clone from (imported 1:1 style)")
    result = ecapa_mod.smart_init_clone(engine, str(ref), ecapa_steps=ecapa_steps)

    style_fn = f"style_{voice['id']}.json"
    ecapa_mod.save_style_json(
        str(voices_mod.VOICES_DIR / style_fn),
        result["ttl"], result["dp"],
        metadata={
            "app": "ECAPAFlow",
            "method": "ecapa_smart_init",
            "gradient_iterations": 0,
            "ecapa_steps_requested": result["requested_steps"],
            "ecapa_steps_effective": result["effective_steps"],
            "ranking": result["ranking"],
            "est_sim_pct": result["est_sim_pct"],
            "timings_ms": result["timings_ms"],
            "source_file": voice["ref_filename"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    clone_info = {
        "est_sim_pct": result["est_sim_pct"],
        "ranking": result["ranking"][:5],
        "ecapa_steps": result["effective_steps"],
        "timings_ms": result["timings_ms"],
        "verified_sim_pct": None,
    }
    updated = voices_mod.update_voice(
        voice["id"], cloned=True, style_filename=style_fn, clone_info=clone_info,
    )

    # Refresh the engine's style cache for this voice id.
    cache_key = f"ecapaflow:{voice['id']}"
    engine.voice_cache.pop(cache_key, None)

    # Async verification — one synthesis + one embed; updates the registry.
    ttl, dp, target = result["ttl"], result["dp"], result["target_emb"]

    def _verify():
        try:
            v = ecapa_mod.verify_clone(engine, ttl, dp, target)
            cur = voices_mod.get_voice(voice["id"])
            if cur and cur.get("clone_info"):
                ci = cur["clone_info"]
                ci["verified_sim_pct"] = v["verified_sim_pct"]
                ci["verify_ms"] = v["verify_ms"]
                voices_mod.update_voice(voice["id"], clone_info=ci)
                ecapa_mod.log(
                    f"verify [{cur['name']}]: achieved ECAPA sim {v['verified_sim_pct']:.1f}%"
                )
        except Exception as e:
            engine.log_error("verify_clone", e)

    threading.Thread(target=_verify, name="ecapaflow-verify", daemon=True).start()

    return {
        "voice": updated,
        "clone": {
            "est_sim_pct": result["est_sim_pct"],
            "ranking": result["ranking"],
            "effective_steps": result["effective_steps"],
            "timings_ms": result["timings_ms"],
        },
    }


def _resolve_style(voice: dict):
    """Load the style for a registry voice: built-ins straight from the
    engine (cached there), cloned voices from their style JSON."""
    if voice.get("builtin_code"):
        return engine.get_voice(voice["builtin_code"])
    sp = voices_mod.style_path(voice)
    if not sp or not sp.exists():
        return None
    cache_key = f"ecapaflow:{voice['id']}"
    cached = engine.voice_cache.get(cache_key)
    if cached is not None:
        return cached
    return engine.load_voice_from_path(str(sp), cache_key)


# ---- routes ----
@app.get("/", response_class=HTMLResponse)
def index():
    html = (HERE / "static" / "index.html").read_text(encoding="utf-8")
    # Cache-bust the script tag with app.js's mtime: after an update, a
    # plain reload gets the new frontend instead of a stale browser-cached
    # one (bit us during 2026-07-05 verification — old JS survived reloads).
    try:
        v = int((HERE / "static" / "app.js").stat().st_mtime)
        html = html.replace('src="/static/app.js"', f'src="/static/app.js?v={v}"')
    except Exception:
        pass
    return HTMLResponse(html)


def _display_rtf(row: dict) -> float:
    """Single effective-RTF number for the UI (mode-button tooltips): what
    the fixed+variable model predicts for a full-size, steady-state chunk —
    representative of sustained throughput, unlike the tiny first chunk."""
    chunk_audio_s = max(0.5, CHUNK_CHARS / max(float(row.get("cps", DEFAULT_CPS)), 1.0))
    infer_s = float(row.get("fixed_s", DEFAULT_FIXED_S)) + \
        float(row.get("rate_rtf", DEFAULT_RATE_RTF)) * chunk_audio_s
    return round(infer_s / chunk_audio_s, 4)


@app.get("/api/status")
def api_status():
    s = engine.status_dict()
    s["ecapa"] = ecapa_mod.status()
    s["languages"] = UI_LANGUAGES
    s["default_ecapa_steps"] = DEFAULT_ECAPA_STEPS
    s["max_ecapa_steps"] = MAX_ECAPA_STEPS
    s["max_text_chars"] = MAX_TEXT_CHARS
    s["modes"] = QUALITY_MODES
    hist = _rtf_history.snapshot()
    for row in hist.values():
        if RtfHistory._valid(row):
            row["rtf"] = _display_rtf(row)
    s["rtf_history"] = hist
    return s


@app.get("/api/voices")
def api_voices():
    """Every voice, each carrying the one field the UI needs to decide
    whether it may be selected: `status` (see _voice_status), plus the live
    clone progress when one is running."""
    out = []
    for v in voices_mod.list_voices():
        v = dict(v)
        v["status"] = _voice_status(v)
        job = voiceforge.get_job(v["id"])
        if job is not None and job.progress.phase not in ("done",):
            v["clone_job"] = job.as_dict()
        out.append(v)
    return {"voices": out}


@app.post("/api/voices/upload")
async def api_voices_upload(
    file: UploadFile = File(...),
    vtype: str = Form("uploaded"),
    name: Optional[str] = Form(None),
):
    if vtype not in ("uploaded", "recorded"):
        vtype = "uploaded"
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    orig_stem = Path(file.filename or "voice").stem or "voice"
    ext = (Path(file.filename or "voice.wav").suffix or ".wav").lower()
    fname = f"ref_{uuid.uuid4().hex[:10]}{ext}"
    dest = voices_mod.REFS_DIR / fname
    dest.write_bytes(raw)

    stats = await run_in_threadpool(ecapa_mod.analyze_ref_audio, str(dest))
    if not stats.get("ok"):
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=stats.get("reason", "reference rejected"))

    if name and name.strip():
        display = name.strip()[:64]
    elif vtype == "recorded":
        existing = [v for v in voices_mod.list_voices() if v["type"] == "recorded"]
        display = "Recorded Voice" if not existing else f"Recorded Voice ({len(existing) + 1})"
    else:
        display = orig_stem[:64]

    voice = voices_mod.add_voice(display, vtype, fname, stats.get("duration_s", 0.0), stats)
    return {"voice": voice, "warning": stats.get("warning", "")}


@app.post("/api/voices/import_style")
async def api_voices_import_style(
    style: UploadFile = File(...),
    ref_audio: Optional[UploadFile] = File(None),
    name: Optional[str] = Form(None),
):
    """Import a ready-made Supertonic voice-style JSON (e.g. a paid_parity
    clone trained on Colab, or a purchased Voice Builder voice) as a voice
    tab. Optional reference audio enables the Source Voice button and the
    async verified-similarity check."""
    import json as _json

    raw = await style.read()
    try:
        data = _json.loads(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="not a valid JSON file")
    for key in ("style_ttl", "style_dp"):
        if key not in data or "data" not in data.get(key, {}) or "dims" not in data.get(key, {}):
            raise HTTPException(
                status_code=400,
                detail=f"not a Supertonic voice-style JSON (missing {key}.data/dims)")

    meta = data.get("metadata") or {}
    display = (name or "").strip()[:64] or Path(style.filename or "voice").stem[:64] or "Imported Voice"

    # Optional reference audio (for Source playback + verification)
    ref_fn = None
    ref_dur = 0.0
    ref_stats: dict = {}
    if ref_audio is not None and ref_audio.filename:
        rdata = await ref_audio.read()
        ext = (Path(ref_audio.filename).suffix or ".wav").lower()
        ref_fn = f"ref_{uuid.uuid4().hex[:10]}{ext}"
        (voices_mod.REFS_DIR / ref_fn).write_bytes(rdata)
        ref_stats = await run_in_threadpool(
            ecapa_mod.analyze_ref_audio, str(voices_mod.REFS_DIR / ref_fn))
        if ref_stats.get("ok"):
            ref_dur = ref_stats.get("duration_s", 0.0)
        else:
            # keep the import, just drop the unusable reference
            try:
                (voices_mod.REFS_DIR / ref_fn).unlink(missing_ok=True)
            except Exception:
                pass
            ref_fn = None
            ref_stats = {}

    voice = voices_mod.add_voice(display, "imported", ref_fn or "", ref_dur, ref_stats)
    style_fn = f"style_{voice['id']}.json"
    (voices_mod.VOICES_DIR / style_fn).write_bytes(raw)

    clone_info = {
        "imported": True,
        "est_sim_pct": None,
        "trained_sim_pct": meta.get("final_sim_pct"),
        "mode": meta.get("mode"),
        "iterations": meta.get("num_steps_run") or meta.get("num_steps_planned"),
        "verified_sim_pct": None,
    }
    voice = voices_mod.update_voice(
        voice["id"], cloned=True, style_filename=style_fn, clone_info=clone_info)

    # Async verification if we have reference audio to compare against.
    if ref_fn:
        vid = voice["id"]

        def _verify_import():
            try:
                import numpy as _np
                ttl = _np.array(data["style_ttl"]["data"], dtype=_np.float32).reshape(
                    *data["style_ttl"]["dims"])
                dp = _np.array(data["style_dp"]["data"], dtype=_np.float32).reshape(
                    *data["style_dp"]["dims"])
                target, _eff = ecapa_mod.embed_reference(
                    str(voices_mod.REFS_DIR / ref_fn), steps=DEFAULT_ECAPA_STEPS)
                v = ecapa_mod.verify_clone(engine, ttl, dp, target)
                cur = voices_mod.get_voice(vid)
                if cur and cur.get("clone_info"):
                    ci = cur["clone_info"]
                    ci["verified_sim_pct"] = v["verified_sim_pct"]
                    voices_mod.update_voice(vid, clone_info=ci)
                    ecapa_mod.log(f"verify import [{cur['name']}]: "
                                  f"achieved ECAPA sim {v['verified_sim_pct']:.1f}%")
            except Exception as e:
                engine.log_error("verify_import", e)

        threading.Thread(target=_verify_import, name="ecapaflow-verify-import",
                         daemon=True).start()

    return {"voice": voice}


@app.patch("/api/voices/{voice_id}")
async def api_voices_rename(
    voice_id: str,
    name: Optional[str] = Form(None),
    lang: Optional[str] = Form(None),
):
    """Rename and/or set a voice's language tag. The tag is the app's OWN
    organizational label, not a claim that Supertonic trained this voice on
    that language — Supertonic's 10 built-in voices are architecturally
    decoupled timbres usable with any of its 31 languages (confirmed via
    the official docs: "no per-language fine-tuning", "voices are more
    consistent across languages"). The tag only drives Auto Voice mode
    (pick the voice a user tagged for a paragraph's language, if any).
    """
    fields: dict = {}
    if name is not None:
        fields["name"] = name.strip()[:64] or "Voice"
    if lang is not None:
        lang = lang.strip().lower()
        fields["lang"] = lang if lang and lang != "any" else None
    if not fields:
        raise HTTPException(status_code=400, detail="nothing to update")
    v = voices_mod.update_voice(voice_id, **fields)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")
    return {"voice": v}


@app.delete("/api/voices/{voice_id}")
def api_voices_delete(voice_id: str):
    if not voices_mod.remove_voice(voice_id):
        raise HTTPException(status_code=404, detail="voice not found")
    engine.voice_cache.pop(f"ecapaflow:{voice_id}", None)
    return {"ok": True}


@app.get("/api/voices/{voice_id}/audio")
def api_voices_audio(voice_id: str):
    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")
    if not v.get("ref_filename"):
        # No reference clip (built-in or style-only import): serve the
        # synthesized self-introduction instead.
        p = _ensure_preview(v)
        return FileResponse(str(p), media_type="audio/wav")
    p = voices_mod.ref_path(v)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="reference audio missing")
    media = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
             ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".webm": "audio/webm"}
    return FileResponse(str(p), media_type=media.get(p.suffix.lower(), "application/octet-stream"))


@app.get("/api/voices/{voice_id}/preview")
async def api_voices_preview(voice_id: str):
    """Self-introduction preview for the voice tab's play button: synthesized
    in the voice's own style (cached). Voices without a style yet (uploaded
    but never cloned) fall back to their raw reference clip."""
    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")
    try:
        p = await run_in_threadpool(_ensure_preview, v)
        return FileResponse(str(p), media_type="audio/wav")
    except HTTPException:
        if v.get("ref_filename"):
            rp = voices_mod.ref_path(v)
            if rp.is_file():
                media = {".wav": "audio/wav", ".mp3": "audio/mpeg",
                         ".flac": "audio/flac", ".ogg": "audio/ogg",
                         ".m4a": "audio/mp4", ".webm": "audio/webm"}
                return FileResponse(
                    str(rp),
                    media_type=media.get(rp.suffix.lower(), "application/octet-stream"))
        raise


@app.get("/api/voices/{voice_id}/style")
def api_voices_style(voice_id: str):
    """Download the voice's Supertonic style JSON — cloned, imported and
    built-in voices alike. The file round-trips through Upload Voice /
    import_style, so styles can be backed up or shared between machines."""
    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")
    if v.get("builtin_code"):
        if not engine.loaded:
            raise HTTPException(status_code=503, detail="engine still loading")
        p = Path(engine.voice_styles_dir) / f"{v['builtin_code']}.json"
    else:
        p = voices_mod.style_path(v)
    if not p or not p.exists():
        raise HTTPException(
            status_code=404, detail="no style for this voice yet — clone it first")
    safe = re.sub(r"[^\w\-. ]+", "", v["name"]).strip() or "voice"
    return FileResponse(str(p), media_type="application/json",
                        filename=f"{safe}_style.json")


def _finish_forge(job: voiceforge.ForgeJob) -> None:
    """Persist a finished slow clone and mark the voice ready. Runs on the
    job's own thread; the endpoint never waits for it."""
    v = voices_mod.get_voice(job.voice_id)
    if v is None or job.result is None:
        if v is not None and job.progress.phase in ("failed", "cancelled"):
            voices_mod.update_voice(job.voice_id, clone_status=job.progress.phase)
        return
    res = job.result
    style_fn = f"style_{v['id']}.json"
    ecapa_mod.save_style_json(
        str(voices_mod.VOICES_DIR / style_fn), res["ttl"], res["dp"],
        metadata={
            "app": "ECAPAFlow",
            "method": "ecapa_smart_init+es_search",
            "search_evals": res["evals"],
            "search_generations": res["generations"],
            "search_seconds": res["elapsed_s"],
            "depth": job.depth,
            "baseline_sim_pct": res["baseline_sim_pct"],
            "final_sim_pct": res["final_sim_pct"],
            "accepted_search_result": res["accepted_search_result"],
            "ranking": res["ranking"],
            "source_file": v.get("ref_filename", ""),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    voices_mod.update_voice(
        v["id"], cloned=True, clone_status="ready", style_filename=style_fn,
        clone_info={
            "est_sim_pct": res["final_sim_pct"],
            # The held-out number IS the verified number here — it was
            # measured on a sentence the search never saw, which is a
            # stricter test than the old post-hoc verification.
            "verified_sim_pct": res["final_sim_pct"],
            "baseline_sim_pct": res["baseline_sim_pct"],
            "improved_points": round(res["final_sim_pct"] - res["baseline_sim_pct"], 2),
            "ranking": res["ranking"][:5],
            "ecapa_steps": res.get("effective_steps"),
            "depth": job.depth,
            "search_evals": res["evals"],
            "search_seconds": res["elapsed_s"],
        },
    )
    engine.voice_cache.pop(f"ecapaflow:{v['id']}", None)
    print(f"[clone {v['name']}] {job.depth}: {res['evals']} candidates in "
          f"{res['elapsed_s']:.0f}s -> {res['final_sim_pct']:.1f}% "
          f"(instant clone was {res['baseline_sim_pct']:.1f}%)", flush=True)


@app.post("/api/voices/{voice_id}/clone")
async def api_voices_clone(voice_id: str,
                           ecapa_steps: int = Form(DEFAULT_ECAPA_STEPS),
                           depth: str = Form("standard")):
    """Start a slow, thorough clone. Returns immediately with a job status;
    the voice stays unusable until it finishes."""
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Supertonic 3 engine still loading")
    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")
    if v.get("builtin_code"):
        raise HTTPException(status_code=400, detail="built-in voice — nothing to clone")
    ref = voices_mod.ref_path(v)
    if not v.get("ref_filename") or not ref.is_file():
        raise HTTPException(
            status_code=400,
            detail="this voice has no reference audio to clone from (imported 1:1 style)")
    if depth not in voiceforge.DEPTHS:
        depth = "standard"
    steps = max(1, min(MAX_ECAPA_STEPS, int(ecapa_steps)))
    voices_mod.update_voice(voice_id, clone_status="training", cloned=False)
    job = voiceforge.start_job(engine, v, str(ref), depth, steps, _finish_forge)
    return {"job": job.as_dict(), "voice": voices_mod.get_voice(voice_id)}


@app.get("/api/voices/{voice_id}/clone/status")
def api_clone_status(voice_id: str):
    job = voiceforge.get_job(voice_id)
    v = voices_mod.get_voice(voice_id)
    if v is None:
        raise HTTPException(status_code=404, detail="voice not found")
    return {"job": job.as_dict() if job else None,
            "voice_status": _voice_status(v),
            "voice": v}


@app.post("/api/voices/{voice_id}/clone/cancel")
def api_clone_cancel(voice_id: str):
    job = voiceforge.get_job(voice_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no clone running for this voice")
    job.cancel()
    voices_mod.update_voice(voice_id, clone_status="new")
    return {"cancelled": True, "job": job.as_dict()}


@app.get("/api/clone/depths")
def api_clone_depths():
    """The clone-depth choices, with their real time budgets."""
    return {"depths": [
        {"key": k, "label": lbl, "seconds": secs, "description": desc}
        for k, (lbl, secs, desc) in voiceforge.DEPTHS.items()
    ]}


def _voice_for_lang(target_lang: str, exclude_id: str) -> Optional[dict]:
    """Auto Voice: find a registry voice the user tagged with this language
    (an app-assigned label, not a Supertonic claim — see api_voices_rename),
    skipping the already-selected voice and anything not actually ready to
    synthesize with (never cloned, style missing on disk)."""
    for cand in voices_mod.list_voices():
        if cand.get("lang") != target_lang or cand["id"] == exclude_id:
            continue
        if cand.get("builtin_code"):
            return cand
        sp = voices_mod.style_path(cand)
        if cand.get("cloned") and sp and sp.exists():
            return cand
    return None


@app.post("/api/synthesize")
async def api_synthesize(
    voice_id: str = Form(...),
    text: str = Form(...),
    lang: str = Form("en"),
    steps: int = Form(16),
    speed: float = Form(1.05),
    silence: float = Form(0.30),
    ecapa_steps: int = Form(DEFAULT_ECAPA_STEPS),
    streaming: int = Form(1),
    auto_voice: int = Form(0),
    natural: int = Form(1),
    master: int = Form(1),
    norm_mode: str = Form("fast"),
):
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Supertonic 3 engine still loading")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text required")
    if lang != "auto" and lang not in SUPERTONIC_LANGS:
        lang = "na"
    # Text normalization tier. "fast" = the regex engine (German/English,
    # microseconds). "quality" = the same rules plus a num2words pass that
    # renders numbers, dates, times, currency, ordinals, ranges, fractions
    # and units in the ACTUAL language of the text, across nine languages.
    # Both are deterministic; neither is a language model.
    norm_mode = "quality" if str(norm_mode).lower().startswith("q") else "fast"

    # steps <= 0 = the Auto quality mode: resolve to the highest step count
    # this machine still renders faster than realtime (measured, not guessed).
    auto_quality = int(steps) <= 0
    if auto_quality:
        steps = _auto_steps()

    # The natural-speech layer: punctuation-scaled pauses, breaths, room
    # tone, seam de-clicking, level matching, per-sentence rate variation.
    # Pure numpy on audio the engine already produced — zero inference cost,
    # and it lowers RTF because its pauses are extra output seconds that
    # took no time to make.
    pcfg = prosody.ProsodyConfig(
        enabled=bool(int(natural)),
        base_gap_s=max(0.0, float(silence)),
        para_gap_s=PARAGRAPH_PAUSE_S,
    )
    # Master chain (high-pass, de-ess, presence shelf, soft compression,
    # tiny-room early reflections, auto level) — the polish LoudFlow gets
    # from its Pedalboard chain, rebuilt in scipy so pauses run through it
    # too and the reverb rings into them instead of stopping at the seam.
    mcfg = mastering.MasterConfig(enabled=bool(int(master)))

    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")

    # A voice may only be used once it is FULLY cloned (Lauro, 2026-07-27).
    # Auto-cloning on first use is gone: it silently shipped a one-second
    # smart-init blend under the same name as a finished clone, so the user
    # had no way to tell which one they were listening to. Now the refusal
    # says exactly what state the voice is in and what to do about it.
    auto_cloned = False
    clone_ms = 0.0
    vstatus = _voice_status(v)
    if vstatus != "ready":
        job = voiceforge.get_job(v["id"])
        if vstatus == "training" and job is not None:
            p = job.progress
            raise HTTPException(
                status_code=409,
                detail=(f"“{v['name']}” is still cloning — {p.message} "
                        f"({p.evals} candidates tried, "
                        f"{max(0, int(p.eta_s // 60))} min left). "
                        f"It becomes usable the moment it finishes."))
        if vstatus == "failed":
            raise HTTPException(
                status_code=409,
                detail=f"Cloning “{v['name']}” failed. Start it again from the voice tab.")
        raise HTTPException(
            status_code=409,
            detail=(f"“{v['name']}” has not been cloned yet. Press Clone on its "
                    f"tab and pick how thorough it should be — the voice "
                    f"unlocks when the clone finishes."))

    style = await run_in_threadpool(_resolve_style, v)
    if style is None:
        raise HTTPException(status_code=500, detail="cloned style missing on disk")

    # Paragraphs first (line breaks in the textarea), THEN LoudFlow text
    # normalization per paragraph (regex DE/EN safety net — mT5 path was
    # deliberately dropped in LoudFlow, see normalizer.py docstring). The
    # normalizer collapses whitespace, so it must run after the paragraph split.
    # In Auto mode the language is detected per paragraph, so the normalizer
    # (years, currency, units) and the engine both get the right language.
    paragraphs = _split_paragraphs(text.strip()[:MAX_TEXT_CHARS])
    para_langs = [_detect_lang(p) if lang == "auto" else lang for p in paragraphs]
    para_list: list[tuple[list[str], str]] = []
    for p, pl in zip(paragraphs, para_langs):
        # Korean is chunked at 120 chars inside the pip package — match it
        # so one engine call stays one model pass for ko too.
        limit = 120 if pl == "ko" else CHUNK_CHARS
        sents = _split_sentences(
            normalize_text(p, lang="en" if pl == "na" else pl, mode=norm_mode),
            limit, natural=pcfg.enabled)
        if sents:
            para_list.append((sents, pl))
    if not para_list:
        raise HTTPException(status_code=400, detail="text is empty after normalization")
    norm_text = "\n".join(" ".join(sents) for sents, _ in para_list)
    lang_label = lang if lang != "auto" else (
        "auto:" + ",".join(dict.fromkeys(pl for _, pl in para_list)))

    # Auto Voice: for each language actually appearing in this script, use
    # whichever registry voice the user tagged for it (if any and if it's
    # ready to synthesize with), instead of the single manually-selected
    # voice for the whole thing — e.g. a German paragraph plays in a
    # user-tagged German voice, an English paragraph in a user-tagged
    # English voice. Falls back to the selected voice/style wherever no
    # tagged voice exists for that language.
    do_auto_voice = bool(int(auto_voice))
    lang_to_style: dict[str, Any] = {}
    lang_to_name: dict[str, str] = {}
    if do_auto_voice:
        for pl in dict.fromkeys(pl for _, pl in para_list):
            cand = _voice_for_lang(pl, v["id"])
            if not cand:
                continue
            try:
                cand_style = await run_in_threadpool(_resolve_style, cand)
            except Exception:
                cand_style = None
            if cand_style is not None:
                lang_to_style[pl] = cand_style
                lang_to_name[pl] = cand["name"]

    sr = engine.sample_rate
    sid = uuid.uuid4().hex[:12]
    do_stream = bool(int(streaming))

    # Flatten to (paragraph_index, sentence, lang): the atoms the stream loop
    # merges into engine chunks. Paragraph indices let it insert the right
    # silence (sentence gap within a paragraph, longer pause at a break)
    # BETWEEN yields — silence is free playback runway.
    sent_flat: list[tuple[int, str, str]] = [
        (pi, s, pl) for pi, (sents, pl) in enumerate(para_list) for s in sents
    ]
    n_sents = len(sent_flat)
    total_chars = sum(len(s) for _, s, _ in sent_flat)

    # Seed the chunk scheduler from measured history for this steps value and
    # give the frontend a total-duration estimate for its progress bar.
    fixed_seed, rate_seed, cps_seed, _safety_seed = _rtf_history.estimates(int(steps))
    # SPEECH seconds and PAUSE seconds are estimated separately and shipped
    # separately (X-SPEECH-S / X-EST-AUDIO-S). The client's buffer math needs
    # them apart: a pause costs no inference at all, so billing it at the
    # generation rate makes a paragraph-heavy script wait seconds longer
    # before playback than the arithmetic actually requires.
    est_speech_s = total_chars / cps_seed
    _n_para_breaks = max(0, len(para_list) - 1)
    _n_sent_gaps = max(0, n_sents - 1 - _n_para_breaks)
    est_pause_s = (PARAGRAPH_PAUSE_S * _n_para_breaks
                   + (max(0.0, float(silence)) * 0.9 if pcfg.enabled
                      else max(0.0, float(silence))) * _n_sent_gaps)
    est_audio_s = est_speech_s + est_pause_s
    # Will this script outrun realtime generation? At this steps value, if
    # the model can't keep up (RTF > 1 — e.g. Quality mode on modest
    # hardware), no chunk-scheduling policy can prevent a mid-stream stall
    # for a long enough script — only fewer steps would, and that's off the
    # table (never trade quality for pacing). Estimate steady-state chunk
    # count from the cap size, so the frontend can warn instead of the user
    # being surprised mid-playback.
    _steady_chunk_audio_s = max(0.5, (CHUNK_CHARS / max(cps_seed, 1.0)))
    _n_chunks_est = max(1, round(est_audio_s / _steady_chunk_audio_s))
    est_infer_s = fixed_seed * _n_chunks_est + rate_seed * est_audio_s
    will_likely_stall = do_stream and est_audio_s > 3.0 and est_infer_s > est_audio_s * 1.05

    _remember_synth_stats(sid, {
        "done": False, "rtf": None, "ttfa_s": None, "infer_s": 0.0,
        "audio_s": 0.0, "chunks": n_sents, "chunks_done": 0, "error": "",
        "streaming": do_stream, "voices_used": [v["name"]],
    })

    # Flat fallback fillers — used only when the natural layer is switched
    # off (the A/B reference: fixed gap, digital silence, exactly what the
    # app did before prosody.py existed).
    gap_pcm = b"\x00\x00" * int(max(0.0, float(silence)) * sr)
    pause_pcm = b"\x00\x00" * int(PARAGRAPH_PAUSE_S * sr)

    # Sync generator: Starlette iterates it in a threadpool, so the blocking
    # engine calls don't stall the event loop.
    def _stream():
        infer_total = 0.0
        synth_audio_s = 0.0   # pure speech seconds (no gap/pause fillers)
        chars_done = 0
        pcm_bytes = 0
        done_sents = 0
        ttfa = None
        error = ""
        loudness = prosody.Loudness()
        peak_guard = prosody.PeakGuard(target_peak=0.85)
        master_chain = mastering.Master(sr, mcfg)
        room_floor = 0.0       # measured noise floor of the voice itself
        voice_level = 0.0      # running speech RMS, for breath/room-tone level
        chars_since_rest = 0   # how long the speaker has gone without a real
                               # pause — feeds the breath-debt term
        pause_s_total = 0.0
        prev_text = ""         # what the last chunk ended on — decides the
                               # length of the pause that follows it
        voices_used: set[str] = set()  # actually used, not just eligible —
                                        # headers must be set before the body
                                        # starts, before this is known, so it
                                        # only surfaces via /stats, not a header
        t0 = time.perf_counter()
        t_play = None         # proxy for when the client started playback
        try:
            yield _wav_header_streaming(sr)
            prev_para: Optional[int] = None
            i = 0
            while i < n_sents:
                pi, _sent, chunk_lang = sent_flat[i]
                cap = 120 if chunk_lang == "ko" else CHUNK_CHARS
                if not do_stream:
                    allowed = cap
                elif t_play is None:
                    allowed = 1   # first chunk: exactly one sentence — the
                                  # fastest possible first audio at full quality
                else:
                    # Fresh read every chunk: reflects this same stream's own
                    # chunks so far (updated below) plus cross-run history.
                    fixed_est, rate_est, cps_est, safety_mult = _rtf_history.estimates(int(steps))
                    runway = pcm_bytes / 2.0 / sr - (time.perf_counter() - t_play)
                    allowed = _stream_allowance(
                        runway, fixed_est * safety_mult, rate_est * safety_mult,
                        cps_est, cap)
                j = _take_chunk(sent_flat, i, allowed)
                chunk_text = " ".join(s for _, s, _ in sent_flat[i:j])

                if prev_para is not None:
                    para_break = pi != prev_para
                    if pcfg.enabled:
                        # The rest is chosen by what the PREVIOUS chunk ended
                        # on (a comma suspends, a question mark lands), how
                        # long the speaker has gone without breathing, and a
                        # deterministic ±11% wobble so the rhythm never turns
                        # into a metronome.
                        gap_s = prosody.pause_seconds(
                            prev_text, pcfg, para_break=para_break,
                            chars_since_rest=chars_since_rest)
                        gap_wav = prosody.gap_audio(
                            gap_s, sr, pcfg, floor=room_floor,
                            speech_level=voice_level, key=prev_text[-40:])
                        # Through the same chain as the speech: the room the
                        # last word was spoken in has to still be there while
                        # nobody is talking.
                        filler = _pcm16(peak_guard.apply(
                            master_chain.process(gap_wav)))
                        pause_s_total += gap_s
                        if gap_s >= pcfg.breath_min_pause_s:
                            chars_since_rest = 0   # that pause WAS the breath
                    else:
                        filler = pause_pcm if para_break else gap_pcm
                    if filler:
                        pcm_bytes += len(filler)
                        yield filler
                prev_para = sent_flat[j - 1][0]

                chunk_style = lang_to_style.get(chunk_lang, style)
                voices_used.add(lang_to_name.get(chunk_lang, v["name"]))
                # Per-chunk speaking rate. The duration predictor returns one
                # total length per chunk and `speed` divides it, so this is
                # the only timing lever the engine has — prosody.rate_scale
                # spends it on paragraph position, sentence type and a small
                # wobble instead of leaving every sentence metronomically
                # identical. Costs nothing.
                first_of_para = i == 0 or sent_flat[i - 1][0] != pi
                last_of_para = j >= n_sents or sent_flat[j][0] != pi
                chunk_speed = float(speed) * prosody.rate_scale(
                    chunk_text, pcfg, first_of_para=first_of_para,
                    last_of_para=last_of_para)
                wav, _d, infer = engine.synthesize(
                    chunk_text, chunk_style, int(steps), chunk_speed,
                    chunk_lang, float(silence))
                infer_total += infer
                done_sents = j
                # Rumble filter, edge trim (so THIS module owns the rhythm,
                # not the model's leftover padding) and seam de-click fades.
                wav = prosody.shape_chunk(wav, sr, pcfg)
                if pcfg.enabled:
                    if pcfg.loudness:
                        wav = loudness.apply(wav, sr)
                    lvl = prosody.speech_rms(wav, sr)
                    if lvl > 0:
                        voice_level = lvl if voice_level <= 0 else 0.7 * voice_level + 0.3 * lvl
                    fl = prosody.noise_floor(wav, sr)
                    room_floor = fl if room_floor <= 0 else 0.7 * room_floor + 0.3 * fl
                    chars_since_rest += len(chunk_text)
                wav = peak_guard.apply(master_chain.process(wav))
                prev_text = chunk_text
                chunk_audio_s = len(wav) / sr
                synth_audio_s += chunk_audio_s
                chars_done += len(chunk_text)
                # Feed the persistent (fixed_s, rate_rtf, cps) model with this
                # chunk's real measurement immediately — sharpens the estimate
                # the NEXT chunk of THIS stream reads (a few lines up), and
                # every future run at this steps value. Per chunk (not once
                # per whole stream) so a 1-sentence first chunk and a 300-char
                # later chunk each teach the model what they actually are,
                # instead of being blended into one polluted average.
                if chunk_audio_s > 0.05:
                    _rtf_history.update(int(steps), len(chunk_text), chunk_audio_s, infer)
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                pcm = _pcm16(wav)
                pcm_bytes += len(pcm)
                # live progress for /api/synthesize/stats/{sid} while streaming
                st = _SYNTH_STATS.get(sid)
                if st is not None and not st.get("done"):
                    st.update({
                        "chunks_done": done_sents,
                        "infer_s": round(infer_total, 3),
                        "audio_s": round(pcm_bytes / 2.0 / sr, 3),
                        "ttfa_s": round(ttfa, 3),
                        "voices_used": sorted(voices_used),
                    })
                yield pcm
                if t_play is None:
                    t_play = time.perf_counter()
                i = j
            # Ending: a short outro block (silence by default, room tone only
            # if it is switched on) so the file does not stop dead on the last
            # consonant, then whatever is still ringing in the reverb (≤55 ms).
            if pcfg.enabled and voice_level > 0:
                outro = prosody.gap_audio(
                    0.20, sr, prosody.ProsodyConfig(
                        enabled=True, breath=False, room_tone=pcfg.room_tone),
                    floor=room_floor, speech_level=voice_level, key="outro")
                outro_pcm = _pcm16(peak_guard.apply(master_chain.process(outro)))
                pcm_bytes += len(outro_pcm)
                yield outro_pcm
            tail = peak_guard.apply(master_chain.flush())
            if tail.size:
                tail_pcm = _pcm16(tail)
                pcm_bytes += len(tail_pcm)
                yield tail_pcm
        except GeneratorExit:  # client aborted the stream
            raise
        except Exception as e:
            engine.log_error("synthesize_stream", e)
            error = f"{type(e).__name__}: {e}"
        finally:
            audio_s = pcm_bytes / 2.0 / sr
            rtf = round(infer_total / audio_s, 4) if audio_s > 0 else 0.0
            # Teach the Auto quality mode what this step count really costs
            # end to end (only complete runs — an aborted stream's RTF is
            # whatever fraction of it happened to be rendered).
            if not error and done_sents == n_sents:
                _rtf_history.note_stream(int(steps), rtf, audio_s)
            _remember_synth_stats(sid, {
                "done": True,
                "rtf": rtf,
                "ttfa_s": round(ttfa, 3) if ttfa is not None else None,
                "infer_s": round(infer_total, 3),
                "audio_s": round(audio_s, 3),
                "chunks": n_sents,
                "chunks_done": done_sents,
                "error": error,
                "streaming": do_stream,
                "voices_used": sorted(voices_used) if voices_used else [v["name"]],
                "steps": int(steps),
                "auto_quality": auto_quality,
                "natural": pcfg.enabled,
                "master": mcfg.enabled,
                "speech_s": round(synth_audio_s, 3),
                "pause_s": round(pause_s_total, 3),
                # gated K-weighted loudness of the engine output BEFORE the
                # auto-level move (BS.1770) — how far off target the voice
                # itself was, which is the number worth watching per voice
                "lufs_in": (round(master_chain.measured_lufs(), 2)
                            if master_chain.measured_lufs() is not None else None),
                "level_db": round(20.0 * float(np.log10(max(master_chain.level_gain, 1e-6))), 2),
            })
            print(f"[synth {sid}] {v['name']}: {done_sents}/{n_sents} sentences "
                  f"({lang_label}, {'stream' if do_stream else 'batch'}), "
                  f"steps {int(steps)}{' auto' if auto_quality else ''} -> "
                  f"{audio_s:.1f}s audio "
                  f"({synth_audio_s:.1f}s speech + {pause_s_total:.1f}s pauses), "
                  f"infer {infer_total:.1f}s, RTF {rtf}"
                  + (f", TTFA {ttfa:.2f}s" if ttfa is not None else "")
                  + (", natural" if pcfg.enabled else "")
                  + (", master" if mcfg.enabled else "")
                  + (f", ERROR {error}" if error else ""), flush=True)

    ci = (v.get("clone_info") or {})
    headers = {
        "X-SYNTH-ID": sid,
        "X-SR": str(sr),
        "X-CHUNKS": str(n_sents),
        "X-STEPS": str(int(steps)),
        "X-LANG": lang_label,
        "X-DEVICE": engine.device_label,
        "X-AUTO-CLONED": "1" if auto_cloned else "0",
        "X-CLONE-MS": f"{clone_ms:.0f}",
        "X-EST-SIM": str(ci.get("est_sim_pct", "")),
        "X-EST-AUDIO-S": f"{est_audio_s:.1f}",
        # Opening bid for the client's buffer controller. The server never
        # sleeps and never holds audio back — it only says what it expects,
        # and the client's own live measurement overrides this within a few
        # seconds (see the PRIOR_FADE logic in createStreamPlayer). A wrong
        # value here is therefore cheap, which is the point of the split.
        "X-SPEECH-S": f"{est_speech_s:.1f}",
        "X-RTF-HI": f"{_expected_stream_rtf(int(steps)):.3f}",
        # Expected wall time for ONE chunk. A stream can average well under
        # realtime and still leave the player dry waiting for a single chunk,
        # because a chunk is one whole model pass whose fixed cost is seconds
        # at high step counts. The client sizes its start buffer on this as
        # well as on the rate, then switches to its own measured arrival gaps.
        "X-CHUNK-LAT-S": f"{(fixed_seed + rate_seed * _steady_chunk_audio_s) * _safety_seed:.2f}",
        "X-WILL-STALL": "1" if will_likely_stall else "0",
        "X-AUTO-QUALITY": "1" if auto_quality else "0",
        "X-NATURAL": "1" if pcfg.enabled else "0",
        "X-MASTER": "1" if mcfg.enabled else "0",
        "X-NORMALIZED": urllib.parse.quote(norm_text[:900]),
        "Access-Control-Expose-Headers": (
            "X-SYNTH-ID,X-SR,X-CHUNKS,X-STEPS,X-LANG,X-DEVICE,X-AUTO-CLONED,"
            "X-CLONE-MS,X-EST-SIM,X-EST-AUDIO-S,X-WILL-STALL,X-NORMALIZED,"
            "X-AUTO-QUALITY,X-NATURAL,X-MASTER,X-SPEECH-S,X-RTF-HI,"
            "X-CHUNK-LAT-S"
        ),
    }
    return StreamingResponse(_stream(), media_type="audio/wav", headers=headers)


@app.get("/api/synthesize/stats/{sid}")
def api_synthesize_stats(sid: str):
    st = _SYNTH_STATS.get(sid)
    if st is None:
        raise HTTPException(status_code=404, detail="unknown synthesis id")
    return st


@app.post("/api/benchmark/ecapa_steps")
async def api_benchmark_ecapa_steps(
    voice_id: str = Form(...),
    steps_list: str = Form("1,2,4,8,12,16,24,32"),
):
    """Sweep ECAPA encoder steps on one reference: for each value, run
    smart-init, then verify (synthesize sample + ECAPA cosine vs target).
    Used to calibrate DEFAULT_ECAPA_STEPS."""
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="engine still loading")
    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")
    ref = voices_mod.ref_path(v)
    if not ref.exists():
        raise HTTPException(status_code=404, detail="reference audio missing")

    values = sorted({max(1, min(MAX_ECAPA_STEPS, int(s))) for s in steps_list.split(",") if s.strip()})

    def _sweep():
        rows = []
        for s in values:
            res = ecapa_mod.smart_init_clone(engine, str(ref), ecapa_steps=s)
            ver = ecapa_mod.verify_clone(engine, res["ttl"], res["dp"], res["target_emb"])
            rows.append({
                "steps": s,
                "effective_steps": res["effective_steps"],
                "clone_ms": res["timings_ms"]["total"],
                "embed_ms": res["timings_ms"]["embed"],
                "est_sim_pct": res["est_sim_pct"],
                "verified_sim_pct": ver["verified_sim_pct"],
                "top3": [r["voice"] for r in res["ranking"][:3]],
                "weights": [r["weight"] for r in res["ranking"][:3]],
            })
        return rows

    rows = await run_in_threadpool(_sweep)
    return {"voice": v["name"], "device": engine.device_full_label, "results": rows}


@app.get("/api/errors")
def api_errors():
    return {"errors": engine.errors[-20:]}


# ---- main ----
def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def _open_browser(url: str) -> None:
    def _go():
        time.sleep(2.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def main() -> None:
    import uvicorn
    env_port = os.environ.get("PORT") or os.environ.get("ECAPAFLOW_PORT")
    if env_port:
        port = int(env_port)
    else:
        port = 7873
        for candidate in (7873, 7874, 7875, 7876):
            if not _port_in_use(candidate):
                port = candidate
                break
    # 0.0.0.0 = reachable from other devices on the same network (e.g. the
    # phone as remote control while the laptop does the actual cloning).
    # Set ECAPAFLOW_HOST=127.0.0.1 to go back to localhost-only.
    host = os.environ.get("ECAPAFLOW_HOST", "0.0.0.0")
    url = f"http://localhost:{port}"
    print(f"ECAPAFlow starting on {url} (host {host})")
    try:
        import socket as _s
        lan_ip = _s.gethostbyname_ex(_s.gethostname())[2]
        lan_ip = next((ip for ip in lan_ip if not ip.startswith("127.")), None)
        if host == "0.0.0.0" and lan_ip:
            print(f"  on your phone (same Wi-Fi): http://{lan_ip}:{port}")
    except Exception:
        pass
    if os.environ.get("ECAPAFLOW_NO_BROWSER") != "1":
        _open_browser(url)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
