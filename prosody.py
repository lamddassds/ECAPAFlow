"""Natural-speech layer for ECAPAFlow — everything that makes the stream
sound like one human take instead of a row of TTS clips glued together.

Deliberately NOT a model and NOT an LLM: every function here is a few
hundred microseconds of numpy on audio the engine already produced, so it
costs no inference time at all. It actually IMPROVES the RTF number,
because the pauses it inserts are free seconds of output audio.

What the engine gives us, and why it needs this layer:

* Each streamed chunk is a separate model pass. Its head and tail carry a
  variable amount of near-silence (the DirectML fast path skips the pip
  package's own `silence_duration` padding entirely, the fallback path
  applies it) — so the rhythm between sentences is whatever the model
  happened to leave over, not a rhythm anyone chose. `trim_edges` cuts
  that off so the pause between two sentences is EXACTLY the pause this
  module decided on.
* Chunk boundaries are hard sample cuts into digital zero. That clicks,
  and dead-flat zero between phrases reads as "muted", not as "silence in
  a room". `fades` + `room_tone` fix both.
* Every sentence gets the same gap regardless of how it ended. Humans
  don't do that: a comma is a breath-length hesitation, a question mark
  lands and lingers, a paragraph is a reset. `pause_seconds` scales the
  gap by the punctuation that caused it, by how long the speaker just
  went without breathing, and by a small deterministic jitter so the
  rhythm never turns metronomic.
* Long scripts drift in level chunk to chunk. `Loudness` nudges each
  chunk toward a running target, hard-clamped so it can never pump.

Everything is deterministic: the "randomness" (jitter, noise, breath) is
seeded from a hash of the text, so the same script renders identically
twice — the same audio the user approved is the audio they download.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
from scipy import signal

# ---------------------------------------------------------------- pause map
# Multipliers on the user's Silence Duration slider (the "period" gap = 1.0).
# Calibrated against how long speakers actually rest on each boundary:
# commas and colons are mid-thought suspensions (the voice does not fully
# reset), terminal marks get the full stop, and ellipses/dashes trail.
#
# Anchored to the millisecond ranges published for commercial TTS pause
# controls (comma 150-200 ms, semicolon 200-250, colon 150-200, terminal
# marks 300-400, ellipsis 400-600, paragraph 400-1000) at the default
# 0.30 s Silence Duration — so the slider stays the master control and the
# table only says how each mark relates to it.
PUNCT_PAUSE = {
    ",": 0.55,
    ";": 0.75,
    ":": 0.60,
    ".": 1.00,
    "!": 1.10,
    "?": 1.15,
    "…": 1.60,
    "—": 1.30,
    "–": 1.20,
    "-": 1.20,
    ")": 1.00,
    '"': 1.05,
    "»": 1.05,
    "”": 1.05,
    "'": 1.00,
}
# A chunk that ends on no punctuation at all is a hard mid-sentence split:
# the thought continues, so barely rest.
NO_PUNCT_PAUSE = 0.38

# A speaker who has been going for a while rests longer at the next chance
# (that is where the breath goes). Adds up to +35% over ~2 sentences worth
# of characters, then stops growing.
BREATH_DEBT_CHARS = 700.0
BREATH_DEBT_MAX = 0.35

PAUSE_JITTER = 0.11      # ±11% deterministic wobble — kills the metronome
MIN_PAUSE_S = 0.04
MAX_PAUSE_S = 2.20


@dataclass
class ProsodyConfig:
    """One knob per effect so any of them can be A/B'd in isolation."""
    enabled: bool = True
    base_gap_s: float = 0.30      # the Silence Duration slider
    para_gap_s: float = 0.60      # paragraph break, before punctuation scaling
    trim: bool = True             # cut the model's ragged head/tail silence
    fades: bool = True            # de-click the seams
    # Room tone and breath are OFF by default. Both were implemented, both
    # measured as designed — and Lauro heard them immediately (2026-07-27):
    # "I hear the rustling, I think it could be breathing, remove that again."
    # That is the verdict that counts. A noise floor you can notice is worse
    # than digital silence, and an inhale you notice is worse than none: the
    # listener has no model for why a synthetic voice would breathe, so it
    # reads as an artefact rather than as a person. The code stays because
    # the levels are the only thing wrong with it — re-enable per request.
    room_tone: bool = False       # gaps get the recording's own noise floor
    breath: bool = False          # audible inhale before a phrase after a rest
    breath_gain: float = 0.06     # relative to the speech RMS (~-24 dB)
    breath_min_pause_s: float = 0.34
    loudness: bool = True         # match chunk-to-chunk level
    rate_jitter: float = 0.02     # random part of the per-chunk rate; the
                                  # structured part (paragraph position,
                                  # sentence type) lives in rate_scale
    pause_jitter: float = PAUSE_JITTER
    seed: int = 0


# --------------------------------------------------------------- text side
_TRAILING_CLOSERS = ')"»”\'’]}'


def tail_punct(text: str) -> str:
    """The punctuation mark that actually ended this chunk, looking through
    a closing quote/bracket ('… said the fox." ' ends on a period, not a
    quote) and collapsing '...' to the single ellipsis character."""
    s = (text or "").rstrip()
    if not s:
        return ""
    if s.endswith("..."):
        return "…"
    i = len(s) - 1
    while i > 0 and s[i] in _TRAILING_CLOSERS:
        i -= 1
    return s[i]


def _jitter(key: str, spread: float) -> float:
    """Deterministic value in [-spread, +spread] derived from the text, so
    the wobble is reproducible across renders of the same script."""
    if spread <= 0:
        return 0.0
    h = hashlib.blake2b(key.encode("utf-8", "ignore"), digest_size=4).digest()
    frac = int.from_bytes(h, "big") / 0xFFFFFFFF   # [0, 1]
    return (frac * 2.0 - 1.0) * spread


def pause_seconds(prev_text: str, cfg: ProsodyConfig, *, para_break: bool,
                  chars_since_rest: int = 0) -> float:
    """How long to rest after `prev_text` before the next chunk starts."""
    if not cfg.enabled:
        return cfg.para_gap_s if para_break else cfg.base_gap_s
    base = cfg.para_gap_s if para_break else cfg.base_gap_s
    mark = tail_punct(prev_text)
    weight = PUNCT_PAUSE.get(mark, NO_PUNCT_PAUSE if mark and not mark.isalnum()
                             else NO_PUNCT_PAUSE)
    if mark.isalnum():          # ended mid-word: a hard split, keep it tight
        weight = NO_PUNCT_PAUSE
    debt = 1.0 + min(BREATH_DEBT_MAX, chars_since_rest / BREATH_DEBT_CHARS)
    wobble = 1.0 + _jitter(prev_text[-40:] or "seed", cfg.pause_jitter)
    return float(np.clip(base * weight * debt * wobble, MIN_PAUSE_S, MAX_PAUSE_S))


def rate_scale(text: str, cfg: ProsodyConfig, *, first_of_para: bool = False,
               last_of_para: bool = False) -> float:
    """Per-chunk speaking-rate multiplier.

    This is the ONLY timing lever the engine exposes: the duration predictor
    returns a single total length per chunk, which `speed` divides — there is
    no per-phoneme duration to shape. So the human-ness that a real speaker
    puts INSIDE a sentence has to be spent here, across sentences, where it
    still maps onto something real:

    * a paragraph's last chunk slows down — final lengthening at a discourse
      boundary is one of the most reliable prosodic universals;
    * its opening chunk is slightly slower than its middle (speakers settle
      into a paragraph and then pick up);
    * questions run slightly slower than statements;
    * very short utterances are relatively slower than long ones;
    * and a small deterministic wobble on top, because even after all that a
      constant rate is the giveaway.

    Bounded to ±6% overall — past that it stops reading as delivery and
    starts reading as a speed control.
    """
    if not cfg.enabled:
        return 1.0
    scale = 1.0
    if last_of_para:
        scale *= 0.97
    elif first_of_para:
        scale *= 0.99
    else:
        scale *= 1.01
    mark = tail_punct(text)
    if mark == "?":
        scale *= 0.985
    if len(text.strip()) < 25:
        scale *= 0.98
    scale *= 1.0 + _jitter("rate:" + (text[:60] or "seed"), cfg.rate_jitter)
    return float(np.clip(scale, 0.94, 1.06))


_CLAUSE_BREAK_RE = re.compile(r"[,;:—–]\s")


def clause_split_point(s: str, limit: int) -> int:
    """Where to cut a run-on sentence that exceeds `limit` characters.

    Prefers a real clause boundary (comma, semicolon, colon, dash) in the
    second half of the window — cutting there gives the model a phrase it
    can shape, instead of half a noun phrase. Falls back to the last space,
    then to a hard character cut.
    """
    window = s[:limit]
    best = -1
    for m in _CLAUSE_BREAK_RE.finditer(window):
        if m.start() >= limit // 3:
            best = m.start() + 1        # keep the punctuation with the left part
    if best > 0:
        return best
    cut = window.rfind(" ", limit // 2)
    return cut if cut != -1 else limit


def continuation_text(piece: str) -> str:
    """A piece that got hard-split mid-sentence has no terminal punctuation,
    so the model renders it with a sentence-final falling contour and the
    listener hears a full stop that isn't there. A trailing comma asks for a
    continuation contour instead — the phrase stays suspended, which is
    exactly what it is."""
    s = piece.rstrip()
    if not s or not s[-1].isalnum():
        return piece
    return s + ","


# ---------------------------------------------------------------- DSP side
def dc_block(wav: np.ndarray, sr: int, cutoff: float = 45.0) -> np.ndarray:
    """High-pass. Removes the DC offset and sub-bass rumble that would
    otherwise make the seams thump where two chunks meet — and, more
    importantly, would bias the edge-trim detector below into thinking
    silence is signal."""
    if wav.size < 16:
        return wav
    sos = signal.butter(1, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, wav.astype(np.float32)).astype(np.float32)


def _frame_rms(wav: np.ndarray, sr: int, win_ms: float = 10.0) -> tuple[np.ndarray, int]:
    hop = max(1, int(sr * win_ms / 1000.0))
    n = wav.size // hop
    if n < 1:
        return np.array([float(np.sqrt(np.mean(wav ** 2)) if wav.size else 0.0)]), hop
    frames = wav[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1)), hop


def noise_floor(wav: np.ndarray, sr: int) -> float:
    """RMS of the quietest 10% of frames — the recording's own silence."""
    rms, _ = _frame_rms(wav, sr)
    if rms.size == 0:
        return 0.0
    return float(np.percentile(rms, 10.0))


def speech_rms(wav: np.ndarray, sr: int) -> float:
    """RMS of the loud 40% of frames — level of the speech itself, not
    diluted by however much silence this chunk happens to contain."""
    rms, _ = _frame_rms(wav, sr)
    if rms.size == 0:
        return 0.0
    loud = rms[rms >= np.percentile(rms, 60.0)]
    return float(np.sqrt(np.mean(loud ** 2))) if loud.size else float(np.mean(rms))


def trim_edges(wav: np.ndarray, sr: int, *, keep_ms: float = 30.0,
               floor_db: float = -50.0, max_trim_frac: float = 0.45) -> np.ndarray:
    """Cut the model's leading/trailing near-silence so THIS module owns the
    rhythm. `keep_ms` of the original silence stays as a natural onset/decay,
    and the trim is capped so a genuinely quiet chunk (a whispered line) can
    never be eaten.

    The threshold sits 50 dB under the chunk's loudest frame, not the ~40 dB
    a VAD would use: a sentence-final /s/ or /f/ decays a long way down, and
    clipping that tail is far more audible than leaving a few extra
    milliseconds of silence — which the pause logic absorbs anyway."""
    if wav.size < sr // 50:
        return wav
    rms, hop = _frame_rms(wav, sr)
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0:
        return wav
    thresh = peak * (10.0 ** (floor_db / 20.0))
    loud = np.nonzero(rms > thresh)[0]
    if loud.size == 0:
        return wav
    keep = int(sr * keep_ms / 1000.0)
    start = max(0, int(loud[0]) * hop - keep)
    end = min(wav.size, (int(loud[-1]) + 1) * hop + keep)
    max_trim = int(wav.size * max_trim_frac)
    start = min(start, max_trim)
    end = max(end, wav.size - max_trim)
    if end - start < sr // 100:
        return wav
    return wav[start:end]


def apply_fades(wav: np.ndarray, sr: int, *, in_ms: float = 6.0,
                out_ms: float = 14.0) -> np.ndarray:
    """Raised-cosine in/out. Without it every chunk boundary is a step
    discontinuity — an audible tick at each seam, the single most obvious
    'this was stitched together' cue."""
    if wav.size == 0:
        return wav
    out = wav.copy()
    ni = min(int(sr * in_ms / 1000.0), out.size // 2)
    no = min(int(sr * out_ms / 1000.0), out.size // 2)
    if ni > 1:
        out[:ni] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ni))).astype(np.float32)
    if no > 1:
        out[-no:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, no))).astype(np.float32)
    return out


def _seeded_noise(n: int, key: str) -> np.ndarray:
    rng = np.random.default_rng(
        int.from_bytes(hashlib.blake2b(key.encode("utf-8", "ignore"),
                                       digest_size=8).digest(), "big"))
    return rng.standard_normal(n).astype(np.float32)


def _one_pole_lp(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Smoothing filter — turns white noise into something with the spectral
    tilt of room tone / breath instead of hiss."""
    if x.size < 16:
        return x
    sos = signal.butter(1, min(cutoff, sr * 0.45), btype="lowpass",
                        fs=sr, output="sos")
    return signal.sosfilt(sos, x.astype(np.float32)).astype(np.float32)


def room_tone(n: int, level: float, sr: int, key: str) -> np.ndarray:
    """Fill a gap with the recording's own noise floor instead of digital
    zero. Silence that is exactly 0.0 for half a second is a hole in the
    recording; matched room tone reads as the speaker simply not talking."""
    if n <= 0 or level <= 0:
        return np.zeros(max(0, n), dtype=np.float32)
    noise = _one_pole_lp(_seeded_noise(n, "room:" + key), sr, 3000.0)
    rms = float(np.sqrt(np.mean(noise ** 2))) or 1.0
    return (noise * (level / rms)).astype(np.float32)


def breath(sr: int, dur_s: float, level: float, key: str) -> np.ndarray:
    """A synthetic inhale: band-limited noise under an asymmetric envelope
    (slow swell, quick release), sitting ~27 dB under the speech. Placed at
    the END of a pause, because that is where a real speaker takes the
    breath they are about to talk on."""
    n = int(sr * max(0.05, dur_s))
    if n <= 0 or level <= 0:
        return np.zeros(0, dtype=np.float32)
    noise = _seeded_noise(n, "breath:" + key)
    # 4th-order band-pass over 350–2400 Hz: the airy region a real inhale
    # lives in. A gentler slope leaves enough top octave in to read as tape
    # hiss rather than as a person — the whole effect fails on that detail.
    sos = signal.butter(4, [350.0, 2400.0], btype="bandpass", fs=sr, output="sos")
    band = signal.sosfilt(sos, noise).astype(np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    env = (t ** 0.75) * ((1.0 - t) ** 0.45)
    env /= float(env.max()) or 1.0
    out = band * env
    rms = float(np.sqrt(np.mean(out ** 2))) or 1.0
    return (out * (level / rms)).astype(np.float32)


def gap_audio(seconds: float, sr: int, cfg: ProsodyConfig, *, floor: float,
              speech_level: float, key: str) -> np.ndarray:
    """The audio that goes BETWEEN two chunks: room tone for the whole rest,
    with an inhale tucked into its tail when the rest is long enough to have
    been a breath."""
    n = int(sr * max(0.0, seconds))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if cfg.enabled and cfg.room_tone and speech_level > 0:
        # Bounded both ways. The upper bound keeps room tone from ever being
        # noticed; the LOWER bound exists because some chunks come back with
        # a noise floor so low it quantizes to literal zero at 16 bit — and a
        # dead-zero hole between two phrases is exactly the "muted" artifact
        # this is here to prevent. -76 dB relative to the speech is still
        # inaudible in isolation but keeps the recording alive.
        # …and an ABSOLUTE lower bound of ~2 LSB at 16 bit: below that the
        # room tone quantizes straight back to digital zero on the way out
        # and the whole effect disappears (measured — gaps were still coming
        # out bit-exact silent before this).
        level = float(np.clip(max(floor, 6e-5),
                              speech_level * 0.00016, speech_level * 0.02))
        level = max(level, min(6e-5, speech_level * 0.02))
        buf = room_tone(n, level, sr, key)
    else:
        buf = np.zeros(n, dtype=np.float32)
    if (cfg.enabled and cfg.breath and speech_level > 0
            and seconds >= cfg.breath_min_pause_s):
        b = breath(sr, min(0.26, seconds * 0.62),
                   speech_level * cfg.breath_gain, key)
        if 0 < b.size <= n:
            # End the inhale ~40 ms before the next word starts.
            lead = int(sr * 0.04)
            start = max(0, n - b.size - lead)
            buf[start:start + b.size] += b
    return buf


class Loudness:
    """Running level matcher across chunks of one stream.

    A per-chunk normaliser would flatten the delivery (quiet lines are
    supposed to be quiet). This only corrects DRIFT: it tracks a slow target
    and moves each chunk at most `max_step_db` toward it, so a deliberately
    soft sentence stays soft while a chunk that came out 4 dB hot gets
    walked back over a couple of chunks instead of jumping.
    """

    def __init__(self, max_step_db: float = 1.5, alpha: float = 0.25) -> None:
        self.target: float | None = None
        self.max_step = 10.0 ** (max_step_db / 20.0)
        self.alpha = alpha

    def apply(self, wav: np.ndarray, sr: int) -> np.ndarray:
        lvl = speech_rms(wav, sr)
        if lvl <= 1e-6:
            return wav
        if self.target is None:
            self.target = lvl
            return wav
        gain = float(np.clip(self.target / lvl, 1.0 / self.max_step, self.max_step))
        self.target = (1.0 - self.alpha) * self.target + self.alpha * (lvl * gain)
        return (wav * gain).astype(np.float32)


class PeakGuard:
    """Stream-wide peak ceiling with a gain that can only go DOWN.

    The old per-chunk `_soft_limit` rescaled every chunk independently, so a
    chunk that happened to peak high got quieter than its neighbours — an
    audible level jump mid-sentence, caused by the safety net rather than by
    the voice. Holding one non-increasing gain for the whole stream keeps
    the relative dynamics of the performance intact and still guarantees no
    sample ever clips.
    """

    def __init__(self, target_peak: float = 0.85) -> None:
        self.ceiling = target_peak
        self.gain = 1.0

    def apply(self, wav: np.ndarray) -> np.ndarray:
        if wav.size == 0:
            return wav
        peak = float(np.abs(wav).max()) * self.gain
        if peak > self.ceiling:
            self.gain *= self.ceiling / peak
        return (wav * self.gain).astype(np.float32)


def shape_chunk(wav: np.ndarray, sr: int, cfg: ProsodyConfig) -> np.ndarray:
    """Everything that happens to one chunk before it is handed to the
    stream: rumble removal, edge trim, de-click fades."""
    if wav.size == 0 or not cfg.enabled:
        return wav
    out = dc_block(wav, sr)
    if cfg.trim:
        out = trim_edges(out, sr)
    if cfg.fades:
        out = apply_fades(out, sr)
    return out
