"""Streaming master chain — the polish layer ECAPAFlow was missing.

Why this exists: LoudFlow (same Supertonic 3 engine, same voices) sounds
more "produced" than raw ECAPAFlow output, and the reason is not the model.
LoudFlow runs its output through a Pedalboard chain — high-pass 80 Hz,
compressor at -18 dB / 3:1 / 5 ms / 100 ms, a 3% wet tiny-room reverb, and
a -1 dB limiter. Raw TTS is dry, dynamically uneven and slightly dull; that
chain fixes all three in about a millisecond of DSP.

This module reimplements the same idea in scipy/numpy (pedalboard is not
installed in the localvoice venv and this needs no new dependency), with
three differences that matter for THIS app:

* It is STREAMING-SAFE. Every filter keeps its `zi` state and the reverb
  keeps its tail across calls, so processing chunk-by-chunk is sample-for-
  sample identical to processing the finished file. Without that, each
  chunk would start from a cleared filter state — an audible discontinuity
  at exactly the seams this app is trying to hide.
* The PAUSES go through the chain too. That is what makes reverb tails ring
  out into the silence between sentences instead of being chopped at the
  chunk edge, which is the single strongest "one continuous take" cue.
* It adds the presence shelf and gentle de-essing that the research on TTS
  post-processing recommends (a +2-3 dB shelf around 6 kHz removes the
  muffled-robot quality; unmanaged, that same shelf makes sibilance worse,
  so a mild de-esser sits in front of it).

Everything is conservative on purpose: over-compression pumps, heavy reverb
turns a voice into a room, and aggressive de-essing lisps. The defaults are
the low end of every published range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal


@dataclass
class MasterConfig:
    enabled: bool = True
    hp_hz: float = 75.0          # rumble / plosive cleanup (LoudFlow: 80)
    deess: bool = True
    deess_hz: float = 5800.0     # sibilance band starts here
    deess_max_db: float = 4.0    # never more than this much reduction
    shelf_hz: float = 6000.0     # presence shelf — undoes "muffled TTS"
    shelf_db: float = 2.0        # measured: +2.5 dB here lands as +1.9 dB in
                                 # the 6-10 kHz band and +2.5 dB above 10 kHz.
                                 # Backed off to 2.0 because on a voice that
                                 # is already sibilant, brightness is the one
                                 # mistake you notice immediately — raise it
                                 # if the result sounds dull, not the reverse
    comp: bool = True
    comp_threshold_db: float = -20.0
    comp_ratio: float = 2.0      # gentle; >4:1 on speech pumps
    comp_attack_ms: float = 5.0
    comp_release_ms: float = 120.0
    comp_makeup_db: float = 1.5
    reverb: bool = True
    reverb_wet: float = 0.035    # LoudFlow ships 0.03 — just "air", not a room
    reverb_ms: float = 55.0      # early reflections only, no tail
    target_lufs: float = -18.0   # between the podcast standard (-16) and EBU
                                 # R128 (-23): loud enough to sit next to
                                 # normal media, with room under the ceiling
    level_max_db: float = 4.0    # hard clamp on the automatic level move
    level_rate_db_s: float = 0.9  # …and on how fast it may get there. Per
                                  # SECOND of audio, not per block: chunk
                                  # boundaries are decided by the streaming
                                  # scheduler and shift with machine timing,
                                  # so a per-block limit would make the same
                                  # script level differently run to run


def _biquad_high_shelf(sr: int, f0: float, gain_db: float, q: float = 0.707):
    """RBJ cookbook high-shelf as second-order sections."""
    a_ = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * np.sqrt(a_) * alpha
    b0 = a_ * ((a_ + 1) + (a_ - 1) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2 * a_ * ((a_ - 1) + (a_ + 1) * cos_w0)
    b2 = a_ * ((a_ + 1) + (a_ - 1) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a_ + 1) - (a_ - 1) * cos_w0 + two_sqrt_a_alpha
    a1 = 2 * ((a_ - 1) - (a_ + 1) * cos_w0)
    a2 = (a_ + 1) - (a_ - 1) * cos_w0 - two_sqrt_a_alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def k_weighting_sos(sr: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting as SOS: the RLB high-pass (~38 Hz) plus the
    +4 dB head/torso shelf at ~1.68 kHz.

    The standard publishes fixed coefficients for 48 kHz only. Designing both
    stages from their analog prototypes at the ACTUAL sample rate (44.1 kHz
    here) is the usual portable fix and matches the published response to a
    few hundredths of a dB across the speech band.
    """
    shelf = _biquad_high_shelf(sr, 1681.97, 3.999, q=0.7071)
    hp = signal.butter(2, 38.0, btype="highpass", fs=sr, output="sos")
    return np.vstack([shelf, hp])


def _one_pole_coeffs(sr: int, tau_ms: float) -> tuple[np.ndarray, np.ndarray]:
    a = float(np.exp(-1.0 / (sr * max(tau_ms, 0.1) / 1000.0)))
    return np.array([1.0 - a]), np.array([1.0, -a])


def _early_reflection_ir(sr: int, length_ms: float) -> np.ndarray:
    """Sparse early-reflection impulse response for a very small room.

    Twelve taps with decaying, alternating-sign gains and prime-ish spacing:
    dense enough to read as space, sparse enough that it cannot smear
    consonants (a real reverb tail on speech at this length would). Cost is
    a dozen shifted adds per block.
    """
    n = int(sr * length_ms / 1000.0)
    ir = np.zeros(n, dtype=np.float32)
    ir[0] = 1.0
    taps_ms = [7.3, 11.1, 14.7, 19.3, 23.9, 28.1, 33.7, 38.3, 43.1, 47.9, 52.3]
    for i, ms in enumerate(taps_ms):
        idx = int(sr * ms / 1000.0)
        if idx < n:
            ir[idx] += ((-1.0) ** i) * float(0.72 ** (i + 1))
    return ir


class Master:
    """One instance per synthesis stream. Feed it every block in order —
    speech chunks AND the pauses between them."""

    def __init__(self, sr: int, cfg: MasterConfig | None = None) -> None:
        self.sr = sr
        self.cfg = cfg or MasterConfig()
        c = self.cfg
        self.sos_hp = signal.butter(2, c.hp_hz, btype="highpass", fs=sr, output="sos")
        self.zi_hp = signal.sosfilt_zi(self.sos_hp) * 0.0
        self.sos_shelf = _biquad_high_shelf(sr, c.shelf_hz, c.shelf_db)
        self.zi_shelf = signal.sosfilt_zi(self.sos_shelf) * 0.0
        self.sos_ess = signal.butter(2, c.deess_hz, btype="highpass", fs=sr, output="sos")
        self.zi_ess = signal.sosfilt_zi(self.sos_ess) * 0.0
        self.b_att, self.a_att = _one_pole_coeffs(sr, c.comp_attack_ms)
        self.zi_att = np.zeros(max(len(self.a_att), len(self.b_att)) - 1)
        self.b_rel, self.a_rel = _one_pole_coeffs(sr, c.comp_release_ms)
        self.zi_rel = np.zeros(max(len(self.a_rel), len(self.b_rel)) - 1)
        self.b_ess, self.a_ess = _one_pole_coeffs(sr, 8.0)
        self.zi_ess_env = np.zeros(1)
        self.sos_k = k_weighting_sos(sr)
        self.zi_k = signal.sosfilt_zi(self.sos_k) * 0.0
        self.ir = _early_reflection_ir(sr, c.reverb_ms) if c.reverb else None
        self.tail = np.zeros(0, dtype=np.float32)
        self.level_gain = 1.0          # slow, clamped auto-level
        self._k_power_sum = 0.0        # gated K-weighted energy so far
        self._k_samples = 0

    # -- pieces ---------------------------------------------------------
    def _deess(self, x: np.ndarray) -> np.ndarray:
        """SPLIT-BAND de-esser: attenuate only the sibilant band, leave the
        rest of the voice untouched.

        The obvious implementation — measure the high band, duck the whole
        signal — is what a broadband compressor does, and on speech it is
        audible as the voice dipping every time an /s/ arrives. Subtracting a
        scaled copy of the band instead means an /s/ gets quieter while the
        vowel around it does not move at all. Reduction is capped low on
        purpose: over-de-essing lisps, which is worse than the sibilance.
        """
        band, self.zi_ess = signal.sosfilt(self.sos_ess, x, zi=self.zi_ess)
        env, self.zi_ess_env = signal.lfilter(
            self.b_ess, self.a_ess, np.abs(band), zi=self.zi_ess_env)
        broadband = float(np.sqrt(np.mean(x ** 2))) or 1e-6
        excess = np.clip(env / (broadband * 1.5 + 1e-9) - 1.0, 0.0, 1.0)
        # how much of the band to remove, 0..(1 - 10^(-max_db/20))
        keep = 10.0 ** (-self.cfg.deess_max_db / 20.0)
        remove = (1.0 - keep) * excess
        return (x - band * remove).astype(np.float32)

    def _compress(self, x: np.ndarray) -> np.ndarray:
        """Soft leveler: one-pole detector → static 2:1 curve → one-pole
        smoothing of the gain itself.

        Honest about what this is, after measurement: the second smoothing
        stage dominates, so the EFFECTIVE attack is the release time constant
        (~120 ms), not `comp_attack_ms`. Sweeping that field from 0.1 ms to
        5 ms moves the measured attack by under 6 ms. It is a slow leveler,
        not a peak compressor — which is the right tool here (peaks are
        handled by PeakGuard, and fast compression on speech pumps), but the
        field name promises something it does not deliver, so: `comp_attack_ms`
        sets the DETECTOR, `comp_release_ms` sets the audible time constant.
        """
        c = self.cfg
        env, self.zi_att = signal.lfilter(
            self.b_att, self.a_att, np.abs(x), zi=self.zi_att)
        env_db = 20.0 * np.log10(np.maximum(env, 1e-7))
        over = np.maximum(0.0, env_db - c.comp_threshold_db)
        gain_db = -over * (1.0 - 1.0 / c.comp_ratio)
        gain_db, self.zi_rel = signal.lfilter(
            self.b_rel, self.a_rel, gain_db, zi=self.zi_rel)
        return (x * 10.0 ** ((gain_db + c.comp_makeup_db) / 20.0)).astype(np.float32)

    def _reverb(self, x: np.ndarray) -> np.ndarray:
        """Sparse early reflections, overlap-added across blocks so the
        space does not restart at every chunk boundary."""
        if self.ir is None or x.size == 0:
            return x
        wet_full = signal.fftconvolve(x, self.ir)[: x.size + self.ir.size - 1]
        if self.tail.size:
            n = min(self.tail.size, wet_full.size)
            wet_full[:n] += self.tail[:n]
            if self.tail.size > wet_full.size:      # block shorter than the tail
                rest = self.tail[wet_full.size:]
                wet_full = np.concatenate([wet_full, rest])
        out = wet_full[: x.size]
        self.tail = wet_full[x.size:].astype(np.float32)
        w = self.cfg.reverb_wet
        return ((1.0 - w) * x + w * out).astype(np.float32)

    def _autolevel(self, x: np.ndarray) -> np.ndarray:
        """Walk the stream toward the target loudness, measured the way
        broadcast does it (K-weighted, gated) rather than by plain RMS.

        Three properties matter. It is CUMULATIVE — the estimate covers every
        gated block so far, so one quiet opening sentence cannot set the level
        for a whole script the way a first-block calibration did. It is
        RATE-LIMITED to `level_rate_db_s` per second of audio inside a total
        clamp of `level_max_db`. And the gain is RAMPED across the block
        rather than switched, because the rate limit is per second while a
        block is seconds long — applying the whole allowance at sample zero
        put a measured +2.9 dB step on a chunk seam.
        """
        # Gate: only blocks with speech in them count toward the loudness
        # estimate (BS.1770 gates at -70 LUFS to keep silence from dragging
        # the measurement down; room tone is exactly that case).
        kw, self.zi_k = signal.sosfilt(self.sos_k, x, zi=self.zi_k)
        power = float(np.mean(kw ** 2))
        if power > 1e-8:                       # ≈ -80 dB, well under speech
            self._k_power_sum += power * x.size
            self._k_samples += x.size
        start_gain = self.level_gain
        if self._k_samples > self.sr * 0.4:    # ≥400 ms measured: BS.1770 block
            mean_sq = self._k_power_sum / max(self._k_samples, 1)
            lufs = -0.691 + 10.0 * np.log10(max(mean_sq, 1e-12))
            want_db = float(np.clip(self.cfg.target_lufs - lufs,
                                    -self.cfg.level_max_db, self.cfg.level_max_db))
            cur_db = 20.0 * np.log10(max(self.level_gain, 1e-6))
            budget = self.cfg.level_rate_db_s * (x.size / self.sr)
            step = float(np.clip(want_db - cur_db, -budget, budget))
            self.level_gain = 10.0 ** ((cur_db + step) / 20.0)
        if abs(self.level_gain - start_gain) < 1e-6:
            return (x * self.level_gain).astype(np.float32)
        # RAMP the gain across the block instead of applying the new value to
        # every sample including the first. The rate limit is per SECOND of
        # audio, and a chunk can be four seconds long, so a step application
        # put the entire allowance — measured at +2.9 dB — on one sample at a
        # chunk seam. That is an audible level jump, and it is bigger than the
        # 1.5 dB per chunk that prosody.Loudness is careful never to exceed.
        # Interpolating in dB keeps the move perceptually linear.
        ramp = np.linspace(20.0 * np.log10(max(start_gain, 1e-6)),
                           20.0 * np.log10(max(self.level_gain, 1e-6)),
                           x.size, dtype=np.float32)
        return (x * (10.0 ** (ramp / 20.0))).astype(np.float32)

    def measured_lufs(self) -> Optional[float]:
        """Gated K-weighted loudness of everything processed so far, before
        the auto-level gain — None until 400 ms of speech has been seen."""
        if self._k_samples <= self.sr * 0.4:
            return None
        mean_sq = self._k_power_sum / max(self._k_samples, 1)
        return float(-0.691 + 10.0 * np.log10(max(mean_sq, 1e-12)))

    # -- public ---------------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        """Process one block in stream order. Pass the pauses through too —
        that is what lets the reverb ring into them."""
        if not self.cfg.enabled or x.size == 0:
            return x
        y = np.asarray(x, dtype=np.float32)
        y, self.zi_hp = signal.sosfilt(self.sos_hp, y, zi=self.zi_hp)
        if self.cfg.deess:
            y = self._deess(y)
        y, self.zi_shelf = signal.sosfilt(self.sos_shelf, y, zi=self.zi_shelf)
        # Gain staging BEFORE dynamics, not after. With the auto-level last,
        # the compressor saw the raw engine output at about -25 dBFS, sat
        # entirely under its -20 dB threshold and did nothing at all (audited:
        # mean gain reduction -0.00 to -0.11 dB across twelve real previews).
        # Levelling first is both the conventional order and the one that
        # makes the compressor a working part of the chain.
        y = self._autolevel(y)
        if self.cfg.comp:
            y = self._compress(y)
        if self.cfg.reverb:
            y = self._reverb(y)
        return np.asarray(y, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Whatever is still ringing when the script ends (≤55 ms)."""
        if self.tail.size == 0:
            return np.zeros(0, dtype=np.float32)
        out = (self.cfg.reverb_wet * self.tail).astype(np.float32)
        self.tail = np.zeros(0, dtype=np.float32)
        return out
