"""Slow, thorough voice cloning — the quality path.

ECAPAFlow's original experiment was smart-init ALONE: blend the top-3 built-in
voices by ECAPA similarity and ship in under a second. That is a genuinely
good trick, and on a voice close to the built-ins it lands around 50%
similarity. It also stops there, because a fixed softmax over three voices is
all it ever does — no amount of waiting improves it.

This module is the other end of that trade, per Lauro's instruction on
2026-07-27: *"Make the voice clone take a minute or two, or an hour, until
that's fully done. Don't go fast, and just do it well."*

## What it actually optimizes

The reference recording gives an ECAPA target embedding. A candidate style is
scored by SYNTHESIZING with it and measuring the ECAPA cosine of the result
against that target — the same number the app already reports as "verified
similarity". So the thing being optimized is exactly the thing being measured,
with no proxy in between.

The search space is deliberately small and well-conditioned:

    style_ttl = Σ softmax(w)_i · S_i   +   Σ α_j · P_j

where `S_i` are the ten built-in voice styles and `P_j` are the leading
principal components of those ten styles. The first term is smart-init
generalised — every voice, freely weighted, instead of top-3 at a fixed
temperature. The second is a low-rank residual that can move the style
somewhere no blend can reach, bounded so it cannot wander out of the manifold
the vocoder was trained on (which is what produces artefacts).

## Why a gradient-free search

The Colab recipe (`colab/train_paid_parity.py`) backpropagates through a torch
copy of the whole TTS stack and reaches 0.72-0.95 similarity — on a CUDA GPU.
On this laptop (no CUDA, torch CPU) one such iteration costs minutes, so the
400-1500 iterations that recipe needs is a multi-day run. An evolution
strategy over ~22 dimensions, evaluated through the existing DirectML ONNX
path, gets thousands of real evaluations in the same hour instead. Different
tool for a different machine — and it optimizes the measured objective
directly, so its progress numbers are never a proxy.

## Honesty features

* Two training utterances, one HELD-OUT utterance. The reported similarity is
  the held-out one, because an optimizer given a single sentence will happily
  overfit to that sentence's phonetics.
* The best candidate is only accepted if it beats the smart-init baseline on
  the held-out sentence. A long run that finds nothing returns the baseline
  and says so, rather than shipping a worse voice with a better-looking number.
* Every evaluation is logged with elapsed time, so the progress the UI shows
  is measurement, not an animation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

import ecapa as ecapa_mod

# Utterances used to score a candidate. Two for training, one held out — kept
# short (the objective is speaker identity, not content) and phonetically
# broad, so the optimizer cannot win by fitting one sentence's phonemes.
TRAIN_TEXTS = (
    "The quick brown fox jumps over the lazy dog while she watches.",
    "Every good voice carries its own weight, its own colour, its own rhythm.",
)
HOLDOUT_TEXT = "Nothing about this sentence appeared during the search at all."

EVAL_STEPS = 6          # synthesis steps per evaluation: enough for the
                        # speaker encoder to judge identity, cheap enough for
                        # thousands of evaluations
EVAL_SPEED = 1.05
N_COMPONENTS = 12       # low-rank residual dimensions
RESIDUAL_CAP = 0.45     # max residual norm as a fraction of the blend norm

DEPTHS = {
    # key: (label, seconds, description)
    "quick": ("Quick", 120, "about two minutes — a real search, small budget"),
    "standard": ("Standard", 900, "about fifteen minutes — the recommended one"),
    "deep": ("Deep", 3600, "an hour — for a voice you intend to keep"),
}


@dataclass
class ForgeProgress:
    """Everything the UI needs to say what is happening, in plain numbers."""
    phase: str = "starting"        # starting|smart-init|searching|verifying|done|failed|cancelled
    message: str = ""
    evals: int = 0
    generation: int = 0
    elapsed_s: float = 0.0
    budget_s: float = 0.0
    baseline_sim: float = 0.0      # smart-init, held-out
    best_sim: float = 0.0          # current best, held-out
    train_sim: float = 0.0         # current best on the training pair
    improved_pct: float = 0.0
    eta_s: float = 0.0
    error: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("elapsed_s", "budget_s", "eta_s"):
            d[k] = round(float(d[k]), 1)
        for k in ("baseline_sim", "best_sim", "train_sim", "improved_pct"):
            d[k] = round(float(d[k]) * 100.0, 2) if k != "improved_pct" else round(float(d[k]), 2)
        return d


class Cancelled(Exception):
    pass


def _style_of(engine: Any, ttl: np.ndarray, dp: np.ndarray):
    from supertonic.core import Style
    return Style(ttl.astype("float32"), dp.astype("float32"))


def _similarity(engine: Any, ttl: np.ndarray, dp: np.ndarray,
                target: np.ndarray, texts) -> float:
    """Mean ECAPA cosine of synthesized speech against the reference."""
    sims = []
    style = _style_of(engine, ttl, dp)
    for t in texts:
        wav, _dur, _infer = engine.synthesize(
            t, style, steps=EVAL_STEPS, speed=EVAL_SPEED, lang="en",
            silence_duration=0.0)
        emb = ecapa_mod._embed_wav_16k_np(wav)
        n = float(np.linalg.norm(emb))
        if n > 0:
            sims.append(float(np.dot(emb / n, target)))
    return float(np.mean(sims)) if sims else 0.0


def _basis(engine: Any) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """The ten built-in styles as a matrix, plus the PCA components used for
    the low-rank residual."""
    names = list(ecapa_mod.BUILTIN_VOICES)
    ttls, dps = [], []
    for n in names:
        st = engine.get_voice(n)
        ttls.append(np.asarray(st.ttl, dtype="float32").reshape(-1))
        dps.append(np.asarray(st.dp, dtype="float32").reshape(-1))
    S = np.stack(ttls)                      # (10, D)
    D = np.stack(dps)
    mean = S.mean(axis=0, keepdims=True)
    centred = S - mean
    # 10 voices → at most 9 meaningful directions; SVD is exact and instant
    # at this size, so there is no reason to approximate it.
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:min(N_COMPONENTS, vt.shape[0])]
    return names, S, D, comps


def _compose(S: np.ndarray, comps: np.ndarray, w: np.ndarray,
             alpha: np.ndarray, shape) -> np.ndarray:
    p = np.exp(w - w.max())
    p = p / p.sum()
    blend = p @ S
    if alpha.size:
        residual = alpha @ comps
        cap = RESIDUAL_CAP * float(np.linalg.norm(blend))
        rn = float(np.linalg.norm(residual))
        if rn > cap > 0:
            residual = residual * (cap / rn)
        blend = blend + residual
    return blend.reshape(shape).astype("float32")


def forge_voice(
    engine: Any,
    ref_audio_path: str,
    budget_s: float,
    ecapa_steps: int = 24,
    on_progress: Optional[Callable[[ForgeProgress], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    seed: int = 12345,
    progress: Optional[ForgeProgress] = None,
) -> dict:
    """Run the full slow clone. Blocks for up to `budget_s` seconds.

    Pass `progress` to have the caller's own object updated in place — that
    is how the HTTP layer serves live status without a queue: the job holds
    the object, this function writes to it, the endpoint reads it.
    """
    prog = progress or ForgeProgress()
    prog.budget_s = float(budget_s)
    prog.phase = "smart-init"
    prog.message = "Analysing the reference recording…"
    t0 = time.perf_counter()

    def emit():
        prog.elapsed_s = time.perf_counter() - t0
        prog.eta_s = max(0.0, prog.budget_s - prog.elapsed_s)
        if on_progress:
            try:
                on_progress(prog)
            except Exception:
                pass

    def check():
        if should_cancel and should_cancel():
            raise Cancelled()

    emit()

    # ---- 1. smart init: the starting point and the number to beat ----------
    init = ecapa_mod.smart_init_clone(engine, ref_audio_path, ecapa_steps=ecapa_steps)
    ttl0 = np.asarray(init["ttl"], dtype="float32")
    dp0 = np.asarray(init["dp"], dtype="float32")
    target = np.asarray(init["target_emb"], dtype="float32")
    target = target / max(1e-9, float(np.linalg.norm(target)))
    shape = ttl0.shape
    check()

    prog.phase = "searching"
    prog.message = "Measuring the starting point…"
    emit()

    names, S, Dm, comps = _basis(engine)
    # Seed the weights from smart-init's own ranking so generation 0 IS the
    # smart-init blend — the search can only go up from there.
    rank = {r["voice"]: r["sim"] for r in init["ranking"]}
    w0 = np.array([rank.get(n, 0.0) for n in names], dtype="float32")
    w0 = (w0 - w0.mean()) / max(1e-6, w0.std()) * 2.0
    a0 = np.zeros(comps.shape[0], dtype="float32")

    baseline_holdout = _similarity(engine, ttl0, dp0, target, (HOLDOUT_TEXT,))
    prog.baseline_sim = baseline_holdout
    prog.best_sim = baseline_holdout
    prog.evals += 1
    emit()
    check()

    best_ttl = ttl0.copy()
    best_train = _similarity(engine, ttl0, dp0, target, TRAIN_TEXTS)
    prog.train_sim = best_train
    prog.evals += 1
    best_w, best_a = w0.copy(), a0.copy()
    emit()

    rng = np.random.default_rng(seed)
    sigma_w, sigma_a = 0.6, 0.25
    lam = 6                       # children per generation
    gen = 0
    since_improve = 0

    # ---- 2. the search ----------------------------------------------------
    while (time.perf_counter() - t0) < budget_s:
        check()
        gen += 1
        prog.generation = gen
        improved = False
        for _ in range(lam):
            if (time.perf_counter() - t0) >= budget_s:
                break
            check()
            w = best_w + rng.standard_normal(best_w.shape).astype("float32") * sigma_w
            a = best_a + rng.standard_normal(best_a.shape).astype("float32") * sigma_a
            ttl = _compose(S, comps, w, a, shape)
            s = _similarity(engine, ttl, dp0, target, TRAIN_TEXTS)
            prog.evals += 1
            if s > best_train:
                best_train, best_w, best_a, best_ttl = s, w, a, ttl
                improved = True
                prog.train_sim = best_train
            prog.message = (f"Generation {gen}, {prog.evals} candidates tried — "
                            f"best match {best_train * 100:.1f}%")
            emit()

        # 1/5th-success rule: widen the search while it keeps paying off,
        # tighten it when it stops. Classic ES step-size control, and the
        # reason this converges instead of wandering.
        if improved:
            sigma_w *= 1.15
            sigma_a *= 1.15
            since_improve = 0
        else:
            sigma_w *= 0.82
            sigma_a *= 0.82
            since_improve += 1
        sigma_w = float(np.clip(sigma_w, 0.02, 1.5))
        sigma_a = float(np.clip(sigma_a, 0.005, 0.8))
        # Converged: the step size has collapsed and nothing has improved for
        # a while. Stop early rather than burn the rest of an hour.
        if since_improve >= 8 and sigma_w <= 0.05:
            prog.message = "Converged — no further improvement available."
            break

    # ---- 3. honest verification on the held-out sentence ------------------
    prog.phase = "verifying"
    prog.message = "Checking the result on a sentence it never saw…"
    emit()
    check()
    cand_holdout = _similarity(engine, best_ttl, dp0, target, (HOLDOUT_TEXT,))
    prog.evals += 1

    accepted = cand_holdout > baseline_holdout
    final_ttl = best_ttl if accepted else ttl0
    final_holdout = cand_holdout if accepted else baseline_holdout
    prog.best_sim = final_holdout
    prog.improved_pct = (final_holdout - baseline_holdout) * 100.0
    prog.phase = "done"
    prog.message = (
        f"Done — {final_holdout * 100:.1f}% match "
        f"({'+' if prog.improved_pct >= 0 else ''}{prog.improved_pct:.1f} points "
        f"over the instant clone)" if accepted else
        f"Done — the search found nothing better than the instant clone "
        f"({baseline_holdout * 100:.1f}%), so that is what was kept.")
    emit()

    return {
        "ttl": final_ttl,
        "dp": dp0,
        "target_emb": target,
        "ranking": init["ranking"],
        "accepted_search_result": bool(accepted),
        "baseline_sim_pct": round(baseline_holdout * 100.0, 2),
        "final_sim_pct": round(final_holdout * 100.0, 2),
        "train_sim_pct": round(best_train * 100.0, 2),
        "evals": prog.evals,
        "generations": gen,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "budget_s": float(budget_s),
        "est_sim_pct": round(final_holdout * 100.0, 2),
        "effective_steps": init.get("effective_steps"),
    }


# --------------------------------------------------------------------- jobs
class ForgeJob:
    """One running clone. Owns its thread, its progress and its cancel flag."""

    def __init__(self, voice_id: str, depth: str, budget_s: float) -> None:
        self.voice_id = voice_id
        self.depth = depth
        self.budget_s = budget_s
        self.progress = ForgeProgress(budget_s=budget_s, phase="queued",
                                      message="Waiting for the engine…")
        self.result: Optional[dict] = None
        self.error = ""
        self._cancel = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.started_at = time.time()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def as_dict(self) -> dict:
        d = self.progress.as_dict()
        d.update(voice_id=self.voice_id, depth=self.depth,
                 depth_label=DEPTHS.get(self.depth, ("Custom",))[0],
                 error=self.error, started_at=self.started_at)
        return d


_JOBS: dict[str, ForgeJob] = {}
_JOBS_LOCK = threading.Lock()


def get_job(voice_id: str) -> Optional[ForgeJob]:
    with _JOBS_LOCK:
        return _JOBS.get(voice_id)


def active_jobs() -> list[dict]:
    with _JOBS_LOCK:
        return [j.as_dict() for j in _JOBS.values()
                if j.progress.phase not in ("done", "failed", "cancelled")]


def start_job(engine: Any, voice: dict, ref_path: str, depth: str,
              ecapa_steps: int, on_done: Callable[[ForgeJob], None]) -> ForgeJob:
    """Start (or return the already-running) clone for this voice."""
    vid = voice["id"]
    with _JOBS_LOCK:
        existing = _JOBS.get(vid)
        if existing and existing.progress.phase not in ("done", "failed", "cancelled"):
            return existing
        label, budget, _desc = DEPTHS.get(depth, DEPTHS["standard"])
        job = ForgeJob(vid, depth, float(budget))
        _JOBS[vid] = job

    def _run():
        try:
            job.progress.phase = "starting"
            res = forge_voice(
                engine, ref_path, job.budget_s, ecapa_steps=ecapa_steps,
                should_cancel=lambda: job.cancelled,
                progress=job.progress,          # updated in place, read by HTTP
            )
            job.result = res
        except Cancelled:
            job.progress.phase = "cancelled"
            job.progress.message = "Cancelled — the voice was left as it was."
        except Exception as e:  # noqa: BLE001
            job.error = f"{type(e).__name__}: {e}"
            job.progress.phase = "failed"
            job.progress.message = f"Cloning failed: {job.error}"
        finally:
            try:
                on_done(job)
            except Exception:
                pass

    t = threading.Thread(target=_run, name=f"forge-{vid[:8]}", daemon=True)
    job.thread = t
    t.start()
    return job
