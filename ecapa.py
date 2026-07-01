"""ECAPA smart-init voice cloning for ECAPAFlow.

This is the LoudFlow clone.py "quick" path isolated as a standalone product:
ECAPA smart-init ONLY — no gradient polish afterwards. The experiment: does
the smart-init blend alone produce usable quality at a fraction of the time
(target: under 1 second once models are warm)?

Pipeline per clone:
  1. Quality-gate the reference audio (same thresholds as LoudFlow clone.py).
  2. Build the target speaker embedding from the RAW reference (silence-trimmed
     but NOT gain-normalized — the paid_parity lesson: gain amplifies the noise
     floor and drifts the ECAPA embedding away from the true speaker).
  3. "ECAPA encoder steps": embed the reference in N evenly-spaced 3s windows
     (one batched encoder forward) and average → the more steps, the more of
     the reference the encoder actually looks at, the more robust the target.
     steps=1 falls back to a single whole-clip pass.
  4. Score the 10 built-in Supertonic voices by cosine similarity to the
     target (their embeddings are synthesized once and cached to disk).
  5. Softmax-blend the top-3 voices' style_ttl + style_dp (temperature 0.1)
     → the cloned voice style. Saved as a Supertonic-compatible style JSON.

Module-level caches survive across clones: the ECAPA encoder (~22M params)
and the built-in voice embedding table (data/cache/voice_embeddings.npz).
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BUILTIN_VOICES = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
EMB_CACHE_PATH = CACHE_DIR / "voice_embeddings.npz"

TOP_K = 3
SOFTMAX_TEMPERATURE = 0.1
WINDOW_S = 3.0          # ECAPA analysis window per encoder step
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog near the river."

# Reference-quality thresholds (same policy as LoudFlow clone.py: permissive
# on level, strict only on truly-silent / truly-short clips).
REF_MIN_DURATION_S = 1.5
REF_MAX_DURATION_S = 60.0
REF_MIN_PEAK = 0.005
REF_MIN_RMS = 0.001
REF_FLATNESS_WARN = 0.50
REF_QUIET_PEAK = 0.10
REF_WEAK_RMS = 0.015

_lock = threading.Lock()

_state: dict[str, Any] = {
    "available": False,
    "reason": "not loaded yet",
    "loading": False,
    "device": "",
    "emb_cache_ready": False,
    "emb_cache_building": False,
    "log": [],
}

_cache: dict[str, Any] = {
    "encoder": None,       # speechbrain EncoderClassifier
    "voice_embs": None,    # {name: (192,) float32}
}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    _state["log"].append(line)
    if len(_state["log"]) > 200:
        _state["log"] = _state["log"][-200:]
    try:
        print("[ecapa]", msg, flush=True)
    except UnicodeEncodeError:
        print("[ecapa]", msg.encode("ascii", "replace").decode("ascii"), flush=True)


def status() -> dict:
    return {k: v for k, v in _state.items() if k != "log"} | {"log": list(_state["log"][-30:])}


def _patch_speechbrain_lazy_module() -> None:
    """Make speechbrain LazyModule.__getattr__ raise AttributeError, not
    ImportError. Windows safety net — see LoudFlow clone.py for the full
    story (librosa's lazy loader probes sys.modules with hasattr, which only
    swallows AttributeError; speechbrain's optional deps like k2/numba are
    not installable on Windows and would blow that probe up)."""
    try:
        import speechbrain  # noqa: F401
        import speechbrain.utils.importutils as _su
        if getattr(_su.LazyModule.__getattr__, "_patched_for_windows", False):
            return
        _orig = _su.LazyModule.__getattr__

        def _safe(self, attr):
            try:
                return _orig(self, attr)
            except ImportError as e:
                raise AttributeError(
                    f"speechbrain lazy submodule unavailable: {attr!r} ({e})"
                ) from e

        _safe._patched_for_windows = True
        _su.LazyModule.__getattr__ = _safe
    except Exception:
        pass


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_encoder() -> Any:
    """Load (once) the ECAPA-TDNN speaker encoder.

    speechbrain/spkrec-ecapa-voxceleb, Apache 2.0, 22.3M params.
    """
    if _cache["encoder"] is not None:
        return _cache["encoder"]
    with _lock:
        if _cache["encoder"] is not None:
            return _cache["encoder"]
        _state["loading"] = True
        try:
            _patch_speechbrain_lazy_module()
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            device = _pick_device()
            t0 = time.time()
            log(f"loading ECAPA encoder (speechbrain/spkrec-ecapa-voxceleb) on {device}…")
            enc = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(CACHE_DIR / "spkrec-ecapa-voxceleb"),
                run_opts={"device": device},
            )
            enc.eval()
            for p in enc.parameters():
                p.requires_grad_(False)
            _cache["encoder"] = enc
            _state["available"] = True
            _state["reason"] = ""
            _state["device"] = device
            log(f"ECAPA encoder ready in {time.time()-t0:.1f}s (device={device})")
            return enc
        except Exception as e:
            _state["available"] = False
            _state["reason"] = f"{type(e).__name__}: {e}"
            log(f"ECAPA load FAILED: {_state['reason']}")
            raise
        finally:
            _state["loading"] = False


def start_background_load() -> None:
    def _go():
        try:
            load_encoder()
        except Exception:
            pass
    threading.Thread(target=_go, name="ecapa-load", daemon=True).start()


# ---------------------------------------------------------------- audio utils
def analyze_ref_audio(input_path: str) -> dict:
    """Quality stats + go/no-go verdict for a reference clip (pre-clone gate)."""
    try:
        import soundfile as sf
        from numpy.fft import rfft

        if not Path(input_path).exists():
            return {"ok": False, "reason": f"file not found: {input_path}"}

        data, sr = sf.read(input_path, dtype="float32", always_2d=True)
        channels = int(data.shape[1])
        mono = data.mean(axis=1) if channels > 1 else data[:, 0]
        dur = float(len(mono) / sr) if sr > 0 else 0.0
        peak = float(np.abs(mono).max()) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0

        flat = 0.0
        N = min(len(mono), 65536)
        if N > 256:
            x = mono[:N] - mono[:N].mean()
            spec = np.abs(rfft(x * np.hanning(N))) + 1e-10
            flat = float(np.exp(np.mean(np.log(spec[1:]))) / np.mean(spec[1:]))

        def _dbfs(v: float) -> float:
            return float(20.0 * np.log10(max(v, 1e-9)))

        stats = {
            "duration_s": round(dur, 2),
            "sample_rate": int(sr),
            "channels": channels,
            "peak": round(peak, 4),
            "rms": round(rms, 5),
            "peak_dbfs": round(_dbfs(peak), 1),
            "rms_dbfs": round(_dbfs(rms), 1),
            "flatness": round(flat, 3),
        }

        if dur < REF_MIN_DURATION_S:
            return {**stats, "ok": False,
                    "reason": f"reference too short ({dur:.1f}s); need at least {REF_MIN_DURATION_S:.0f}s of clean speech"}
        if dur > REF_MAX_DURATION_S:
            return {**stats, "ok": False,
                    "reason": f"reference too long ({dur:.1f}s); cap is {REF_MAX_DURATION_S:.0f}s (30s recommended)"}
        if peak < REF_MIN_PEAK:
            return {**stats, "ok": False,
                    "reason": f"reference effectively silent (peak {peak:.4f} = {_dbfs(peak):.0f} dBFS)"}
        if rms < REF_MIN_RMS:
            return {**stats, "ok": False,
                    "reason": f"reference effectively silent (RMS {rms:.5f} = {_dbfs(rms):.0f} dBFS)"}

        warnings_list = []
        if peak < REF_QUIET_PEAK:
            warnings_list.append(f"quiet reference (peak {_dbfs(peak):.0f} dBFS)")
        if rms < REF_WEAK_RMS:
            warnings_list.append(f"weak signal (RMS {_dbfs(rms):.0f} dBFS)")
        if flat > REF_FLATNESS_WARN:
            warnings_list.append(f"high spectral flatness ({flat:.2f}) — clip may be noisy")
        return {**stats, "ok": True, "reason": "", "warning": " | ".join(warnings_list)}
    except Exception as e:
        return {"ok": False, "reason": f"failed to analyze reference audio: {type(e).__name__}: {e}"}


def _load_ref_16k(audio_path: str) -> "Any":
    """Load reference → mono 16 kHz torch tensor (1, T).

    Silence-trimmed (top_db=30) but NOT gain-normalized: the ECAPA target is
    built from raw amplitudes (paid_parity behaviour). Capped at 30s.
    """
    import torch
    import torchaudio

    wav_t, sr = torchaudio.load(audio_path)
    if wav_t.shape[0] > 1:
        wav_t = wav_t.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav_t = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav_t)

    # Trim leading/trailing silence with librosa (numpy round-trip is cheap).
    try:
        import librosa
        y = wav_t.squeeze(0).numpy()
        y_trim, _ = librosa.effects.trim(y, top_db=30)
        if len(y_trim) >= int(0.5 * 16000):
            wav_t = torch.tensor(y_trim, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        log(f"silence trim skipped ({type(e).__name__}: {e})")

    max_n = int(30 * 16000)
    if wav_t.shape[-1] > max_n:
        wav_t = wav_t[:, :max_n]
    return wav_t


def embed_reference(audio_path: str, steps: int = 8) -> tuple[np.ndarray, int]:
    """Build the target speaker embedding with N ECAPA encoder steps.

    steps=1 → one pass over the whole (trimmed, ≤30s) clip.
    steps=N → N windows of WINDOW_S seconds, start positions evenly spaced
    across the clip, run as ONE batched encoder forward, then averaged.
    Windows whose start positions would coincide (clip shorter than the
    requested coverage) are deduplicated, so the effective step count can be
    lower for short clips — it is returned alongside the embedding.

    Returns (embedding (192,) float32 L2-normalized, effective_steps).
    """
    import torch

    enc = load_encoder()
    wav_t = _load_ref_16k(audio_path)   # (1, T) on CPU
    T = wav_t.shape[-1]
    win = int(WINDOW_S * 16000)
    steps = max(1, int(steps))

    if steps == 1 or T <= win:
        with torch.no_grad():
            emb = enc.encode_batch(wav_t.to(enc.device)).squeeze()
        eff = 1
    else:
        starts = np.linspace(0, T - win, num=steps)
        starts = sorted(set(int(round(s)) for s in starts))
        batch = torch.stack([wav_t[0, s:s + win] for s in starts], dim=0)  # (N, win)
        with torch.no_grad():
            embs = enc.encode_batch(batch.to(enc.device))  # (N, 1, 192)
        emb = embs.reshape(len(starts), -1).mean(dim=0)
        eff = len(starts)

    e = emb.detach().cpu().numpy().astype("float32").reshape(-1)
    e = e / max(1e-9, float(np.linalg.norm(e)))
    return e, eff


def _embed_wav_16k_np(wav_44k_1d: np.ndarray) -> np.ndarray:
    """Resample a synthesized 44.1k wav → 16k, single ECAPA pass → (192,)."""
    import torch
    import torchaudio

    enc = load_encoder()
    wav_t = torch.tensor(np.ascontiguousarray(wav_44k_1d), dtype=torch.float32).unsqueeze(0)
    wav_16k = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(wav_t)
    with torch.no_grad():
        emb = enc.encode_batch(wav_16k.to(enc.device)).squeeze()
    e = emb.detach().cpu().numpy().astype("float32").reshape(-1)
    return e


# ------------------------------------------------------- built-in voice table
def build_voice_embedding_cache(engine: Any, force: bool = False) -> dict:
    """ECAPA embeddings for the 10 built-in voices (synthesize sample → embed).

    Cached to disk; subsequent calls load in ~10 ms. `engine` is the
    SupertonicEngine singleton (must be loaded)."""
    if _cache["voice_embs"] is not None and not force:
        return _cache["voice_embs"]

    cached: dict[str, np.ndarray] = {}
    if EMB_CACHE_PATH.exists() and not force:
        try:
            with np.load(EMB_CACHE_PATH, allow_pickle=False) as data:
                cached = {k: data[k].astype("float32") for k in data.files}
            log(f"loaded {len(cached)} cached built-in voice embeddings")
        except Exception as e:
            log(f"emb cache load failed ({e}) — rebuilding")
            cached = {}

    missing = [v for v in BUILTIN_VOICES if v not in cached]
    if missing:
        _state["emb_cache_building"] = True
        log(f"building ECAPA embeddings for {len(missing)} built-in voices (one-time)…")
        try:
            for vname in missing:
                t0 = time.time()
                wav, _dur, _infer = engine.synthesize(
                    SAMPLE_TEXT, vname, steps=4, speed=1.05, lang="en",
                    silence_duration=0.3,
                )
                cached[vname] = _embed_wav_16k_np(wav)
                log(f"  {vname}: embedded ({time.time()-t0:.1f}s)")
            np.savez(EMB_CACHE_PATH, **cached)
            log(f"saved voice embedding cache → {EMB_CACHE_PATH.name}")
        finally:
            _state["emb_cache_building"] = False

    _cache["voice_embs"] = cached
    _state["emb_cache_ready"] = len(cached) >= len(BUILTIN_VOICES)
    return cached


# ----------------------------------------------------------------- smart init
def smart_init_clone(
    engine: Any,
    ref_audio_path: str,
    ecapa_steps: int = 8,
    top_k: int = TOP_K,
    temperature: float = SOFTMAX_TEMPERATURE,
) -> dict:
    """The whole ECAPAFlow experiment in one function: reference audio in,
    blended voice style out. No gradient training.

    Returns dict with:
      ttl, dp            — blended style arrays (1,50,256) / (1,8,16) float32
      target_emb         — (192,) float32 (L2-normalized)
      ranking            — [{voice, sim, weight}] all 10, sorted desc
      est_sim_pct        — blend-weighted expected similarity (instant estimate)
      effective_steps    — deduplicated encoder step count actually run
      timings_ms         — {embed, rank, blend, total}
    """
    t_total = time.perf_counter()

    voice_embs = build_voice_embedding_cache(engine)
    if not voice_embs:
        raise RuntimeError("no built-in voice embeddings available — smart init impossible")

    t0 = time.perf_counter()
    target, eff_steps = embed_reference(ref_audio_path, steps=ecapa_steps)
    embed_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sims: list[tuple[str, float]] = []
    for v, emb in voice_embs.items():
        e = emb / max(1e-9, float(np.linalg.norm(emb)))
        sims.append((v, float(np.dot(target, e))))
    sims.sort(key=lambda x: -x[1])
    rank_ms = (time.perf_counter() - t0) * 1000

    top = sims[:top_k]
    s_arr = np.array([s for _, s in top], dtype="float32")
    s_shift = s_arr - s_arr.max()
    w = np.exp(s_shift / max(1e-6, temperature))
    w = w / w.sum()

    t0 = time.perf_counter()
    ttl_blend: Optional[np.ndarray] = None
    dp_blend: Optional[np.ndarray] = None
    for (vname, _), weight in zip(top, w):
        style = engine.get_voice(vname)
        if ttl_blend is None:
            ttl_blend = style.ttl.astype("float32") * float(weight)
            dp_blend = style.dp.astype("float32") * float(weight)
        else:
            ttl_blend = ttl_blend + style.ttl.astype("float32") * float(weight)
            dp_blend = dp_blend + style.dp.astype("float32") * float(weight)
    blend_ms = (time.perf_counter() - t0) * 1000

    ranking = [
        {"voice": v, "sim": round(s, 4),
         "weight": round(float(w[i]), 4) if i < len(w) else None}
        for i, (v, s) in enumerate(sims)
    ]
    est_sim = float(np.dot(w, s_arr))
    total_ms = (time.perf_counter() - t_total) * 1000

    log(
        f"smart-init: steps={eff_steps}  top-{top_k} "
        + ", ".join(f"{v}×{w[i]:.2f}" for i, (v, _) in enumerate(top))
        + f"  est_sim={est_sim*100:.0f}%  "
        f"({embed_ms:.0f}ms embed + {rank_ms:.1f}ms rank + {blend_ms:.1f}ms blend = {total_ms:.0f}ms)"
    )

    return {
        "ttl": ttl_blend,
        "dp": dp_blend,
        "target_emb": target,
        "ranking": ranking,
        "est_sim_pct": round(max(0.0, min(1.0, est_sim)) * 100.0, 1),
        "effective_steps": eff_steps,
        "requested_steps": int(ecapa_steps),
        "timings_ms": {
            "embed": round(embed_ms, 1),
            "rank": round(rank_ms, 2),
            "blend": round(blend_ms, 2),
            "total": round(total_ms, 1),
        },
    }


def verify_clone(engine: Any, ttl: np.ndarray, dp: np.ndarray,
                 target_emb: np.ndarray) -> dict:
    """Measure ACHIEVED similarity: synthesize a sample with the blended style
    and ECAPA-compare it to the target. Costs one synthesis (+1 embed), so it
    runs after cloning, not inside the <1s clone path."""
    from supertonic.core import Style

    t0 = time.perf_counter()
    style = Style(ttl, dp)
    wav, _dur, _infer = engine.synthesize(
        SAMPLE_TEXT, style, steps=8, speed=1.05, lang="en", silence_duration=0.3,
    )
    emb = _embed_wav_16k_np(wav)
    emb = emb / max(1e-9, float(np.linalg.norm(emb)))
    cos = float(np.dot(emb, target_emb / max(1e-9, float(np.linalg.norm(target_emb)))))
    return {
        "verified_sim": round(cos, 4),
        "verified_sim_pct": round(max(0.0, min(1.0, cos)) * 100.0, 1),
        "verify_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------- style files
def save_style_json(path: str, ttl: np.ndarray, dp: np.ndarray, metadata: dict) -> None:
    """Write a Supertonic-compatible voice style JSON (same schema the pip
    package's get_voice_style_from_path loads)."""
    ttl = np.asarray(ttl, dtype=np.float32).reshape(1, 50, 256)
    dp = np.asarray(dp, dtype=np.float32).reshape(1, 8, 16)
    payload = {
        "style_ttl": {
            "data": ttl.flatten().tolist(),
            "dims": [1, 50, 256],
            "type": "float32",
        },
        "style_dp": {
            "data": dp.flatten().tolist(),
            "dims": [1, 8, 16],
            "type": "float32",
        },
        "metadata": metadata,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
