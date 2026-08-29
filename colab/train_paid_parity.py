"""Headless paid_parity voice-clone trainer for Supertonic 3 (Colab / CUDA ready).

1:1 port of LoudFlow's production training job (supertonic3_test/clone.py
`_train_job`) stripped of the FastAPI/threading/UI-state machinery so it runs
top-to-bottom on a Colab A100 (or CPU, slowly).

Recipe (paid_parity family — verified against clone.py):
  * vocoder_steps=5 during training (matches runtime step distribution)
  * ECAPA target embedding from the RAW reference (no gain normalization),
    multi-window averaged (3.0 s window, 1.5 s hop)
  * smart-init: top-3 softmax blend (temperature 0.1) over the 10 built-in
    voices ranked by ECAPA cosine to the target
  * AdamW on style_ttl ONLY (style_dp fixed from the smart-init blend),
    weight_decay=0, grad_clip 1.0, seed 749
  * base_lr 0.004 (400/120-iter modes) / 0.003 (1000+ modes)
  * schedule: linear warmup -> linear decay to 20% (400/120-iter modes) or
    warmup + ReduceLROnPlateau(patience=200, factor=0.5) (1000+ modes)
  * paid_parity_clean: projected gradient descent — after each step,
    ||style_ttl|| is capped at the smart-init norm
  * fixed noisy latent sampled once per run; dataloader cycles configs.texts
  * best-loss tracking, early stop at ECAPA deficit <= 0.05 (~95% sim)
  * save via utils.save_style + clone.py-compatible v3 metadata

Depends ONLY on the voice_clone_repo (helper.py ONNX TTS + utils/*), never on
the `supertonic` pip package.

Usage:
    from train_paid_parity import train_voice
    result = train_voice("ref.wav", "out.json", mode="paid_parity",
                         model_dir="/content/supertonic3",
                         repo_dir="/content/voice_clone_repo")

CLI:
    python train_paid_parity.py ref.wav out.json --mode paid_parity \
        --model-dir /content/supertonic3 --repo-dir /content/voice_clone_repo
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# Windows consoles default to cp1252 and choke on the emoji some repo modules
# print. Replace unencodable characters instead of crashing. No-op on Linux.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Defaults (Colab layout). Override via args or env vars.
# ---------------------------------------------------------------------------
DEFAULT_REPO_DIR = os.environ.get("VOICE_CLONE_REPO", "/content/voice_clone_repo")
DEFAULT_MODEL_DIR = os.environ.get("SUPERTONIC_MODEL_DIR", "/content/supertonic3")

SEED = 749
SPEED = 1.05
VOCODER_STEPS = 5           # all paid_parity modes train at the runtime step count
GRAD_CLIP = 1.0
DEFAULT_EARLY_STOP_LOSS = 0.05   # ECAPA deficit; 0.05 ~ 95% similarity
BUILTIN_VOICES = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")

# mode -> num_steps (all use smart init). Mirrors clone.py MODE_PRESETS
# (paid_parity family only; paid_parity_spec is excluded — it needs
# spectral_loss.py which is not part of voice_clone_repo).
MODE_PRESETS: dict[str, int] = {
    "paid_parity":       400,   # base_lr 0.004, linear schedule
    "paid_parity_clean": 400,   # + norm-cap projected GD
    "paid_parity_fast":  120,   # preview tier, same clean recipe
    "paid_parity_long":  1000,  # base_lr 0.003, ReduceLROnPlateau
    "paid_parity_xlong": 1800,  # base_lr 0.003, ReduceLROnPlateau
    "paid_parity_max":   2000,  # base_lr 0.003, ReduceLROnPlateau
}
_PLATEAU_MODES = {"paid_parity_long", "paid_parity_xlong", "paid_parity_max"}


def _log(msg: str) -> None:
    try:
        print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    except UnicodeEncodeError:
        print("[train]", msg.encode("ascii", "replace").decode("ascii"), flush=True)


# ---------------------------------------------------------------------------
# Environment guards (harmless on Linux, required on Windows)
# ---------------------------------------------------------------------------
def _patch_speechbrain_lazy_module() -> None:
    """ImportError -> AttributeError inside speechbrain LazyModule.__getattr__.

    Windows-only safety net (optional speechbrain subpackages like k2/numba are
    not installable there and break hasattr probes from librosa's lazy loader).
    On Linux/Colab this is a no-op that never fires. Idempotent."""
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


def _ensure_torchaudio_load() -> None:
    """Guarantee a working torchaudio.load.

    torchaudio >= 2.9 delegates load() to the optional `torchcodec` package;
    when torchcodec is missing (or ABI-incompatible with the installed torch),
    torchaudio.load raises. The repo's utils/loss.py depends on it, so probe
    by actually decoding a tiny wav and, on failure, monkeypatch
    torchaudio.load with a soundfile-based equivalent (returns (C, T) float32
    tensor + sample rate, same contract). Harmless no-op where torchaudio.load
    already works. Idempotent."""
    import tempfile

    import numpy as np
    import torch
    import torchaudio
    if getattr(torchaudio.load, "_sf_fallback", False):
        return
    probe = None
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            probe = fh.name
        sf.write(probe, np.zeros(160, dtype="float32"), 16000)
        torchaudio.load(probe)
        return  # native load works
    except Exception:
        pass
    finally:
        if probe is not None:
            try:
                os.unlink(probe)
            except OSError:
                pass

    import soundfile as sf

    def _sf_load(filepath, *args, **kwargs):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), int(sr)

    _sf_load._sf_fallback = True
    torchaudio.load = _sf_load
    _log("torchaudio.load unavailable (torchcodec missing?) — patched with "
         "soundfile fallback")


def _setup_repo(repo_dir: str) -> None:
    repo_dir = str(Path(repo_dir).resolve())
    if not (Path(repo_dir) / "helper.py").exists():
        raise RuntimeError(f"voice_clone_repo not found at {repo_dir} "
                           f"(expected helper.py inside)")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    _patch_speechbrain_lazy_module()
    _ensure_torchaudio_load()


# ---------------------------------------------------------------------------
# Heavy-object cache (reused across train_voice calls, e.g. batch mode)
# ---------------------------------------------------------------------------
_CACHE: dict[str, Any] = {
    "model": None,          # utils.SupertonicModel (4 PyTorch nets + SpeakerID)
    "model_dir": None,      # onnx dir the cached model was built from
    "ref_audio_path": None, # reference currently loaded into the encoder
    "tts": None,            # helper.TextToSpeech (ONNX, CPU)
    "dataloader": None,
}


def _multi_window_embed_target(speaker_encoder, audio_path: str,
                               window_s: float = 3.0, hop_s: float = 1.5):
    """Average ECAPA embeddings over overlapping windows of the reference.
    Returns a torch tensor (192,) on the encoder's device. Port of clone.py."""
    import torch
    import torchaudio
    wav_t, sr = torchaudio.load(audio_path)
    if wav_t.shape[0] > 1:
        wav_t = wav_t.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav_t = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav_t)
    T = wav_t.shape[-1]
    win = int(window_s * 16000)
    hop = int(hop_s * 16000)
    if T <= win:
        with torch.no_grad():
            return speaker_encoder.embed_wav(wav_t).detach()
    embs = []
    for start in range(0, T - win + 1, hop):
        chunk = wav_t[:, start:start + win]
        try:
            with torch.no_grad():
                embs.append(speaker_encoder.embed_wav(chunk).detach())
        except Exception:
            continue
    if not embs:
        with torch.no_grad():
            return speaker_encoder.embed_wav(wav_t).detach()
    avg = torch.stack(embs, dim=0).mean(dim=0)
    _log(f"multi-window target embedding: averaged {len(embs)} windows of {window_s:.1f}s")
    return avg


def _retarget_encoder(model, ref_audio_path: str) -> None:
    """(Re-)embed the reference into the model's ECAPA encoder, multi-window."""
    import torch
    sb_enc = model.voice_encoder
    sb_enc.target_wav_path = ref_audio_path
    sb_enc.emb_original = _multi_window_embed_target(sb_enc, ref_audio_path)
    sb_enc.norm_emb_original = torch.nn.functional.normalize(
        sb_enc.emb_original, p=2, dim=0
    )


def _ensure_model(onnx_dir: str, ref_audio_path: str):
    """Load (or re-target) SupertonicModel; PyTorch conversion cached."""
    from utils import SupertonicModel
    if _CACHE["model"] is not None and _CACHE["model_dir"] == onnx_dir:
        if _CACHE["ref_audio_path"] != ref_audio_path:
            _log("re-targeting cached ECAPA encoder to new reference (multi-window)")
            _retarget_encoder(_CACHE["model"], ref_audio_path)
            _CACHE["ref_audio_path"] = ref_audio_path
        return _CACHE["model"]
    _log("converting ONNX -> PyTorch (one-time; ~30-60s on GPU box, 60-180s CPU)...")
    t0 = time.time()
    model = SupertonicModel(onnx_dir, ref_audio_path)
    _log(f"converted in {time.time()-t0:.1f}s (cached for subsequent clones)")
    # Replace the single-shot embedding from __init__ with the multi-window average.
    try:
        _retarget_encoder(model, ref_audio_path)
    except Exception as e:
        _log(f"multi-window embed failed ({e}) — falling back to single-shot")
    _CACHE["model"] = model
    _CACHE["model_dir"] = onnx_dir
    _CACHE["ref_audio_path"] = ref_audio_path
    return model


def _ensure_tts(onnx_dir: str):
    from helper import load_text_to_speech
    if _CACHE["tts"] is None:
        _log("loading ONNX TTS (CPU, used for latent sampling + smart-init samples)...")
        _CACHE["tts"] = load_text_to_speech(onnx_dir)
    return _CACHE["tts"]


def _ensure_dataloader(tts):
    from utils import get_train_dataloader
    from configs import texts
    if _CACHE["dataloader"] is None:
        _CACHE["dataloader"] = get_train_dataloader(tts, texts)
    return _CACHE["dataloader"]


# ---------------------------------------------------------------------------
# Smart init (top-3 softmax blend over built-in voices)
# ---------------------------------------------------------------------------
def _generate_voice_sample(onnx_tts, voice_style,
                           text: str = "The quick brown fox jumps over the lazy dog near the river."):
    import numpy as np
    wav, _ = onnx_tts(text=text, lang="en", style=voice_style, total_step=4, speed=SPEED)
    wav_1d = wav.squeeze() if wav.ndim > 1 else wav
    if wav_1d.dtype != np.float32:
        wav_1d = wav_1d.astype(np.float32)
    return wav_1d


def _embed_wav_16k(wav_44k_1d, speaker_encoder):
    import torch
    import torchaudio
    wav_t = torch.tensor(wav_44k_1d, dtype=torch.float32).unsqueeze(0)
    wav_16k = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(wav_t)
    with torch.no_grad():
        emb = speaker_encoder.embed_wav(wav_16k)
    return emb.detach().cpu().numpy().astype("float32")


def _build_voice_embedding_cache(onnx_tts, voice_styles_dir: str, speaker_encoder,
                                 cache_path: Optional[Path]) -> dict:
    import numpy as np
    from helper import load_voice_style as load_voice_styles

    cached: dict = {}
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cached = {k: data[k].astype("float32") for k in data.files}
            _log(f"loaded {len(cached)} cached built-in voice embeddings")
        except Exception as e:
            _log(f"emb cache load failed ({e}) — rebuilding")
            cached = {}

    missing = [v for v in BUILTIN_VOICES if v not in cached]
    if not missing:
        return cached

    _log(f"building ECAPA embeddings for {len(missing)} built-in voices (one-time)...")
    for vname in missing:
        vpath = os.path.join(voice_styles_dir, f"{vname}.json")
        if not os.path.exists(vpath):
            _log(f"  {vname}: style file missing -> skipped")
            continue
        try:
            t0 = time.time()
            style = load_voice_styles([vpath])
            wav = _generate_voice_sample(onnx_tts, style)
            cached[vname] = _embed_wav_16k(wav, speaker_encoder)
            _log(f"  {vname}: embedded ({time.time()-t0:.1f}s)")
        except Exception as e:
            _log(f"  {vname}: failed ({type(e).__name__}: {e})")

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, **cached)
            _log(f"saved voice embedding cache -> {cache_path}")
        except Exception as e:
            _log(f"emb cache save failed: {e}")
    return cached


def _smart_blend_init(onnx_tts, voice_styles_dir: str, speaker_encoder,
                      cache_path: Optional[Path],
                      top_k: int = 3, temperature: float = 0.1):
    """Rank built-in voices by ECAPA cosine to the target, softmax-blend top-K.
    Returns (ttl_np, dp_np, ranking)."""
    import numpy as np
    from helper import load_voice_style as load_voice_styles

    voice_embs = _build_voice_embedding_cache(onnx_tts, voice_styles_dir,
                                              speaker_encoder, cache_path)
    if not voice_embs:
        raise RuntimeError("no built-in voice embeddings available — smart init impossible")

    target = speaker_encoder.emb_original.detach().cpu().numpy().astype("float32")
    t_norm = target / max(1e-9, float(np.linalg.norm(target)))

    sims: list[tuple[str, float]] = []
    for v, emb in voice_embs.items():
        e = emb / max(1e-9, float(np.linalg.norm(emb)))
        sims.append((v, float(np.dot(t_norm, e))))
    sims.sort(key=lambda x: -x[1])
    _log(f"voice similarity ranking: {[(v, round(s, 3)) for v, s in sims]}")

    top = sims[:top_k]
    s_arr = np.array([s for _, s in top], dtype="float32")
    s_shift = s_arr - s_arr.max()
    w = np.exp(s_shift / max(1e-6, temperature))
    w = w / w.sum()

    ttl_blend = None
    dp_blend = None
    for (vname, _), weight in zip(top, w):
        vpath = os.path.join(voice_styles_dir, f"{vname}.json")
        s = load_voice_styles([vpath])
        if ttl_blend is None:
            ttl_blend = s.ttl.astype("float32") * float(weight)
            dp_blend = s.dp.astype("float32") * float(weight)
        else:
            ttl_blend = ttl_blend + s.ttl.astype("float32") * float(weight)
            dp_blend = dp_blend + s.dp.astype("float32") * float(weight)

    ranking = [(v, round(s, 4), round(float(w[i]), 4) if i < len(w) else None)
               for i, (v, s) in enumerate(sims)]
    _log("blend top-%d: %s" % (top_k, ", ".join(
        f"{v}x{w[i]:.2f}" for i, (v, _) in enumerate(top))))
    return ttl_blend, dp_blend, ranking


# ---------------------------------------------------------------------------
# Reference sanity check (warn-only except truly broken files)
# ---------------------------------------------------------------------------
def _check_ref(ref_audio_path: str) -> dict:
    import numpy as np
    import soundfile as sf
    p = Path(ref_audio_path)
    if not p.exists():
        raise FileNotFoundError(f"reference audio not found: {ref_audio_path}")
    data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    dur = float(len(mono) / sr) if sr else 0.0
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
    stats = {"duration_s": round(dur, 2), "sample_rate": int(sr),
             "peak": round(peak, 4), "rms": round(rms, 5)}
    if dur < 1.5:
        raise RuntimeError(f"reference too short ({dur:.1f}s) — need >= 1.5s of speech")
    if peak < 0.005 or rms < 0.001:
        raise RuntimeError(f"reference effectively silent (peak={peak:.4f}, rms={rms:.5f})")
    if dur > 60.0:
        _log(f"WARN: reference is long ({dur:.1f}s); ~10-30s of clean speech is ideal")
    if peak < 0.10:
        _log(f"WARN: quiet reference (peak {peak:.3f}); SNR may suffer")
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train_voice(
    ref_audio_path: str,
    out_json_path: str,
    mode: str = "paid_parity",
    device: str = "cuda",
    num_steps: Optional[int] = None,
    model_dir: Optional[str] = None,
    repo_dir: Optional[str] = None,
    early_stop_loss: float = DEFAULT_EARLY_STOP_LOSS,
    log_every: int = 10,
    emb_cache_path: Optional[str] = None,
) -> dict:
    """Run the full paid_parity recipe on one reference wav.

    Args:
        ref_audio_path: RAW reference wav (used directly as ECAPA target —
            paid_parity deliberately skips gain-normalizing preprocessing).
        out_json_path:  where to save the Supertonic voice-style JSON.
        mode:           one of MODE_PRESETS keys.
        device:         'cuda' (default). Advisory: the repo modules pin
            cuda-if-available at import; on a CUDA box training runs on GPU,
            otherwise CPU. Passing 'cpu' on a CUDA box only prints a warning.
        num_steps:      optional override of the mode's iteration count
            (e.g. 1400). When set, the LR recipe is picked by budget:
            >= 600 steps -> "long" recipe (base_lr 0.003, ReduceLROnPlateau
            patience=200 factor=0.5, 5% warmup); < 600 -> standard recipe
            (base_lr 0.004, linear warmup -> decay to 20%). vocoder_steps=5,
            RAW target, grad_clip=1.0 and early stop stay the same.
        model_dir:      Supertonic 3 dir containing onnx/ and voice_styles/.
        repo_dir:       path to voice_clone_repo.
        early_stop_loss: stop once best ECAPA deficit <= this (0 disables).
        emb_cache_path: npz cache for built-in voice embeddings
            (default: <out_json dir>/_voice_embeddings.npz).

    Returns dict: out_path, mode, num_steps_planned/run, init/final loss+sim,
        best_iter, elapsed_s, device, early_stopped.
    """
    overall_t0 = time.time()
    if mode not in MODE_PRESETS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {sorted(MODE_PRESETS)}")

    repo_dir = str(Path(repo_dir or DEFAULT_REPO_DIR))
    model_dir = str(Path(model_dir or DEFAULT_MODEL_DIR))
    onnx_dir = os.path.join(model_dir, "onnx")
    voice_styles_dir = os.path.join(model_dir, "voice_styles")
    if not os.path.isdir(onnx_dir):
        raise RuntimeError(f"onnx dir not found: {onnx_dir}")
    if not os.path.isdir(voice_styles_dir):
        raise RuntimeError(f"voice_styles dir not found: {voice_styles_dir}")
    _setup_repo(repo_dir)

    import numpy as np
    import torch

    # Seed EARLY so the fixed noisy latent is reproducible too (clone.py seeds
    # right before the loop; seeding here as well is a strict superset).
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Simplified device pick: cuda-if-available (repo modules pin the same at
    # import time, so this always matches where tensors actually live).
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and DEVICE.type != "cuda":
        _log("WARN: cuda requested but not available — training on CPU (slow)")
    elif device == "cpu" and DEVICE.type == "cuda":
        _log("WARN: cpu requested but repo modules pin cuda-if-available — using cuda")
    _log(f"device: {DEVICE}")

    ref_audio_path = str(ref_audio_path)
    out_path = Path(out_json_path)
    ref_stats = _check_ref(ref_audio_path)
    steps_overridden = num_steps is not None
    num_steps = int(num_steps) if steps_overridden else MODE_PRESETS[mode]
    _log(f"clone start: ref={Path(ref_audio_path).name} mode={mode} steps={num_steps}")
    _log(f"{mode}: ECAPA target = RAW reference (multi-window 3.0s/1.5s averaged)")

    # ---- 1. heavy load (cached across calls) ----
    model = _ensure_model(onnx_dir, ref_audio_path)
    TTS = _ensure_tts(onnx_dir)
    dataloader = _ensure_dataloader(TTS)
    data_iter = iter(dataloader)

    from helper import load_voice_style as load_voice_styles  # noqa: F401

    def _to_dev(a):
        return torch.tensor(a, dtype=torch.float32).to(DEVICE)

    # ---- 2. smart init ----
    _log("running smart auto-init (ECAPA similarity over built-in voices)...")
    t_si = time.time()
    cache_p = Path(emb_cache_path) if emb_cache_path else \
        (out_path.parent / "_voice_embeddings.npz")
    ttl_np, dp_np, ranking = _smart_blend_init(
        TTS, voice_styles_dir, model.voice_encoder, cache_p, top_k=3, temperature=0.1
    )
    _log(f"smart init done in {time.time()-t_si:.1f}s")
    init_ttl = _to_dev(ttl_np)
    init_dp = _to_dev(dp_np)

    # ---- 3. duration prediction + fixed noisy latent ----
    tmp_input_ids, tmp_attention_mask = next(data_iter)
    tmp_input_ids = tmp_input_ids.to(DEVICE)
    tmp_attention_mask = tmp_attention_mask.to(DEVICE)
    _log(f"training vocoder_steps={VOCODER_STEPS} (mode={mode})")
    with torch.no_grad():
        init_dur = model.dp_model(tmp_input_ids, init_dp, tmp_attention_mask) / SPEED
        init_dur = init_dur.detach().cpu().numpy()
    noisy_latent_fixed, latent_mask = TTS.sample_noisy_latent(duration=init_dur)
    noisy_latent_fixed = torch.tensor(noisy_latent_fixed, dtype=torch.float32).to(DEVICE)
    latent_mask = torch.tensor(latent_mask, dtype=torch.float32).to(DEVICE)

    # ---- 4. trainable style vectors (style_ttl only; style_dp frozen) ----
    style_ttl = init_ttl.clone().requires_grad_(True)
    style_dp = init_dp.clone()
    norm_cap = float(init_ttl.norm()) if mode == "paid_parity_clean" else None
    if norm_cap is not None:
        _log(f"paid_parity_clean: norm-cap ||init_ttl||={norm_cap:.3f} (projecting each step)")

    # ---- 5. baseline ----
    with torch.no_grad():
        _, init_loss_t = model(
            tmp_input_ids, tmp_attention_mask, style_ttl,
            VOCODER_STEPS, noisy_latent_fixed, latent_mask,
        )
    init_loss = float(init_loss_t.detach().item())
    init_sim_pct = max(0.0, min(100.0, (1.0 - init_loss) * 100.0))
    _log(f"smart-init baseline: loss={init_loss:.4f}  sim={init_sim_pct:.0f}%")

    best_loss = init_loss
    best_ttl = style_ttl.detach().clone()
    best_dp = style_dp.detach().clone()
    best_iter = 0
    early_stopped = False
    iters_run = 0

    # ---- 6. training loop ----
    torch.manual_seed(SEED)   # same seeding point as clone.py
    np.random.seed(SEED)

    # Recipe selection: the mode preset decides normally; with a custom step
    # count the recipe is picked by budget (>= 600 steps -> long recipe).
    if steps_overridden:
        use_plateau = num_steps >= 600
        _log(f"custom steps={num_steps} -> "
             f"{'long recipe (plateau LR)' if use_plateau else 'standard recipe (linear LR)'}")
    else:
        use_plateau = mode in _PLATEAU_MODES
    base_lr = 0.003 if use_plateau else 0.004
    optimizer = torch.optim.AdamW([style_ttl], lr=base_lr, weight_decay=0.0)
    if use_plateau:
        warmup_steps = max(1, num_steps // 20)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=200, factor=0.5, min_lr=base_lr * 0.01,
        )
        _log(f"training: base_lr={base_lr}  grad_clip={GRAD_CLIP}  warmup={warmup_steps}  "
             f"steps={num_steps}  schedule=ReduceLROnPlateau(patience=200, factor=0.5)")
    else:
        warmup_steps = max(1, num_steps // 10) if num_steps > 0 else 1
        plateau_scheduler = None
        _log(f"training: base_lr={base_lr}  grad_clip={GRAD_CLIP}  warmup={warmup_steps}  "
             f"steps={num_steps}  schedule=linear_warmup_then_decay_to_20pct")

    t_train = time.time()
    for step in range(num_steps):
        if use_plateau:
            if step < warmup_steps:
                lr = base_lr * (step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr
            else:
                lr = optimizer.param_groups[0]["lr"]
        else:
            if step < warmup_steps:
                lr = base_lr * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
                lr = base_lr * (1.0 - 0.8 * progress)
            for g in optimizer.param_groups:
                g["lr"] = lr

        try:
            text_ids_batch, current_text_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            text_ids_batch, current_text_mask = next(data_iter)
        text_ids_batch = text_ids_batch.to(DEVICE)
        current_text_mask = current_text_mask.to(DEVICE)

        _, loss = model(
            text_ids_batch, current_text_mask, style_ttl,
            VOCODER_STEPS, noisy_latent_fixed, latent_mask,
        )
        loss.backward()
        try:
            gn = style_ttl.grad.norm().item() if style_ttl.grad is not None else 0.0
        except Exception:
            gn = 0.0
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

        if norm_cap is not None:
            with torch.no_grad():
                cur = float(style_ttl.norm())
                if cur > norm_cap:
                    style_ttl.mul_(norm_cap / cur)

        step_loss = float(loss.detach().item())
        if use_plateau and step >= warmup_steps:
            plateau_scheduler.step(step_loss)

        if step_loss < best_loss:
            best_loss = step_loss
            best_ttl = style_ttl.detach().clone()
            best_iter = step + 1

        iters_run = step + 1
        elapsed = time.time() - t_train
        ips = iters_run / max(elapsed, 1e-6)
        best_sim_pct = max(0.0, min(100.0, (1.0 - best_loss) * 100.0))
        if iters_run % max(1, log_every) == 0 or step == 0 or iters_run == num_steps:
            eta = (num_steps - iters_run) / max(ips, 1e-6)
            _log(f"iter {iters_run}/{num_steps}  loss={step_loss:.4f}  best={best_loss:.4f}  "
                 f"sim={best_sim_pct:.0f}%  lr={lr:.4f}  gn={gn:.3f}  {ips:.2f}/s  eta {eta:.0f}s")

        if early_stop_loss > 0 and best_loss <= early_stop_loss:
            _log(f"early stop at iter {iters_run}: best_loss {best_loss:.4f} <= {early_stop_loss:.4f}")
            early_stopped = True
            break

    # ---- 7. save (utils.save_style + clone.py-compatible metadata) ----
    final_sim_pct = max(0.0, min(100.0, (1.0 - best_loss) * 100.0))
    from utils import save_style
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_style(str(out_path), best_ttl, best_dp, ref_audio_path)
    _log(f"saved {out_path.name} (best loss {best_loss:.4f}, sim {final_sim_pct:.0f}%)")

    elapsed_total = round(time.time() - overall_t0, 1)
    try:
        with open(out_path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        saved_meta = saved.get("metadata", {}) or {}
        saved_meta.update({
            "schema_version": 3,
            "trainer": "colab_train_paid_parity",
            "mode": str(mode),
            "smart_init": True,
            "num_steps_planned": int(num_steps),
            "num_steps_run": int(iters_run),
            "base_lr": float(base_lr),
            "vocoder_steps_train": int(VOCODER_STEPS),
            "early_stop_threshold": float(early_stop_loss),
            "early_stopped": bool(early_stopped),
            "was_cancelled": False,
            "init_loss": round(float(init_loss), 6),
            "init_sim_pct": round(float(init_sim_pct), 2),
            "final_loss": round(float(best_loss), 6),
            "final_sim_pct": round(float(final_sim_pct), 2),
            "init_total_loss": round(float(init_loss), 6),
            "final_total_loss": round(float(best_loss), 6),
            "lambda_mel": 0.0,
            "mel_loss_enabled": False,
            "best_iter": int(best_iter),
            "smart_init_ranking": [
                {"voice": v, "sim": s, "weight": w} for (v, s, w) in ranking
            ],
            "ref_audio_stats": ref_stats,
            "preprocess_warning": "",
            "training_device": str(DEVICE),
            "training_elapsed_s": elapsed_total,
            "iter_per_sec": round(iters_run / max(time.time() - t_train, 1e-6), 3),
        })
        saved["metadata"] = saved_meta
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(saved, fh)
        _log(f"persisted v3 training metadata ({len(saved_meta)} fields)")
    except Exception as e:
        _log(f"WARN: could not write extended metadata: {type(e).__name__}: {e}")

    return {
        "out_path": str(out_path),
        "mode": mode,
        "num_steps_planned": num_steps,
        "num_steps_run": iters_run,
        "init_loss": round(init_loss, 6),
        "init_sim_pct": round(init_sim_pct, 2),
        "final_loss": round(best_loss, 6),
        "final_sim_pct": round(final_sim_pct, 2),
        "best_iter": best_iter,
        "early_stopped": early_stopped,
        "device": str(DEVICE),
        "elapsed_s": elapsed_total,
    }


# ---------------------------------------------------------------------------
# Verification helpers (used by the notebook's "Verify & listen" cell)
# ---------------------------------------------------------------------------
def synth_with_style(style_json_path: str,
                     text: str = "This is a test of the cloned voice.",
                     lang: str = "en", total_step: int = 8, speed: float = SPEED,
                     model_dir: Optional[str] = None,
                     repo_dir: Optional[str] = None):
    """Synthesize `text` with a trained style JSON via the ONNX TTS.
    Returns (wav_1d_float32, sample_rate)."""
    import numpy as np
    repo_dir = str(Path(repo_dir or DEFAULT_REPO_DIR))
    model_dir = str(Path(model_dir or DEFAULT_MODEL_DIR))
    _setup_repo(repo_dir)
    from helper import load_voice_style as load_voice_styles
    tts = _ensure_tts(os.path.join(model_dir, "onnx"))
    style = load_voice_styles([str(style_json_path)])
    wav, _ = tts(text=text, lang=lang, style=style, total_step=total_step, speed=speed)
    wav_1d = wav.squeeze() if wav.ndim > 1 else wav
    return wav_1d.astype(np.float32), int(tts.sample_rate)


def ecapa_similarity(wav_44k_1d, ref_audio_path: str,
                     repo_dir: Optional[str] = None) -> float:
    """ECAPA cosine similarity between a generated 44.1kHz wav and the raw
    reference (multi-window target embedding, same as training)."""
    import torch
    repo_dir = str(Path(repo_dir or DEFAULT_REPO_DIR))
    _setup_repo(repo_dir)
    model = _CACHE.get("model")
    if model is not None:
        enc = model.voice_encoder
    else:
        from utils.loss import SpeakerID
        enc = SpeakerID(target_wav_path=str(ref_audio_path))
    ref_emb = _multi_window_embed_target(enc, str(ref_audio_path))
    gen_emb = torch.tensor(_embed_wav_16k(wav_44k_1d, enc)).to(ref_emb.device)
    a = torch.nn.functional.normalize(ref_emb, p=2, dim=0)
    b = torch.nn.functional.normalize(gen_emb, p=2, dim=0)
    return float(torch.dot(a, b).item())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="paid_parity Supertonic-3 voice clone trainer")
    ap.add_argument("ref_audio", help="reference wav (raw, unprocessed)")
    ap.add_argument("out_json", help="output voice-style JSON path")
    ap.add_argument("--mode", default="paid_parity", choices=sorted(MODE_PRESETS))
    ap.add_argument("--steps", type=int, default=None, help="override iteration count")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--early-stop", type=float, default=DEFAULT_EARLY_STOP_LOSS)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    try:
        result = train_voice(
            args.ref_audio, args.out_json, mode=args.mode, device=args.device,
            num_steps=args.steps, model_dir=args.model_dir, repo_dir=args.repo_dir,
            early_stop_loss=args.early_stop, log_every=args.log_every,
        )
    except Exception as e:
        _log(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
