"""Supertonic 3 engine wrapper for ECAPAFlow: thread-safe synthesis, background
loading, provider auto-selection (CUDA > DirectML > CPU), voice cache, warmup.

Adapted from LoudFlow supertonic3_test/engine.py. Differences:
  * synthesize() exposes silence_duration (inter-chunk padding, the UI's
    "Silence Duration" slider) — the pip package has supported it all along.
  * no RVC stage-2 hooks (ECAPAFlow deliberately stays single-system).
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Any, Optional

import numpy as np

# Apply env vars BEFORE importing supertonic/onnxruntime so threading settings
# are picked up at session creation time.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

VOICES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
PRECACHE = ["M1", "F1"]


def detect_providers() -> tuple[list[str], str]:
    """Return (provider_list, label). Prefer CUDA > DirectML > CPU."""
    try:
        import onnxruntime as ort  # type: ignore
        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"], "CPU"

    chosen: list[str] = []
    label = "CPU"
    if "CUDAExecutionProvider" in available:
        chosen.append("CUDAExecutionProvider")
        label = "CUDA"
    elif "DmlExecutionProvider" in available:
        chosen.append("DmlExecutionProvider")
        label = "DirectML"
    chosen.append("CPUExecutionProvider")
    return chosen, label


def _patch_providers(providers: list[str]) -> None:
    """Override supertonic's default ONNX provider list.

    supertonic.loader imports DEFAULT_ONNX_PROVIDERS by name at module load,
    so we patch both the original list and the loader's binding to be safe.
    """
    try:
        import supertonic.config as _cfg
        import supertonic.loader as _ldr

        _cfg.DEFAULT_ONNX_PROVIDERS.clear()
        _cfg.DEFAULT_ONNX_PROVIDERS.extend(providers)
        _ldr.DEFAULT_ONNX_PROVIDERS = providers
    except Exception:
        pass


def detect_gpu_info() -> dict:
    """Best-effort GPU detection. Returns {has_cuda, gpu_name, vram_gb}."""
    info: dict[str, Any] = {"has_cuda": False, "gpu_name": "", "vram_gb": 0.0}
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                info["has_cuda"] = True
                info["gpu_name"] = parts[0]
                try:
                    info["vram_gb"] = round(float(parts[1]) / 1024.0, 1)
                except Exception:
                    pass
    except Exception:
        pass

    if not info["gpu_name"]:
        try:
            import subprocess
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=4,
            )
            if out.returncode == 0:
                names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
                if names:
                    info["gpu_name"] = " / ".join(names)
        except Exception:
            pass
    return info


class SupertonicEngine:
    """Thread-safe Supertonic 3 wrapper with background load and warmup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.tts = None  # type: ignore[assignment]
        self.loaded = False
        self.loading = False
        self.status = "Idle"
        self.errors: list[dict] = []
        self.provider_list, self.device_label = detect_providers()
        self.gpu_info = detect_gpu_info()
        self.voice_cache: dict[str, Any] = {}
        self.sample_rate = 44100
        self.warmup_done = False
        self._load_thread: Optional[threading.Thread] = None

    @property
    def device_full_label(self) -> str:
        if self.device_label == "CUDA" and self.gpu_info.get("gpu_name"):
            return f"CUDA {self.gpu_info['gpu_name']}"
        if self.device_label == "DirectML" and self.gpu_info.get("gpu_name"):
            return f"DirectML {self.gpu_info['gpu_name']}"
        return self.device_label

    def status_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "message": self.status,
            "device": self.device_full_label,
            "device_short": self.device_label,
            "providers": self.provider_list,
            "voices": VOICES,
            "errors": self.errors[-20:],
            "warmup_done": self.warmup_done,
            "sample_rate": self.sample_rate,
        }

    def log_error(self, where: str, exc: BaseException) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=4)
        self.errors.append({"ts": time.time(), "where": where, "msg": msg, "tb": tb})

    def clear_errors(self) -> None:
        self.errors.clear()

    def start_background_load(self) -> None:
        if self.loaded or self.loading:
            return
        self.loading = True
        self.status = "Initializing..."
        t = threading.Thread(target=self._load, name="supertonic-load", daemon=True)
        self._load_thread = t
        t.start()

    def _load(self) -> None:
        try:
            with self._load_lock:
                gpu_opt_name = None
                if self.device_label == "CUDA":
                    gpu_opt_name = "CUDAExecutionProvider"
                elif self.device_label == "DirectML":
                    gpu_opt_name = "DmlExecutionProvider"
                if gpu_opt_name:
                    try:
                        _patch_providers(self.provider_list)
                    except Exception as e:
                        self.log_error("provider_patch", e)

                self.status = "Loading Supertonic 3 model (first run downloads ~400MB)..."
                from supertonic import TTS
                t0 = time.time()
                self.tts = TTS(auto_download=True)
                self.sample_rate = int(self.tts.sample_rate)
                load_dt = time.time() - t0
                self.status = f"Model loaded in {load_dt:.1f}s. Caching voices..."

                for v in PRECACHE:
                    try:
                        self.voice_cache[v] = self.tts.get_voice_style(voice_name=v)
                    except Exception as e:
                        self.log_error(f"precache:{v}", e)

                self.loaded = True
                self.loading = False
                self.status = "Ready"

            # Warmup outside the load lock so /status responds while warming
            try:
                self.status = "Warming up..."
                self._warmup()
                self.warmup_done = True
                self.status = "Ready"
            except Exception as e:
                self.log_error("warmup", e)

        except Exception as e:
            self.log_error("load", e)
            self.loading = False
            self.loaded = False
            self.status = f"Load failed: {e}"

    def _warmup(self) -> None:
        if not self.tts:
            return
        style = self.voice_cache.get("M1") or self.tts.get_voice_style(voice_name="M1")
        # Warm up through self.synthesize() — the SAME path production
        # requests take, including the IOBinding fast path — so the one-time
        # DirectML graph/shader-capture cost is paid here, not on the first
        # user click. (self.synthesize acquires self._lock itself.)
        self.synthesize("Warm up.", style, 5, 1.05, "en", 0.3)

    # ----- voice loading -----
    def get_voice(self, name: str) -> Any:
        if not self.loaded or self.tts is None:
            raise RuntimeError("Model not loaded yet")
        if name in self.voice_cache:
            return self.voice_cache[name]
        if name in VOICES:
            style = self.tts.get_voice_style(voice_name=name)
        else:
            style = self.tts.get_voice_style_from_path(name)
        self.voice_cache[name] = style
        return style

    def load_voice_from_path(self, path: str, cache_key: Optional[str] = None) -> Any:
        if not self.loaded or self.tts is None:
            raise RuntimeError("Model not loaded yet")
        style = self.tts.get_voice_style_from_path(path)
        if cache_key:
            self.voice_cache[cache_key] = style
        return style

    @property
    def voice_styles_dir(self) -> str:
        """Directory holding the built-in F1..M5 style JSONs."""
        if self.tts is None:
            raise RuntimeError("Model not loaded yet")
        return str(self.tts.model_dir / "voice_styles")

    # ----- synthesis -----
    def _fast_synthesize_iobinding(
        self, text: str, voice_style: Any, total_steps: int, speed: float,
        lang: Optional[str],
    ):
        """IOBinding-accelerated path for the common case (text fits in one
        internal chunk): keeps the diffusion loop's noisy_latent resident on
        the DirectML device across steps instead of round-tripping it through
        host memory on every one of the `total_steps` calls. Verified
        bit-identical to the pip package's own numpy-passing loop and ~1.6-
        1.8x faster on this hardware (2026-07-05 benchmark). Returns None to
        signal "fall back to the general pip-package path" for anything this
        fast path doesn't handle (multi-chunk text, unusual lang/steps) —
        never raises to the caller.

        A FRESH io_binding() object is allocated every iteration. Reusing one
        IOBinding object across steps via clear_binding_inputs()/
        clear_binding_outputs() was tested and crashes this onnxruntime-
        directml build (heap corruption) — do not "optimize" that away.
        """
        from supertonic.config import AVAILABLE_LANGUAGES
        from supertonic.core import Style
        from supertonic.utils import chunk_text

        if not isinstance(voice_style, Style) or not (1 <= total_steps <= 100):
            return None
        effective_lang = lang if self.tts.is_multilingual else None
        if effective_lang is not None and effective_lang not in AVAILABLE_LANGUAGES:
            return None
        if not text or not text.strip() or len(text) > 100_000:
            return None

        max_chunk_length = 120 if effective_lang == "ko" else 300
        chunks = chunk_text(text, max_chunk_length)
        if len(chunks) != 1:
            return None  # let the pip package's own multi-chunk loop handle it

        core = self.tts.model  # supertonic.core.Supertonic
        text_ids, text_mask = core.text_processor([chunks[0]], effective_lang)
        dur_onnx, *_ = core.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": voice_style.dp, "text_mask": text_mask})
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = core.text_enc_ort.run(
            None, {"text_ids": text_ids, "style_ttl": voice_style.ttl, "text_mask": text_mask})
        xt, latent_mask = core.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_steps], dtype=np.float32)

        out_name = core.vector_est_ort.get_outputs()[0].name
        xt_dml = None
        io = None
        for step in range(total_steps):
            current_step = np.array([step], dtype=np.float32)
            io = core.vector_est_ort.io_binding()
            if xt_dml is None:
                io.bind_cpu_input("noisy_latent", xt)
            else:
                io.bind_ortvalue_input("noisy_latent", xt_dml)
            io.bind_cpu_input("text_emb", text_emb_onnx)
            io.bind_cpu_input("style_ttl", voice_style.ttl)
            io.bind_cpu_input("text_mask", text_mask)
            io.bind_cpu_input("latent_mask", latent_mask)
            io.bind_cpu_input("current_step", current_step)
            io.bind_cpu_input("total_step", total_step_np)
            io.bind_output(out_name, "dml")
            core.vector_est_ort.run_with_iobinding(io)
            xt_dml = io.get_outputs()[0]
        xt_final = io.copy_outputs_to_cpu()[0]
        wav, *_ = core.vocoder_ort.run(None, {"latent": xt_final})
        return wav, dur_onnx

    def synthesize(
        self,
        text: str,
        voice: Any,
        steps: int = 8,
        speed: float = 1.05,
        lang: Optional[str] = "en",
        silence_duration: float = 0.3,
    ) -> tuple[np.ndarray, float, float]:
        """Returns (wav_1d_float32, tts_duration_s, infer_time_s)."""
        if not self.loaded or self.tts is None:
            raise RuntimeError("Model not loaded yet")

        if isinstance(voice, str):
            voice_style = self.get_voice(voice)
        else:
            voice_style = voice

        # The clock starts INSIDE the lock. Started outside, every second
        # spent queued behind another thread's synthesis was billed as this
        # call's inference time — and that number is persisted: RtfHistory
        # writes a rung's first whole-stream sample verbatim and Auto quality
        # then reads it forever, so one race (the post-clone verification
        # daemon against a user's first audition) could lock Auto out of a
        # step count permanently. Measuring only the held section is the
        # honest quantity: how long the model took, not how long we waited.
        with self._lock:
            t0 = time.perf_counter()
            result = None
            if self.device_label in ("DirectML", "CUDA"):
                try:
                    result = self._fast_synthesize_iobinding(
                        text, voice_style, int(steps), float(speed), lang if lang else None)
                except Exception as e:
                    self.log_error("iobinding_fast_path", e)
                    result = None
            if result is not None:
                wav, dur_arr = result
            else:
                wav, dur_arr = self.tts.synthesize(
                    text=text,
                    voice_style=voice_style,
                    total_steps=int(steps),
                    speed=float(speed),
                    silence_duration=max(0.0, float(silence_duration)),
                    lang=lang if lang else None,
                    verbose=False,
                )
            infer = time.perf_counter() - t0

        wav_1d = wav.squeeze() if wav.ndim > 1 else wav
        if wav_1d.dtype != np.float32:
            wav_1d = wav_1d.astype(np.float32)

        try:
            tts_dur = float(np.asarray(dur_arr).sum())
        except Exception:
            tts_dur = float(len(wav_1d)) / float(self.sample_rate)

        return wav_1d, tts_dur, infer


# module-level singleton
engine = SupertonicEngine()
