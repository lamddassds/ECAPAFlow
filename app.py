"""ECAPAFlow — Supertonic Voice Builder with real ECAPA smart-init cloning.

The experiment: LoudFlow's production voice-clone path (supertonic3_test/
clone.py, paid_parity* modes) uses ECAPA smart-init + 120-1500 gradient
iterations (8 min - 2 hrs). ECAPAFlow runs the smart-init step ALONE — no
gradient training — and targets under 1 second per clone.

Run: python app.py   (opens http://localhost:7873 after 2s)
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import re
import socket
import struct
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path
from typing import Optional

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
import voices as voices_mod  # noqa: E402
from engine import engine  # noqa: E402
from normalizer import normalize as normalize_text  # noqa: E402

# The 12 languages the Voice Builder UI exposes (Supertonic 3 supports 31;
# these are the ones in the design's language menu).
UI_LANGUAGES = [
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
# by benchmark_steps.py on real references (see README / benchmark results).
DEFAULT_ECAPA_STEPS = 12
MAX_ECAPA_STEPS = 32

MAX_TEXT_CHARS = 100_000

# Streaming synthesis chunking: the first chunk is kept short so audio starts
# flowing fast; later chunks are bigger to amortize per-call overhead.
STREAM_FIRST_CHUNK_CHARS = 160
STREAM_CHUNK_CHARS = 340

app = FastAPI(title="ECAPAFlow")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


# ---- startup ----
@app.on_event("startup")
def _startup():
    engine.start_background_load()
    ecapa_mod.start_background_load()

    def _prewarm():
        # Build the built-in voice embedding table once engine + encoder are
        # up, so the FIRST user clone already hits the <1s path.
        while not engine.loaded:
            if not engine.loading and engine.status.startswith("Load failed"):
                return
            time.sleep(0.5)
        try:
            ecapa_mod.load_encoder()
            ecapa_mod.build_voice_embedding_cache(engine)
        except Exception as e:
            engine.log_error("prewarm", e)

    threading.Thread(target=_prewarm, name="ecapaflow-prewarm", daemon=True).start()


# ---- helpers ----
def _soft_limit(wav: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    if wav.size == 0:
        return wav
    peak = float(np.abs(wav).max())
    if peak > target_peak:
        return wav * (target_peak / peak)
    return wav


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    wav = _soft_limit(wav, target_peak=0.85)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _read_audio_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        import librosa
        return float(librosa.get_duration(path=str(path)))


def _clone_voice_now(voice: dict, ecapa_steps: int) -> dict:
    """Run smart-init on a registry voice, persist the style, update the
    record. Returns {voice, clone} (clone = smart-init result summary).
    Kicks off async verification (synthesize sample + ECAPA-compare)."""
    ref = voices_mod.ref_path(voice)
    if not ref.exists():
        raise HTTPException(status_code=404, detail=f"reference audio missing: {ref.name}")

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
    """Load the cloned style for a registry voice (must be cloned)."""
    sp = voices_mod.style_path(voice)
    if not sp or not sp.exists():
        return None
    cache_key = f"ecapaflow:{voice['id']}"
    cached = engine.voice_cache.get(cache_key)
    if cached is not None:
        return cached
    return engine.load_voice_from_path(str(sp), cache_key)


async def _prepare_voice_for_synthesis(voice_id: str, ecapa_steps: int):
    """Shared front half of both synthesis endpoints: look up the voice,
    auto-clone it if needed, and load its style.
    Returns (voice, style, auto_cloned, clone_ms)."""
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Supertonic 3 engine still loading")
    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")

    # Auto-clone: if this voice was never cloned, run smart-init first (it is
    # the whole point of ECAPAFlow that this costs well under a second).
    auto_cloned = False
    clone_ms = 0.0
    if not v.get("cloned") or not voices_mod.style_path(v) or not voices_mod.style_path(v).exists():
        est = max(1, min(MAX_ECAPA_STEPS, int(ecapa_steps)))
        res = await run_in_threadpool(_clone_voice_now, v, est)
        v = res["voice"]
        auto_cloned = True
        clone_ms = res["clone"]["timings_ms"]["total"]

    style = await run_in_threadpool(_resolve_style, v)
    if style is None:
        raise HTTPException(status_code=500, detail="cloned style missing on disk")
    return v, style, auto_cloned, clone_ms


# ---- streaming synthesis helpers ----
def _wav_stream_header(sr: int) -> bytes:
    """44-byte PCM16 mono WAV header with 'unknown length' sizes (0xFFFFFFFF),
    the standard trick for live/streamed WAV."""
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def _pcm16_bytes(wav: np.ndarray) -> bytes:
    wav = _soft_limit(wav, target_peak=0.85)
    return (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _split_stream_chunks(text: str) -> list[str]:
    """Split normalized text into synthesis chunks along sentence boundaries.
    The first chunk is small (fast time-to-first-audio), later chunks larger.
    Overlong sentences are hard-split on commas/spaces."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", text) if p.strip()]

    sentences: list[str] = []
    for p in parts:
        while len(p) > 400:  # hard-split pathological run-ons
            cut = max(p.rfind(",", 100, 400), p.rfind(" ", 100, 400))
            if cut <= 0:
                cut = 400
            sentences.append(p[: cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            sentences.append(p)

    chunks: list[str] = []
    cur = ""
    for s in sentences:
        limit = STREAM_FIRST_CHUNK_CHARS if not chunks else STREAM_CHUNK_CHARS
        if cur and len(cur) + 1 + len(s) > limit:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text.strip()]


# Stats for finished streams, fetched by the UI right after the audio stream
# ends (RTF badge). Small dict, pruned to the most recent entries.
STREAM_STATS: dict[str, dict] = {}
_STREAM_STATS_LOCK = threading.Lock()


def _store_stream_stats(stream_id: str, stats: dict) -> None:
    with _STREAM_STATS_LOCK:
        STREAM_STATS[stream_id] = stats
        while len(STREAM_STATS) > 50:
            STREAM_STATS.pop(next(iter(STREAM_STATS)))


# ---- routes ----
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((HERE / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
def api_status():
    s = engine.status_dict()
    s["ecapa"] = ecapa_mod.status()
    s["languages"] = UI_LANGUAGES
    s["default_ecapa_steps"] = DEFAULT_ECAPA_STEPS
    s["max_ecapa_steps"] = MAX_ECAPA_STEPS
    s["max_text_chars"] = MAX_TEXT_CHARS
    return s


@app.get("/api/voices")
def api_voices():
    return {"voices": voices_mod.list_voices()}


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


@app.patch("/api/voices/{voice_id}")
async def api_voices_rename(voice_id: str, name: str = Form(...)):
    v = voices_mod.update_voice(voice_id, name=name.strip()[:64] or "Voice")
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
    p = voices_mod.ref_path(v)
    if not p.exists():
        raise HTTPException(status_code=404, detail="reference audio missing")
    media = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
             ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".webm": "audio/webm"}
    return FileResponse(str(p), media_type=media.get(p.suffix.lower(), "application/octet-stream"))


@app.post("/api/voices/{voice_id}/clone")
async def api_voices_clone(voice_id: str, ecapa_steps: int = Form(DEFAULT_ECAPA_STEPS)):
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Supertonic 3 engine still loading")
    v = voices_mod.get_voice(voice_id)
    if not v:
        raise HTTPException(status_code=404, detail="voice not found")
    steps = max(1, min(MAX_ECAPA_STEPS, int(ecapa_steps)))
    try:
        return await run_in_threadpool(_clone_voice_now, v, steps)
    except HTTPException:
        raise
    except Exception as e:
        engine.log_error("clone", e)
        raise HTTPException(status_code=500, detail=f"clone failed: {type(e).__name__}: {e}")


@app.post("/api/synthesize")
async def api_synthesize(
    voice_id: str = Form(...),
    text: str = Form(...),
    lang: str = Form("en"),
    steps: int = Form(16),
    speed: float = Form(1.25),
    silence: float = Form(0.30),
    ecapa_steps: int = Form(DEFAULT_ECAPA_STEPS),
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text required")
    if lang not in SUPERTONIC_LANGS:
        lang = "na"

    v, style, auto_cloned, clone_ms = await _prepare_voice_for_synthesis(voice_id, ecapa_steps)

    # LoudFlow text normalization (regex DE/EN safety net — mT5 path was
    # deliberately dropped in LoudFlow, see normalizer.py docstring).
    norm_text = normalize_text(text.strip()[:MAX_TEXT_CHARS], lang=lang)

    try:
        wav, tts_dur, infer = await run_in_threadpool(
            engine.synthesize, norm_text, style, int(steps), float(speed),
            lang, float(silence),
        )
    except Exception as e:
        engine.log_error("synthesize", e)
        raise HTTPException(status_code=500, detail=f"synthesis failed: {e}")

    audio_sec = float(len(wav)) / float(engine.sample_rate)
    rtf = (infer / audio_sec) if audio_sec > 0 else 0.0
    body = _wav_bytes(wav, engine.sample_rate)

    ci = (v.get("clone_info") or {})
    headers = {
        "X-RTF": f"{rtf:.4f}",
        "X-DUR": f"{audio_sec:.3f}",
        "X-INFER": f"{infer:.3f}",
        "X-STEPS": str(int(steps)),
        "X-LANG": lang,
        "X-DEVICE": engine.device_label,
        "X-AUTO-CLONED": "1" if auto_cloned else "0",
        "X-CLONE-MS": f"{clone_ms:.0f}",
        "X-EST-SIM": str(ci.get("est_sim_pct", "")),
        "X-NORMALIZED": urllib.parse.quote(norm_text[:900]),
        "Access-Control-Expose-Headers": (
            "X-RTF,X-DUR,X-INFER,X-STEPS,X-LANG,X-DEVICE,X-AUTO-CLONED,"
            "X-CLONE-MS,X-EST-SIM,X-NORMALIZED"
        ),
    }
    return StreamingResponse(io.BytesIO(body), media_type="audio/wav", headers=headers)


@app.post("/api/synthesize/stream")
async def api_synthesize_stream(
    voice_id: str = Form(...),
    text: str = Form(...),
    lang: str = Form("en"),
    steps: int = Form(16),
    speed: float = Form(1.25),
    silence: float = Form(0.30),
    ecapa_steps: int = Form(DEFAULT_ECAPA_STEPS),
):
    """Smart streaming synthesis: split the text into sentence chunks, run
    them through the engine one by one and stream WAV bytes out as each chunk
    finishes — the browser starts playing while the tail is still rendering.
    Final RTF/timing stats are stored under X-Stream-Id and fetched by the UI
    via /api/synthesize/stream/{id}/stats once the stream ends."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text required")
    if lang not in SUPERTONIC_LANGS:
        lang = "na"

    v, style, auto_cloned, clone_ms = await _prepare_voice_for_synthesis(voice_id, ecapa_steps)

    norm_text = normalize_text(text.strip()[:MAX_TEXT_CHARS], lang=lang)
    chunks = _split_stream_chunks(norm_text)
    stream_id = uuid.uuid4().hex[:12]
    sr = engine.sample_rate
    steps_i, speed_f, silence_f = int(steps), float(speed), float(silence)

    def gen():
        yield _wav_stream_header(sr)
        total_samples = 0
        infer_total = 0.0
        t0 = time.perf_counter()
        error = ""
        try:
            for chunk in chunks:
                wav, _, infer = engine.synthesize(
                    chunk, style, steps_i, speed_f, lang, silence_f,
                )
                infer_total += infer
                total_samples += int(wav.size)
                yield _pcm16_bytes(wav)
        except Exception as e:  # noqa: BLE001 — surface via stats, stream is already open
            engine.log_error("synthesize_stream", e)
            error = f"{type(e).__name__}: {e}"
        audio_sec = total_samples / float(sr)
        _store_stream_stats(stream_id, {
            "stream_id": stream_id,
            "rtf": (infer_total / audio_sec) if audio_sec > 0 else 0.0,
            "audio_sec": audio_sec,
            "infer_sec": infer_total,
            "wall_sec": time.perf_counter() - t0,
            "chunks": len(chunks),
            "steps": steps_i,
            "lang": lang,
            "device": engine.device_label,
            "auto_cloned": auto_cloned,
            "clone_ms": clone_ms,
            "error": error,
        })

    ci = (v.get("clone_info") or {})
    headers = {
        "X-Stream-Id": stream_id,
        "X-Sample-Rate": str(sr),
        "X-Chunks": str(len(chunks)),
        "X-Auto-Cloned": "1" if auto_cloned else "0",
        "X-Clone-MS": f"{clone_ms:.0f}",
        "X-Est-Sim": str(ci.get("est_sim_pct", "")),
        "X-Device": engine.device_label,
        "Cache-Control": "no-store",
        "Access-Control-Expose-Headers": (
            "X-Stream-Id,X-Sample-Rate,X-Chunks,X-Auto-Cloned,X-Clone-MS,"
            "X-Est-Sim,X-Device"
        ),
    }
    return StreamingResponse(gen(), media_type="audio/wav", headers=headers)


@app.get("/api/synthesize/stream/{stream_id}/stats")
def api_stream_stats(stream_id: str):
    with _STREAM_STATS_LOCK:
        stats = STREAM_STATS.get(stream_id)
    if not stats:
        raise HTTPException(status_code=404, detail="stats not ready")
    return stats


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


def _lan_ip() -> str:
    """Best-effort LAN address of this machine (no traffic is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return ""
    finally:
        s.close()


def main() -> None:
    import uvicorn
    port = 7873
    for candidate in (7873, 7874, 7875, 7876):
        if not _port_in_use(candidate):
            port = candidate
            break
    url = f"http://localhost:{port}"
    print(f"ECAPAFlow starting on {url}")
    lan = _lan_ip()
    if lan:
        print(f"On your phone (same Wi-Fi): http://{lan}:{port}")
    if os.environ.get("ECAPAFLOW_NO_BROWSER") != "1":
        _open_browser(url)
    # 0.0.0.0 so phones/tablets on the same network can use the laptop as the
    # synthesis engine.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
