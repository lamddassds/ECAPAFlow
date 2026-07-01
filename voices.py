"""Voice registry for ECAPAFlow: uploaded/recorded reference clips and their
cloned styles, persisted to data/voices/registry.json so tabs survive restarts.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
REFS_DIR = DATA_DIR / "refs"
VOICES_DIR = DATA_DIR / "voices"
REGISTRY_PATH = VOICES_DIR / "registry.json"

for d in (REFS_DIR, VOICES_DIR):
    d.mkdir(parents=True, exist_ok=True)

_lock = threading.RLock()
_registry: Optional[dict] = None


def _load() -> dict:
    global _registry
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is not None:
            return _registry
        if REGISTRY_PATH.exists():
            try:
                with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                    _registry = json.load(f)
            except Exception:
                _registry = {"voices": []}
        else:
            _registry = {"voices": []}
        return _registry


def _save() -> None:
    with _lock:
        tmp = REGISTRY_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_registry, f, indent=2)
        tmp.replace(REGISTRY_PATH)


def list_voices() -> list[dict]:
    reg = _load()
    with _lock:
        return [dict(v) for v in reg["voices"]]


def get_voice(voice_id: str) -> Optional[dict]:
    reg = _load()
    with _lock:
        for v in reg["voices"]:
            if v["id"] == voice_id:
                return dict(v)
    return None


def add_voice(name: str, vtype: str, ref_filename: str, duration_s: float,
              ref_stats: Optional[dict] = None) -> dict:
    """vtype: 'uploaded' | 'recorded'."""
    reg = _load()
    voice = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "type": vtype,
        "ref_filename": ref_filename,
        "duration_s": round(float(duration_s), 2),
        "created_at": time.time(),
        "cloned": False,
        "style_filename": None,
        "clone_info": None,
        "ref_stats": ref_stats or {},
    }
    with _lock:
        reg["voices"].insert(0, voice)  # newest first, like the prototype tabs
        _save()
    return dict(voice)


def update_voice(voice_id: str, **fields: Any) -> Optional[dict]:
    reg = _load()
    with _lock:
        for v in reg["voices"]:
            if v["id"] == voice_id:
                v.update(fields)
                _save()
                return dict(v)
    return None


def remove_voice(voice_id: str) -> bool:
    reg = _load()
    with _lock:
        for i, v in enumerate(reg["voices"]):
            if v["id"] == voice_id:
                reg["voices"].pop(i)
                _save()
                # best-effort cleanup of files on disk
                for key in ("ref_filename", "style_filename"):
                    fn = v.get(key)
                    if fn:
                        base = REFS_DIR if key == "ref_filename" else VOICES_DIR
                        try:
                            (base / fn).unlink(missing_ok=True)
                        except Exception:
                            pass
                return True
    return False


def ref_path(voice: dict) -> Path:
    return REFS_DIR / voice["ref_filename"]


def style_path(voice: dict) -> Optional[Path]:
    fn = voice.get("style_filename")
    return (VOICES_DIR / fn) if fn else None
