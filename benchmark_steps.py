"""Calibrate DEFAULT_ECAPA_STEPS: embedding-convergence analysis.

The ECAPA Encoder Steps slider controls how many 3s windows of the reference
the speaker encoder averages into the target embedding. This script measures,
per reference clip in data/refs/:

  * cos(emb_N, emb_gold) — how close the N-step embedding is to a dense
    gold-standard embedding (64 windows). Once this saturates, more steps
    cannot change the clone (the blend weights are a smooth function of the
    target embedding), so nothing further can be audible.
  * the top-3 blend (voices + softmax weights) at each N, and its L1 distance
    to the gold blend — the direct input to synthesis.
  * wall-clock embed time at each N.

Pick the default as the highest N that still changes the blend (quality
first), i.e. the last N before both metrics saturate.

Run:  python benchmark_steps.py   (engine not needed — encoder only)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ecapa  # noqa: E402

STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
GOLD_STEPS = 64


def blend_weights(target: np.ndarray, voice_embs: dict) -> dict[str, float]:
    sims = []
    for v, emb in voice_embs.items():
        e = emb / max(1e-9, float(np.linalg.norm(emb)))
        sims.append((v, float(np.dot(target, e))))
    sims.sort(key=lambda x: -x[1])
    top = sims[:ecapa.TOP_K]
    s_arr = np.array([s for _, s in top], dtype="float32")
    w = np.exp((s_arr - s_arr.max()) / ecapa.SOFTMAX_TEMPERATURE)
    w = w / w.sum()
    return {v: float(w[i]) for i, (v, _) in enumerate(top)}


def blend_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def main() -> None:
    ecapa.load_encoder()
    ecapa.warmup_encoder()

    if not ecapa.EMB_CACHE_PATH.exists():
        print("voice_embeddings.npz missing — start the app once first")
        sys.exit(1)
    with np.load(ecapa.EMB_CACHE_PATH, allow_pickle=False) as data:
        voice_embs = {k: data[k].astype("float32") for k in data.files}

    refs = sorted((HERE / "data" / "refs").glob("*.*"))
    if not refs:
        print("no reference clips in data/refs — upload one via the app first")
        sys.exit(1)

    report = {}
    for ref in refs:
        print(f"\n=== {ref.name} ===")
        gold, gold_eff = ecapa.embed_reference(str(ref), steps=GOLD_STEPS)
        gold_blend = blend_weights(gold, voice_embs)
        print(f"gold: {gold_eff} windows, blend "
              + " ".join(f"{v}×{w:.2f}" for v, w in gold_blend.items()))

        rows = []
        for n in STEPS:
            t0 = time.perf_counter()
            emb, eff = ecapa.embed_reference(str(ref), steps=n)
            ms = (time.perf_counter() - t0) * 1000
            cos = float(np.dot(emb, gold))
            bl = blend_weights(emb, voice_embs)
            dl1 = blend_l1(bl, gold_blend)
            rows.append({
                "steps": n, "effective": eff, "embed_ms": round(ms, 1),
                "cos_vs_gold": round(cos, 5), "blend_l1_vs_gold": round(dl1, 4),
                "blend": {v: round(w, 3) for v, w in bl.items()},
            })
            print(f"  N={n:<3} eff={eff:<3} {ms:7.1f}ms  cos_gold={cos:.5f}  "
                  f"blendΔ={dl1:.4f}  " + " ".join(f"{v}×{w:.2f}" for v, w in bl.items()))
        report[ref.name] = {"gold_windows": gold_eff, "gold_blend": gold_blend, "rows": rows}

    out = HERE / "data" / "cache" / "benchmark_steps.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
