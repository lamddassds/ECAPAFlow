# ECAPAFlow VoiceForge — GPU voice-clone training on Google Colab

Trains true 1:1 Supertonic-3 voice clones with LoudFlow's `paid_parity` gradient
recipe on a Colab GPU, and produces voice-style `.json` files you import back
into ECAPAFlow.

## One-time setup

1. In Google Drive, create the folder **`MyDrive/ECAPAFlow/`** and upload
   **`voice_clone_repo.zip`** (from this folder) into it.
2. Create **`MyDrive/ECAPAFlow/refs/`** and put your reference recordings there
   (`.wav`, ~10–30 s of clean speech per voice; raw is fine — the recipe
   deliberately uses the un-normalized file).
3. Open **`ECAPAFlow_VoiceForge.ipynb`** in Colab
   ([colab.research.google.com](https://colab.research.google.com) → Upload).
4. Runtime → Change runtime type → **A100 GPU** (L4/T4 also work, just slower).

The notebook is self-contained: it writes `train_paid_parity.py` itself via
`%%writefile` — only the zip lives on Drive.

## Running

Run the cells top to bottom:

| Cell | What it does |
|---|---|
| 1 Setup | pip installs + GPU check |
| 2 Mount Drive + get code | unzips the repo, writes the trainer |
| 3 Download Supertonic 3 | pulls `Supertone/supertonic-3` from Hugging Face |
| 4 Config | pick `MODE`, ref/output folders, optional custom `STEPS` |
| 4b Open dataset (optional) | builds per-speaker refs from LibriTTS-R (CC BY 4.0) |
| 5 Single clone | train one reference |
| 6 Batch mode | train every wav in `REF_DIR`, resume-safe, appends `results.csv` per voice |
| 7 Verify & listen | inline audio + ECAPA cosine + results table & sim-vs-steps plot |

**Custom step counts:** set `STEPS` in Cell 4 (e.g. `1400`) to override the mode
preset. `STEPS >= 600` uses the long recipe (base_lr 0.003 + ReduceLROnPlateau),
below that the standard recipe (base_lr 0.004, linear warmup→decay). Everything
else (vocoder_steps=5, RAW ECAPA target, grad_clip 1.0, 0.05 early stop) is
identical.

**Open dataset instead of own refs:** Cell 4b downloads LibriTTS-R
(`dev_clean` ~1.2 GB / ~40 speakers, or `train_clean_100` ~9 GB / 247 speakers)
and concatenates each speaker's cleanest utterances into one ~25–30 s reference
wav under `/content/dataset_refs/` (capped at `MAX_SPEAKERS`, default 25).
VCTK (CC BY 4.0, 109 speakers) works the same way if you build refs yourself.

## Modes (iterations / rough time)

| Mode | Iters | ~A100 | ~CPU (for reference) |
|---|---|---|---|
| `paid_parity_fast` | 120 | ~1–2 min | ~8 min |
| `paid_parity` (default) | 400 | ~3–6 min | ~28 min |
| `paid_parity_clean` | 400 | ~3–6 min | ~28 min (norm-capped, cleanest render) |
| `paid_parity_long` | 1000 | ~8–15 min | ~70 min |
| `paid_parity_xlong` | 1800 | ~15–25 min | ~2 h |
| `paid_parity_max` | 2000 | ~17–30 min | ~3–4 h |

Plus ~2–4 min one-time setup per session (ONNX→PyTorch conversion + smart-init
embedding cache). Identity climbs with iterations (~0.44 ECAPA cos @400 →
~0.55 @1000 measured); the metric plateaus around 0.60–0.65 for hard voices.
A100 times are estimates — check the `it/s` the trainer prints.

## Results

Trained styles land in **`MyDrive/ECAPAFlow/clones/`** as
`<ref-name>__<mode>.json`. Batch mode also appends one row per finished voice
to **`clones/results.csv`** (timestamp, ref_file, speaker, mode, steps_planned,
steps_run, early_stopped, init_sim_pct, final_sim_pct, minutes_elapsed,
out_json) — safe against runtime disconnects. Each JSON carries full training
metadata (`final_sim_pct`, mode, iters, lr, device…).

## Import into ECAPAFlow

Download the `.json` from Drive, start ECAPAFlow (`run.bat`), click
**Upload Voice** and pick the `.json` — it is imported as a 1:1 trained voice
(no blending, no re-cloning) and the tab speaks with exactly that voice.
