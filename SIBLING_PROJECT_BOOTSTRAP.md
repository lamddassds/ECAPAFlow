# ECAPAFlow — Structural Report (for bootstrapping a sibling project)

Purpose of this document: let another AI assistant reuse the **UI layer as-is**
in a new project while replacing the **engine layer** (currently Supertonic 3
TTS + ECAPA-TDNN speaker encoder) with a different backend. The report is
organized so "UI layer" and "engine layer" are never mixed in the same
section.

---

## 1. UI Stack

**No frontend framework.** This is hand-written HTML + CSS (inline styles) +
vanilla JavaScript. No React/Vue/Svelte/Gradio/Electron, no build step, no
bundler, no `package.json`, no npm dependency at all.

- `static/index.html` — a single static HTML shell with `<style>` for
  `@font-face` declarations and a few hover-state CSS classes. All layout is
  inline `style="..."` on elements with fixed `id`s. The body contains one
  `<script src="/static/app.js">` tag.
- `static/app.js` (1411 lines) — a hand-rolled "vanilla reactive" pattern:
  one global `state` object (plain JS object, no proxy/observable), and a set
  of `render*()` functions that each `innerHTML = ''` and rebuild a DOM
  subtree from `state` on demand. Event handlers mutate `state` then call the
  relevant `render*()` function(s) directly — there is no virtual DOM, no
  diffing, no reactivity system.
- Fonts are self-hosted variable `.woff2` files under `static/fonts/`
  (Outfit for UI text, JetBrains Mono for numeric/monospace displays) split
  into unicode-range subsets — no CDN, no Google Fonts link.
- **Backend that serves it**: FastAPI (Python). `app.py` defines a single
  `GET /` route that reads `static/index.html` from disk, cache-busts the
  `app.js` script tag with the file's mtime (`?v=<mtime>`), and returns it as
  `HTMLResponse`. `/static` is mounted via `StaticFiles` for `app.js` and the
  font files.
- **How it's launched**: `python app.py` (see `run.bat`). This calls
  `uvicorn.run(app, host, port)` directly (no `uvicorn` CLI invocation, no
  reload flag — it's a plain production-style ASGI run in-process).
  - Default port `7873`, auto-incrementing through `7874`–`7876` if busy
    (`_port_in_use` probes each candidate). Overridable via `PORT` or
    `ECAPAFLOW_PORT` env vars.
  - Default host `0.0.0.0` (LAN-reachable — the app prints a phone-friendly
    LAN URL at startup); set `ECAPAFLOW_HOST=127.0.0.1` for localhost-only.
  - Opens the default browser automatically ~2s after boot unless
    `ECAPAFLOW_NO_BROWSER=1` is set.
- Relevant versions (see `requirements.txt`): `fastapi>=0.100`,
  `uvicorn[standard]>=0.27`, `python-multipart>=0.0.7` (needed for the
  `Form(...)`/`File(...)` multipart endpoints the UI posts to).

**Takeaway for reuse**: the UI layer is exactly two files
(`static/index.html`, `static/app.js`) plus the `static/fonts/` folder. It
talks to the backend exclusively through the JSON/multipart HTTP endpoints
listed in §4/§5 below (all under `/api/...`) — nothing in the frontend
imports Python or touches the engine directly. A sibling project can drop
these two files in unchanged as long as the new backend exposes the same
route shapes (or the JS is adjusted to match new ones).

---

## 2. UI Layout (element-by-element)

Single-page layout, dark theme (`#060606` background, `#f6f6f2` primary
text), centered column (`max-width: 900px`), vertically centered in the
viewport. Top to bottom:

### 2.1 Voice tabs row (`#voice-tabs`)
A wrapping flex row of pill-shaped buttons, one per voice in the registry,
rendered by `renderVoices()`. Each pill contains, left to right:
1. **Preview play/pause button** — small circular icon button (20×20px,
   1px border-radius:50%), plays a synthesized self-introduction for that
   voice. Shows a pulsing animation while loading.
2. **Checkmark icon** — shown only if the voice `cloned === true`.
3. **Voice name** (plain text span).
4. **Language-tag `<select>`** — a native HTML dropdown, options "Any" +
   the 12 UI languages (see §2.5); tags the voice for "Auto Voice" mode.
   Untagged shows dim gray text; tagged shows brighter text.
5. **Close (×) button** — removes the voice tab (calls `DELETE
   /api/voices/{id}`).

The active tab has a brighter border (`rgba(255,255,255,0.85)`) and a subtle
background tint; inactive tabs are `#0e0e0e` with a faint border.

After the voice pills, two more pills are always appended:
- **"+ Upload Voice"** (dashed border) — opens a hidden `<input
  type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.json">` file picker.
  Accepts either an audio reference clip or a Supertonic voice-style JSON.
- **"🎙 Record Voice"** (dashed border) — toggles browser mic recording
  (`MediaRecorder`); while recording, this pill turns into a solid-border,
  pulsing-dot "Recording 0:0X — tap to stop" pill (auto-stops at 45s).

### 2.2 Empty state (`#empty-state`)
Shown instead of the main panel when there are zero voices: a dashed-border
box, centered text — "No voice selected" (bold) + a one-line hint to
upload/record a reference clip.

### 2.3 Main panel (`#main-panel`) — shown once a voice is selected

**Engine status line** (top): a small lightning-bolt SVG icon + static
label text ("Supertonic 3 | Lightning Fast, On-Device, 31-Language TTS")
+ a right-aligned `#engine-status` span showing the live device string
(e.g. "DirectML AMD Radeon(TM) 860M Graphics") once loaded, or a pulsing
loading message before that.

**Title row**: the active voice's name as a large (30px, weight 600)
editable title (`#title-block` — click the pencil icon to turn it into a
text `<input>`, Enter/blur commits a rename via `PATCH
/api/voices/{id}`), plus a right-aligned **"Source Voice" pill button**
(`#source-btn`) showing `Source Voice 0:0X` / `Preview 0:0X` (duration)
with a play/pause icon — plays the voice's raw reference clip (or its
built-in preview for native voices). Hidden entirely for voices with no
playable audio.

**Script input** (`#script-frame`): a large borderless `<textarea
id="script">` (25px font, 5 rows, no visible box — just a 3px white
left-border accent), placeholder "Type or paste the text you want to
hear...", `maxlength=100000`.

**Row below the textarea** (content-type + language + char count):
- **Content-type tabs** (`#content-types`): 5 plain-text labels — Freeform,
  Announcement, Chatbot, Article, Podcast — the active one is bold +
  underlined. Clicking one replaces the script textarea with a canned
  sample paragraph for that type (client-side constant strings, not
  server data).
- **Language selector** (`#lang-toggle` / `#lang-menu`): a small globe-icon
  + current language name + chevron, click opens a floating dropdown panel
  (`position:absolute`, dark card, listing all 12 languages + "Auto" with
  the selected one highlighted).
- **Character counter** (right-aligned): `{count}/100000` in monospace.

**Mode buttons + toggles row**:
- **Quality mode buttons** (`#mode-buttons`): 3 pill buttons — "Speed" (8
  steps), "Medium" (16 steps), "Quality" (32 steps) — each showing the step
  count and, once measured, the live RTF (real-time factor) multiplier
  under the label, e.g. "16 steps · ×0.27". Active mode has a bright
  border. Clicking sets the "Quality" slider (see below) to that step
  count.
- **"Auto Voice" toggle pill** (`#auto-voice-toggle`): a status-dot +
  label ("Auto Voice ON"/"OFF"). When on, each paragraph is spoken by
  whichever voice tab was tagged with that paragraph's detected language.
- **"Streaming" toggle pill** (`#stream-toggle`): same visual pattern —
  "Streaming ON"/"OFF". Controls whether playback starts before generation
  finishes.

**Sliders row** (`#sliders`): custom-built (non-native) horizontal sliders,
each a labeled column with a 150px-wide draggable track (a thin line +
white fill + circular knob) and a monospace value readout to the right.
Three sliders live here:
1. **Quality** (4–64 steps, integer)
2. **Speed** (0.5–2.0×, step 0.05)
3. **Silence Duration** (0–1.0s, step 0.05)

Sliders are pointer-drag driven (`pointerdown`/`pointermove`/`pointerup`),
not native `<input type=range>`.

**Voice Cloning section** (`#clone-section`, boxed panel with border +
subtle background tint):
- Header row: a small mic-icon + "Voice Cloning" label, right-aligned
  `#clone-status` text (monospace, shows clone timing/similarity — see §5).
- **ECAPA Encoder Steps slider** (`#clone-sliders`) — same custom-slider
  widget as above, range 1–32 (dynamic max from server config), this is the
  4th slider but visually mounted inside the clone box instead of the main
  slider row.
- **"Clone Voice" button** (`#clone-btn`) — solid white pill button, right
  side of the box. Label changes to "Cloning…", "Native Voice ✓",
  "Imported 1:1 ✓", or "Voice Cloned ✓" depending on voice type/state.

**Bottom action row**:
- **"Regenerate" button** (`#regen-btn`) — outlined pill button with an
  optional spinning-ring SVG (`#regen-spinner`) shown while generating;
  label toggles between "Regenerate" / "Cloning voice…" / "Stop".
- **Playback bar** — a circular play/pause button (`#result-play-btn`,
  42px), current time (`#result-current`, monospace "0:00"), a 4px-tall
  progress track with two overlaid fills — a dim "buffered" bar
  (`#result-buffered`) and a bright "played" bar (`#result-progress`) — then
  total/estimated duration (`#result-duration`, e.g. "~0:12" while
  streaming, exact once done), an inline RTF readout (`#result-rtf`,
  monospace, e.g. "RTF 0.42"), and a circular **download button**
  (`#download-btn`, hidden until a finished result exists) that saves the
  buffered audio as a `.wav`.

### 2.4 Toast notifications (`#toast`)
A fixed-position (`bottom:28px; right:28px`) dark card that fades in/out for
transient messages (errors, "voice cloned in X s", stall warnings, etc.).
Not a persistent layout element — `display:none` until `showToast()` is
called.

### 2.5 Data-driven lists (client-side constants, overridable by the server)
- `CONTENT_TYPES` / `CONTENT_SAMPLES` — 5 canned content types + sample text,
  purely client-side (never sent to or defined by the backend).
- `LANGUAGES` — 12-language menu, seeded client-side but **overwritten** by
  `GET /api/status`'s `languages` field at load time.
- `MODES` — the 3 quality-mode presets, similarly seeded client-side but
  overwritten by `/api/status`'s `modes` field.

---

## 3. File / Folder Structure

```
ECAPAFlow/
├── app.py                    UI GLUE — FastAPI app: routes, request/response
│                             shaping, streaming synthesis orchestration,
│                             text chunking/pacing, RTF-history model.
│                             Imports ecapa.py, engine.py, voices.py,
│                             normalizer.py. Serves static/index.html + mounts
│                             /static. THIS is the file a sibling project's
│                             backend replaces/rewrites.
│
├── engine.py                  ENGINE LAYER (TTS synthesis) — SupertonicEngine:
│                             ONNX provider detection (CUDA>DirectML>CPU),
│                             background model load, voice-style cache,
│                             synthesize(text, voice, steps, speed, lang,
│                             silence_duration) -> (wav, duration, infer_time).
│                             This is the "different backend engine" referred
│                             to for TTS — swap this file to change the TTS
│                             engine while keeping ECAPA cloning as-is.
│
├── ecapa.py                    ENGINE LAYER (speaker encoder) — THE ECAPA-TDNN
│                             SEAM. Loads speechbrain's EncoderClassifier,
│                             embeds reference audio, ranks/blends built-in
│                             voice styles by cosine similarity, verifies
│                             achieved similarity. See §4 for the exact
│                             function boundary to replace.
│
├── voices.py                   GLUE / DATA LAYER — voice registry (JSON file
│                             persistence at data/voices/registry.json):
│                             CRUD for voice tabs (upload/record/built-in/
│                             imported), built-in voice seeding, file-path
│                             helpers. Engine-agnostic — only stores IDs,
│                             filenames, and whatever dict `clone_info` the
│                             engine layer hands it.
│
├── normalizer.py               GLUE — regex-based DE/EN text normalizer
│                             (numbers, dates, currency, markdown stripping)
│                             run on script text before it reaches the TTS
│                             engine. Independent of both UI and the ECAPA
│                             encoder; only feeds engine.py's synthesize().
│
├── static/
│   ├── index.html             UI LAYER — page shell (see §1/§2)
│   ├── app.js                 UI LAYER — all frontend logic (see §1/§2)
│   └── fonts/*.woff2          UI LAYER — self-hosted Outfit + JetBrains Mono
│
├── data/                      RUNTIME STATE (gitignored-style; not code)
│   ├── refs/                  uploaded/recorded reference audio clips
│   ├── voices/                registry.json + per-voice style_<id>.json
│   └── cache/                 voice_embeddings.npz, rtf_history.json,
│                             previews/*.wav, spkrec-ecapa-voxceleb/ (the
│                             downloaded speechbrain model checkpoint)
│
├── tests/                     Standalone (no pytest required) regression
│   ├── test_detect.py          scripts: language auto-detect, normalizer
│   ├── test_normalizer.py      rules, and the streaming chunk-scheduler
│   └── test_stream_plan.py     math. Not related to the ECAPA seam.
│
├── colab/                     Separate Google-Colab training pipeline
│   ├── ECAPAFlow_VoiceForge.ipynb   (LoudFlow's heavier gradient-trained
│   ├── train_paid_parity.py          "paid_parity" clone method — NOT part
│   └── voice_clone_repo.zip          of the live app's request path at all)
│
├── benchmark_steps.py         Dev tool: sweeps ECAPA-steps values against a
│                             reference clip, used to calibrate
│                             DEFAULT_ECAPA_STEPS in app.py. Calls ecapa.py
│                             directly, bypasses the FastAPI app.
│
├── requirements.txt           see §6
├── run.bat / firewall_freigeben.bat   Windows launch/firewall helper scripts
└── README.md                  Project narrative + measured benchmark results
```

**Boundary summary**: `static/` is 100% UI layer. `app.py` is glue that both
serves the UI and defines the HTTP contract. `engine.py` is the swappable TTS
backend. `ecapa.py` is the swappable speaker-encoder backend (the ECAPA-TDNN
seam). `voices.py` and `normalizer.py` are engine-agnostic support modules
that a sibling project can likely keep unchanged.

---

## 4. Engine Interface — the ECAPA-TDNN Seam

This is the exact boundary to reimplement for a different speaker-embedding
backend. Everything lives in `ecapa.py`; no other file touches the
speechbrain API, torch tensors, or raw embedding vectors directly.

### 4.1 Model load
```python
def load_encoder() -> Any:
    """Loads (once, thread-safe, module-level cached) speechbrain's
    EncoderClassifier from source="speechbrain/spkrec-ecapa-voxceleb"
    (22.3M params, Apache 2.0). Device: CUDA if available else CPU
    (DirectML is NOT used for this model — only for the TTS engine)."""
```
Called from `app.py`'s background prewarm thread (`_prewarm()`,
`app.py:325-346`) and lazily by every embedding call below.

### 4.2 The core embedding call (the actual seam)
```python
def embed_reference(audio_path: str, steps: int = 8) -> tuple[np.ndarray, int]:
    """
    Input:  audio_path — path to a reference audio file (any format
            soundfile/librosa can read). Internally loaded via
            librosa.load(sr=16000, mono=True), silence-trimmed
            (top_db=30), capped at 30s.
    Output: (embedding, effective_steps)
              embedding       — numpy float32 array, shape (192,),
                                L2-NORMALIZED (unit length) so downstream
                                code can use a plain dot product as cosine
                                similarity.
              effective_steps — int, how many of the requested N windows
                                were actually distinct (can be < steps for
                                short clips).
    Behavior: steps=1 (or clip shorter than one 3s window) → single whole-
    clip encoder pass. steps>1 → N evenly-spaced 3-second windows batched
    into ONE encoder forward call, then averaged before final L2-norm.
    """
```
This is the single most important function to reimplement for a different
speaker encoder (e.g. a different pretrained model, different embedding
dimensionality, resemblyzer/WavLM/TitaNet/etc.). **The 192-dim size is not
hardcoded anywhere outside this file** — every downstream consumer only ever
sees the *output of* the higher-level functions below, never the raw vector
directly, so dimensionality can change freely as long as internal
consistency (target vs. voice-table embeddings) is kept.

### 4.3 Embedding synthesized (not reference) audio
```python
def _embed_wav_16k_np(wav_44k_1d: np.ndarray) -> np.ndarray:
    """Input: a 1-D float32 numpy array of audio at 44100 Hz (Supertonic's
    native output rate). Resamples to 16kHz via torchaudio, single encoder
    pass. Output: (192,) float32 — NOT re-normalized here (callers
    normalize as needed, see verify_clone below)."""
```
Used to (a) build the cached embedding table for the 10 built-in TTS voices,
and (b) embed a freshly-synthesized sample when verifying an achieved
clone's similarity.

### 4.4 Higher-level orchestration built on 4.2/4.3 (what `app.py` actually calls)
```python
def build_voice_embedding_cache(engine, force=False) -> dict[str, np.ndarray]:
    """One embedding per built-in TTS voice (synthesizes a fixed sample
    sentence per voice, embeds it, caches to data/cache/voice_embeddings.npz).
    `engine` is the TTS engine singleton — only used to call .synthesize()
    and .get_voice(); this function is the ONE place the ECAPA layer talks
    to the TTS engine layer."""

def smart_init_clone(engine, ref_audio_path: str, ecapa_steps=8, top_k=3,
                      temperature=0.1) -> dict:
    """THE main clone entry point, called from app.py's _clone_voice_now()
    (app.py:496-573).
    1. embed_reference(ref_audio_path, steps=ecapa_steps) -> target embedding
    2. cosine-similarity target against every cached built-in voice embedding
    3. softmax-weight the top-`top_k` by similarity (temperature-scaled)
    4. blend those voices' TTS style tensors (ttl/dp) by the softmax weights
    Returns: {
      "ttl": np.ndarray (1,50,256) f32,       # TTS-engine-specific style
      "dp":  np.ndarray (1,8,16) f32,          # tensors — NOT ECAPA outputs,
                                                # these belong to the TTS
                                                # engine's voice-style format
      "target_emb": np.ndarray (192,) f32,     # L2-normalized — THE ECAPA
                                                # output that matters for a
                                                # swap
      "ranking": [{"voice": str, "sim": float, "weight": float|None}, ...],
                                                # all built-in voices, sorted
      "est_sim_pct": float (0-100),
      "effective_steps": int, "requested_steps": int,
      "timings_ms": {"embed", "rank", "blend", "total"},
    }"""

def verify_clone(engine, ttl, dp, target_emb) -> dict:
    """Synthesizes ONE sample with the blended style, re-embeds it via
    _embed_wav_16k_np, cosine-compares to target_emb.
    Returns: {"verified_sim": float (-1..1), "verified_sim_pct": float
    (0-100), "verify_ms": float}."""
```

### 4.5 Exact call sites in `app.py` (where the seam is exercised)
- `app.py:333-335` — `_prewarm()`: `ecapa_mod.load_encoder()`,
  `ecapa_mod.warmup_encoder()`, `ecapa_mod.build_voice_embedding_cache(engine)`
  (background thread at server startup).
- `app.py:510` — `_clone_voice_now()`: `ecapa_mod.smart_init_clone(engine,
  str(ref), ecapa_steps=ecapa_steps)` — invoked from both the explicit
  "Clone Voice" button (`POST /api/voices/{id}/clone`) and auto-clone-on-
  first-synthesize inside `POST /api/synthesize`.
- `app.py:550` — async post-clone `_verify()` thread: `ecapa_mod.verify_clone(
  engine, ttl, dp, target)`.
- `app.py:749-751` — `_verify_import()` (for imported 1:1 style JSONs with an
  attached reference clip): `ecapa_mod.embed_reference(...)` +
  `ecapa_mod.verify_clone(...)`.
- `app.py:1189-1190` — `/api/benchmark/ecapa_steps` sweep endpoint: calls
  `smart_init_clone` + `verify_clone` in a loop over step-count values (dev/
  calibration tool, not part of the normal user flow).

### 4.6 What a replacement backend must preserve
To swap the encoder without touching `app.py`/`voices.py`/the frontend,
reimplement these public names in `ecapa.py` (or a drop-in replacement
module imported the same way — `import ecapa as ecapa_mod`):
`status()`, `load_encoder()`, `warmup_encoder()`, `start_background_load()`,
`analyze_ref_audio(path)` (audio QC, not encoder-specific — can stay as-is),
`embed_reference(path, steps)`, `build_voice_embedding_cache(engine, force)`,
`smart_init_clone(engine, ref_audio_path, ecapa_steps, top_k, temperature)`,
`verify_clone(engine, ttl, dp, target_emb)`, `save_style_json(...)` (this one
is actually TTS-style-format-specific, not encoder-specific — it round-trips
whatever `ttl`/`dp` tensors the TTS engine produces).

---

## 5. Results / Metrics Display

No charts, no graphs — everything is inline text, badges, and tooltips
consistent with the minimalist UI.

### 5.1 Where it surfaces in the UI
| Location | What's shown | Source |
|---|---|---|
| Voice tab `title` attribute (hover tooltip) | `verified ECAPA similarity {pct}%` | `voice.clone_info.verified_sim_pct` |
| `#clone-status` line (Voice Cloning box) | e.g. `1.62s · est 51% · verified 48% · F1+M2+F3` | `voice.clone_info` (timings_ms.total, est_sim_pct, verified_sim_pct, ranking top-3) |
| Toast after clicking "Clone Voice" | `Voice cloned locally in 1.62s — 24 ECAPA steps, est. similarity 51% (blend F1+M2+F3). Nothing uploaded or purchased.` | response of `POST /api/voices/{id}/clone` |
| Toast after auto-clone during Regenerate | `Voice cloned locally in 1.62s — audio generated.` | `X-CLONE-MS` response header |
| `#result-rtf` (bottom of playback bar) | `RTF 0.42` | computed client-side from `/api/synthesize/stats/{sid}` (TTS engine performance, not an ECAPA metric) |

There is currently **no UI surface at all** for the `/api/benchmark/ecapa_steps`
sweep endpoint's output (steps-vs-similarity comparison table) — it exists
only as a JSON API + the standalone `benchmark_steps.py` dev script.

### 5.2 Underlying data structures (before hitting the UI)

**`smart_init_clone()` return** (in-memory, Python dict — see full shape in
§4.4). Persisted subset into the voice registry as `clone_info`:
```json
{
  "est_sim_pct": 51.3,
  "ranking": [ {"voice": "F1", "sim": 0.7123, "weight": 0.78}, ... up to 5 ],
  "ecapa_steps": 24,
  "timings_ms": {"embed": 1420.3, "rank": 0.4, "blend": 0.2, "total": 1421.1},
  "verified_sim_pct": null   // filled in asynchronously, see below
}
```

**`verify_clone()` return** — merged into the same `clone_info` dict a few
seconds later (async background thread), adding `verified_sim_pct` and
`verify_ms`.

**Registry persistence**: `data/voices/registry.json`, one object per voice
tab, e.g.:
```json
{
  "id": "a1b2c3d4e5f6", "name": "Pixal", "type": "uploaded",
  "ref_filename": "ref_9f8e7d6c5b.wav", "duration_s": 5.9,
  "cloned": true, "style_filename": "style_a1b2c3d4e5f6.json",
  "clone_info": { /* as above */ }, "ref_stats": { /* analyze_ref_audio() */ },
  "lang": null
}
```

**API exposure**:
- `GET /api/voices` → `{"voices": [ full registry entries incl. clone_info ]}`
- `POST /api/voices/{id}/clone` → `{"voice": {...}, "clone": {est_sim_pct,
  ranking (all 10, not just top-5), effective_steps, timings_ms}}`
- `POST /api/benchmark/ecapa_steps` → `{"voice": name, "device": str,
  "results": [ {steps, effective_steps, clone_ms, embed_ms, est_sim_pct,
  verified_sim_pct, top3: [voice,voice,voice], weights: [f,f,f]}, ... ]}`

---

## 6. Dependencies

From `requirements.txt` (there is no `package.json` — the frontend has zero
npm/JS dependencies):

```
supertonic>=1.3.0
numpy>=1.26
soundfile>=0.12
librosa>=0.10
fastapi>=0.100
uvicorn[standard]>=0.27
python-multipart>=0.0.7

speechbrain>=1.0
torch>=2.1
torchaudio>=2.1

onnxruntime-directml ; platform_system == "Windows"
# (README notes: swap for onnxruntime-gpu on CUDA machines — engine.py
#  auto-selects CUDA > DirectML > CPU from whatever is actually installed)
```

### UI-specific (transport/serving only — keep these regardless of engine swap)
- `fastapi>=0.100` — HTTP framework, request routing, `HTMLResponse`/
  `StreamingResponse`.
- `uvicorn[standard]>=0.27` — ASGI server that actually runs the app.
- `python-multipart>=0.0.7` — required for FastAPI's `Form(...)`/
  `File(...)` multipart parsing (voice upload, clone requests, synthesize
  requests all use multipart form bodies, not JSON).
- **Frontend has no dependencies**: no bundler, no framework package, fonts
  are vendored `.woff2` files committed to `static/fonts/`.

### Engine-specific (speaker encoder — the ECAPA-TDNN seam, §4)
- `speechbrain>=1.0` — provides `EncoderClassifier.from_hparams(source=
  "speechbrain/spkrec-ecapa-voxceleb")`, the actual ECAPA-TDNN model wrapper.
- `torch>=2.1` — required by speechbrain; also used directly in `ecapa.py`
  for tensor construction/batching.
- `torchaudio>=2.1` — used narrowly, only for `Resample` in
  `_embed_wav_16k_np`; audio *loading* deliberately avoids
  `torchaudio.load` (comment in code: needs `torchcodec` on recent
  torchaudio builds) in favor of `librosa.load`.

### Engine-specific (TTS synthesis — separate seam, `engine.py`, not the one this report focuses on but relevant if the sibling project also swaps TTS)
- `supertonic>=1.3.0` — the TTS model/inference package itself
  (`from supertonic import TTS`, `supertonic.core.Style`, etc.).
- `onnxruntime-directml` (Windows) / `onnxruntime-gpu` (CUDA, installed
  manually per README) — ONNX Runtime execution providers Supertonic runs
  on. `engine.py`'s `detect_providers()` picks CUDA > DirectML > CPU from
  whatever's installed at runtime.

### Shared / used by both layers
- `numpy>=1.26` — array math throughout (embeddings, PCM audio, style
  tensors).
- `soundfile>=0.12` — WAV read/write (reference audio, synthesized output).
- `librosa>=0.10` — resampling, silence trimming, audio duration reads (used
  by both the ECAPA reference-loading path and general audio-quality checks
  in `ecapa.analyze_ref_audio`).

---

## Summary for the sibling-project AI

- **Reuse unchanged**: `static/index.html`, `static/app.js`, `static/fonts/`,
  and the FastAPI route *shapes* they depend on (`/api/status`,
  `/api/voices*`, `/api/synthesize*`, the multipart form fields, the custom
  `X-*` response headers used for streaming metadata).
- **Reuse likely unchanged**: `voices.py` (registry persistence),
  `normalizer.py` (text normalization), `app.py`'s request-shaping/streaming
  logic — these don't know or care which speaker encoder or TTS engine is
  underneath.
- **Replace for a different speaker-encoder backend**: `ecapa.py`, keeping
  the function signatures in §4.6 intact (`embed_reference`,
  `smart_init_clone`, `verify_clone`, etc.) so `app.py` needs zero changes.
- **Replace for a different TTS backend** (separate concern): `engine.py`'s
  `SupertonicEngine.synthesize(text, voice, steps, speed, lang,
  silence_duration) -> (wav_1d_float32, duration_s, infer_time_s)` is that
  seam — out of scope for "ECAPA-TDNN encoder" per se, but relevant if the
  sibling project is a full engine swap rather than just the encoder.
