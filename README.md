# ECAPAFlow — Supertonic Voice Builder

Pixel-perfect implementation of the Claude Design "Supertonic Voice Builder"
prototype, with the two faked features replaced by real engines:

| Prototype (faked)                          | ECAPAFlow (real)                                          |
|--------------------------------------------|-----------------------------------------------------------|
| `regenerate()` → `window.speechSynthesis`  | **Supertonic 3** (99M params, ONNX) — CUDA > DirectML > CPU |
| `cloneVoice()` → local flag + toast        | **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`, 22.3M) smart-init blend |

## The experiment

LoudFlow's production clone path (`supertonic3_test/clone.py`, `paid_parity*`
modes) uses ECAPA smart-init **plus** 120–1500 gradient iterations
(8 min – 2 hrs) to reach 0.72–0.79 ECAPA cosine similarity.

ECAPAFlow runs the **smart-init step alone** — no gradient training:

1. Quality-gate the reference clip (same thresholds as LoudFlow).
2. Build the target speaker embedding from the RAW reference
   (silence-trimmed, NOT gain-normalized — the paid_parity lesson).
3. **ECAPA encoder steps** (the 4th slider): embed N evenly-spaced 3s
   windows in one batched encoder forward, average them.
4. Rank the 10 built-in Supertonic voices by cosine similarity
   (their embeddings are synthesized once and cached to disk).
5. Softmax-blend the top-3 styles (temperature 0.1) → the cloned voice.

This is a standalone experiment — it does not replace the LoudFlow
`paid_parity` pipeline. Both stay separate until a real side-by-side
listening comparison has been done.

### Measured results (laptop: Ryzen AI 7 350, DirectML/CPU — the SLOW target)

Warm clone latency (after one-time startup warmup):

| Reference                    | ECAPA steps | Clone time | Verified ECAPA sim* |
|------------------------------|------------:|-----------:|--------------------:|
| Recorded Voice (2s, clean)   | 1 (clip ≤3s)|  ~50–80 ms | **48–54 %**          |
| pixal_0000_clean (5.9s)      | 24          |    ~1.6 s  | 4–8 %                |
| pixal_0000_clean (5.9s)      | 12          |    ~0.85 s | (embedding 99.9 % converged) |
| loudflow_stream (30s, quiet) | 12          |    ~0.8 s  | ~6–10 %              |

\* Verified = synthesize a sample with the blend, ECAPA-compare to the target
(runs asynchronously after each clone; stored in the voice registry and shown
as the voice-tab tooltip). For scale: paid_parity reaches 72–79 % after
8 min – 2 hrs; even two real recordings of the same person rarely exceed ~85 %.

**Read "warm" strictly** — the first clone in a fresh process also pays
`load_encoder()`, measured 2026-07-29 at **19 s** one-time (speechbrain +
torch init + the 81 MB checkpoint). A sibling project reported "28 s to clone"
and that is what they were seeing; the embedding itself was ~1.5 s of it.
The app pays this during startup prewarm, so users don't — but anything that
imports `ecapa` fresh does.

**Clone time does NOT scale with reference length.** Every ECAPA step is a
fixed 3 s window and all N windows go through **one batched encoder forward**,
so cost tracks the step count, not the clip. Measured warm, 24 steps:
5.9 s clip → 1.71 s, 15 s clip → **1.45 s**. Clips ≤3 s collapse to a single
window (`effective_steps` = 1) and finish in ~75 ms regardless of the slider —
that, not clip length, is why the 2 s recording above clones in 50–80 ms.

**A third verified data point (2026-07-29, sibling project):** Lauro's own
28.8 s German reference → estimated 16.4 %, **verified 13.1 %** (blend
M1×0.45, M4×0.28, M2×0.27). That sits between the 4–8 % distant case and the
48–54 % close case and confirms the mechanism: smart-init can only return the
nearest mixture of the ten built-ins, and his voice is not near them. The
practical conclusion for any product built on this: **a fast tier should ship
built-in voices only**, or route real cloning through the gradient-trained
`colab/` path (72–95 %).

**Interpretation:** smart-init-only quality depends almost entirely on how
close the target speaker is to the 10 built-in voices. A real male recording
landed at ~51 % (blend M2×0.78) in **under 0.1 s**. The pixal voice is far
from every built-in (best raw cosine 0.14), so its blend can't get close —
the same references that force paid_parity to its longest schedules.
The sub-second target is met: ≤12 steps stays under 1 s even on CPU;
on the RTX 4060 (CUDA) everything is far faster.

### Why the ECAPA-steps default is 24

`benchmark_steps.py` measures how the target embedding converges toward a
dense 64-window gold standard (`data/cache/benchmark_steps.json`):

- pixal 5.9s: cos vs gold 0.971 @1 → 0.9938 @4 → 0.9988 @8 → 0.99944 @12 →
  0.99967 @16 → **0.99984 @24** → 0.99991 @32.
- Between 16 and 24 steps the blend's third voice still flipped to the
  converged choice (a 29 %-weight member changing gender — audible).
  24 → 32 shifts blend weights by <0.5 % (inaudible).
- Clips ≤3s always collapse to 1 window (reported as `effective_steps`).

So 24 is the highest value that still audibly improves quality (quality
over speed, per the brief). Slide down to ≤12 for guaranteed sub-second
clones on CPU. Range 1–32.

## Run it

```
run.bat        (uses C:\Users\lauro\.venvs\localvoice — everything installed)
```
or `python app.py` in any env with `requirements.txt` installed.
Opens http://localhost:7873. First start downloads nothing on this machine
(Supertonic 3 + ECAPA are already in the HF cache) and spends ~30 s on a
one-time warmup + built-in-voice embedding cache (persisted to
`data/cache/voice_embeddings.npz`).

### Hardware targets

- **Desktop (RTX 4060)**: install `onnxruntime-gpu` (synthesis via CUDA) and
  a CUDA build of torch (ECAPA via CUDA). The engine auto-selects
  CUDA > DirectML > CPU at startup; no config needed.
- **Laptop (Ryzen AI 7 350)**: `onnxruntime-directml` (synthesis via DirectML)
  + CPU torch for ECAPA. This is the configuration everything above was
  measured on.

## Streaming synthesis

`/api/synthesize` streams: a WAV header with maxed-out sizes, then raw PCM16
per sentence chunk as the engine finishes it (FastAPI `StreamingResponse`,
sync generator iterated in a threadpool). The frontend starts playback once
~0.3 s is buffered — first audio after ~1.5–2.5 s instead of after the whole
script.

Design decisions (measured on the laptop, DirectML Radeon 860M):

- **Every chunk renders at the full slider quality.** A step ramp (first
  chunks at 5/10 steps for ~2 s first audio) was tried and REMOVED — the
  reduced-step chunks were audibly worse. The TTFA lever that replaced it is
  a **shorter first chunk at full quality**: the first chunk is exactly one
  sentence (`allowed = 1` in the stream loop), every chunk after it is sized
  by the buffer controller. TTFA is **~1–1.5 s at 16 steps**; after that
  playback is gapless. (An older "~5 s at 16 steps" figure predates the
  one-sentence first chunk — it is no longer the behaviour.)
- **Chunks ≤300 chars** (120 for Korean) — matches the pip package's internal
  limit, so one engine call = one model pass = one streamable piece. (This
  does not change quality: the package always split at 300 internally.)
- **RTF vs steps.** Re-measured 2026-07-29 (148-char chunk, latent 114,
  7.91 s of audio, speed 1.25, medians of interleaved reps). The number that
  matters in production is the middle column — see the shape penalty below:

  | steps | DirectML *privileged* | DirectML *penalized* ← production | CPU |
  |---:|---:|---:|---:|
  | 2  | 0.029 | **0.07** | 0.137 |
  | 4  | 0.045 | **0.10** | 0.240 |
  | 8  | 0.076 | **0.18** | 0.473 |
  | 16 | 0.137 | **0.32** | 0.828 |
  | *fit* | 110 ms + 61 ms/step | 271 ms + 136 ms/step | 387 ms + 390 ms/step |

  An older table here read 2→0.12 … 16→0.60. **Those numbers were wrong** —
  two independent re-measurements (this one and a sibling project's, which
  got 0.406 at 16 steps under slightly different conditions) both contradict
  them. Streaming stays glitch-free well past 16 steps on DirectML.

- **The DirectML shape penalty (the single biggest engine fact).** The first
  tensor shape a session ever runs stays fast for the life of that session;
  **every other shape pays ~2.24× per step, permanently**. It is first-come,
  not size-based. Since every chunk has its own latent length, essentially
  every production chunk is penalized. Verified dead ends, do not re-try:
  padding text+latent to a canonical shape (audio changes completely);
  padding the **latent only** with the text path byte-identical (still not
  invariant — audio correlation 0.25, and the error is identical at every pad
  size); pinning `batch_size=1` so the CFG slice bounds constant-fold
  (numerically identical, zero speed win); `mem_pattern=OFF`, `arena=OFF`,
  `ORT_DISABLE_ALL` (all inert); per-shape bucketing (only one shape per
  session can hold the fast slot). Lead worth chasing: ORT logs *"This model
  has shape massaging nodes that will execute on CPU"* for the vector
  estimator.

- **Engine internals worth knowing.** The ONNX graph bakes in *both* the ODE
  integrator and classifier-free guidance:
  `denoised_latent = (noisy_latent + (1/total_step)·v)·latent_mask` with
  `v = 4.0·v_cond − 3.0·v_uncond`, and the batch is tiled ×2 internally — so
  **every step is already two network evaluations**. The velocity is
  recoverable exactly (`v·mask = (out − x_t)·total_step`), and
  `current_step`/`total_step` are floats entering only via `Div` and
  `Reciprocal`, so fractional timesteps are legal and a Heun step collapses to
  `(x_t + out₂)/2`. It costs two passes per step though, so Heun@8 ≡ Euler@16
  in wall clock — a quality experiment, not a speed one.

- **Measured non-levers** (don't spend time re-deriving these):
  hoisting the five constant vector-estimator inputs to device `OrtValue`s is
  **inert** (75.18 → 75.48 ms/step; independently confirmed by a sibling
  project at +0.3%) — GPU compute is ~75 ms/step against ~15 ms/step of
  enqueue, and DirectML overlaps them. `OMP_NUM_THREADS`/`MKL_NUM_THREADS` in
  `engine.py` are a **placebo**: the shipped onnxruntime contains no OpenMP
  runtime at all; the real knob is `intra_op_num_threads` (currently unset).
  **fp16** conversion of the vector estimator works and is numerically clean
  (correlation 0.99999976 vs fp32) but buys only **1.09×** — its real value is
  halving the model, 256 MB → 128 MB.

- **`run_with_iobinding` is non-blocking on DirectML.** Any benchmark that
  does not force a sync measures enqueue time only — a naive timing here
  attributed 771 ms to a device→host copy that was really the queue draining.
- Sentence gaps / paragraph pauses are yielded between chunks — free
  playback runway, and since 2026-07-27 they carry room tone and breaths
  (see below) instead of digital silence.
- Client abort (Stop button) ends the server generator at the next yield;
  stats are still finalized.

## Sounding human (`prosody.py`, `mastering.py`)

Two post-processing layers sit between the engine and the stream. Both are
plain numpy/scipy on audio the model already produced — no extra inference,
no LLM, a few milliseconds per chunk — and both are switchable in the UI so
any claim here can be A/B'd against the raw engine.

**`prosody.py` — Natural (rhythm).** The engine hands back a separate model
pass per chunk, with a ragged amount of near-silence at each end and no idea
what punctuation caused the boundary. This layer:

- scales every pause by the mark that caused it (comma 0.55×, semicolon
  0.75×, colon 0.60×, period 1.00×, `!` 1.10×, `?` 1.15×, ellipsis 1.60×,
  paragraph = its own base) against the Silence Duration slider, which lands
  on the 150–200 ms / 300–400 ms / 400–600 ms ranges commercial TTS uses;
- adds a **breath debt**: the longer the speaker has gone without resting,
  the longer the next rest (up to +35%);
- wobbles every pause ±11% and every sentence's speaking rate ±3%,
  deterministically (hashed from the text — the same script always renders
  identically, so what you approved is what you download);
- **trims** the model's leftover head/tail silence at −50 dB so this layer,
  not the model's padding, owns the rhythm — and de-clicks the seams with
  6 ms/14 ms raised-cosine fades;
- fills the pauses with the recording's own **room tone** and tucks a
  band-limited **inhale** (350–2400 Hz, ~24 dB under the speech) into the
  tail of any rest long enough to have been a breath;
- splits over-long run-ons at **clause** boundaries and marks the left half
  with a trailing comma, so a sentence that hasn't ended doesn't get a
  sentence-final fall;
- matches chunk-to-chunk level (max 1.5 dB per chunk, so quiet lines stay
  quiet) and replaces the old per-chunk peak scaler with one stream-wide,
  never-increasing gain — the old one made a hot chunk quieter than its
  neighbours all by itself.

**`mastering.py` — Studio (polish).** LoudFlow, on the same engine and the
same voices, sounds more produced because it runs a Pedalboard chain. This
is that idea rebuilt in scipy: 75 Hz high-pass → **split-band** de-esser
(5.8 kHz, ≤4 dB, subtracts the sibilant band rather than ducking the whole
voice) → +2.5 dB presence shelf at 6 kHz → 2:1 soft compression (−20 dB,
5 ms/120 ms) → 3.5% tiny-room early reflections → auto-level toward −18 LUFS
measured the broadcast way (ITU-R BS.1770 K-weighting, gated, cumulative),
clamped to ±4 dB and rate-limited to 0.9 dB per second of audio. Every filter
keeps its state across blocks and the **pauses go through the chain too**, so
reverb tails ring into the silence instead of being chopped at the chunk edge.

Note on the one timing lever: the engine is character-based and its duration
predictor returns a single TOTAL length per chunk, which `speed` divides —
there are no per-phoneme durations to shape. So `rate_scale` spends that one
lever on discourse structure instead: a paragraph's last chunk slows ~3%
(final lengthening), its opening runs slightly under its body, questions and
very short lines run slower, plus a ±2% hashed wobble, all bounded to ±6%.

Measured on the laptop (467-char script, 16 steps, Emma): DC offset gone
(−6·10⁻⁴ → −10⁻⁶), sub-80 Hz energy −65%, presence band +2.1 dB, level
−25.3 → −21 dBFS RMS, longest run of digital-zero silence 1.23 s → 0.03 s,
no clipping. RTF 0.36 → 0.41 — higher only because trimming removed ~4 s of
the model's dead air, i.e. there is less audio for the same inference.

### Adaptive playback buffer (client-side)

Ported from QuantFlow's streaming controller (2026-07-27). The server ships an
opening bid in headers (`X-EST-AUDIO-S`, `X-SPEECH-S`, `X-RTF-HI`) and then
streams as fast as it decodes — it never sleeps and never holds audio back.
The **client** decides when to start:

```
required_cushion = remaining_audio × speechShare × max(0, rtf − 1)
bufferTarget     = min(required_cushion + jitter, MAX_WAIT_S)
```

- `rtf` is the **worst** of three live estimators — cumulative, a 0.8 s recent
  window, and a worst-window decayed by 0.5 per closed window — not an
  average: the failure being defended against is a take whose rate is no
  longer what it was when it started, and an average hides that until the
  buffer is gone. Plus 6% headroom, because being early costs a fraction of a
  second and being late costs a stall.
- The clock starts at **chunk 1**, not at the request: prefill, engine-lock
  wait and auto-clone are time-to-first-audio, not production rate.
- The server's estimate is a **floor that fades to zero over 3 s** of observed
  audio, so a stale history row cannot make the client wait on a machine that
  is visibly keeping up.
- **Pauses are excluded** (`speechShare`): they are pre-rendered silence and
  cost no inference, so billing them at `(rtf − 1)` made paragraph-heavy
  scripts wait seconds longer than the arithmetic requires.
- **One term QuantFlow does not need:** arrival latency. Its generation runs
  near RTF 1.0, so an average rate describes it. Here a chunk is one whole
  model pass whose fixed cost is seconds at high step counts, so a stream can
  average RTF 0.5 and still run the player dry waiting for chunk 2 — measured
  on the first real run of this port, one 3.0 s stall. The cushion is now
  `max(rate_need, chunk_latency × 1.15)`, seeded from `X-CHUNK-LAT-S` and
  then from the client's own measured worst arrival gap.
- When the cushion still thins: playback **glides** slower (floor 0.90, EMA
  0.3 per chunk). Note this is `playbackRate`, i.e. resampling — it shifts
  pitch (~1.8 semitones at full deflection), it does not time-stretch. Below
  a 0.02 s hard floor it takes **one recovery pause sized from what the rest
  of the stream needs** (0.12–1.5 s), counted into the stall metrics rather
  than silently absorbed.

### Normalizer fast path

Also from QuantFlow: `normalize()` is memoized (`lru_cache`, keyed on
`(text, lang)`, whitespace collapse inside so the cached value is finished),
and three early-exit probes decide in one scan each whether there is anything
to rewrite. The abbreviation loop (dozens of `re.sub` passes) now runs only
behind a single compiled alternation over the real abbreviation keys.
Measured: ordinary prose 419 µs → **26 µs** on a cache miss, 0.3 µs cached.
The probes run on the **stripped** text, not the raw text — a zero-width
space inside `z.​B.` hides it from the probe otherwise, which is exactly the
bug the test suite caught the day this was added.

### Auto quality

A fourth mode next to Speed/Medium/Quality: it picks the **highest step
count this machine still renders faster than real time**, using the measured
end-to-end RTF of past streams (`stream_rtf` in `data/cache/rtf_history.json`),
falling back to the nearest measured rung scaled linearly in steps, and only
then to the steady-state model with a penalty. Every finished stream feeds
its real RTF back, so the choice self-corrects. It has to work this way: the
steady-state chunk model amortises the per-call fixed cost over a full-size
chunk and is structurally optimistic — it once cleared a rung at 0.80 that
then streamed at 1.10.

## What's in the UI

Everything from the prototype, wired to reality:

- **Voice tabs** — persisted in `data/voices/registry.json`, ✓ = cloned,
  tooltip = verified similarity. Plus two pills the prototype's data model
  implied but had no UI for: **Upload Voice** (wav/mp3/flac/ogg, also accepts
  Supertonic style JSONs) and **Record Voice** (mic → WAV client-side,
  auto-stop at 45 s).
- **10 native voices** seeded once as ready tabs with human names
  (F1–F5 = Emma, Sophia, Olivia, Mia, Luna; M1–M5 = Liam, Noah, Elias, Ben,
  Leon). No cloning needed; deleted tabs stay deleted
  (`data/voices/.builtins_seeded`). Their **Preview** button plays a
  synthesized self-introduction (pre-generated after warmup, cached in
  `data/cache/previews/`). Tabs wrap into rows when they run out of width.
- **Source Voice** button plays the actual reference clip (hidden only for
  style-only imports).
- **Regenerate / Stop** — normalizes the script, then STREAMS the synthesis;
  the button turns into Stop while generating. If the voice was never cloned
  it auto-clones first (<1 s) and tells you in the toast.
- **Download** — appears when generation finishes (also after Stop); builds
  the WAV client-side from the buffered PCM.
- **Clone Voice** — ECAPA smart-init; toast shows time, steps, estimated
  similarity and the blend.
- **Sliders** — Quality 4–64 = denoising steps (`total_steps`, default 16),
  Speed 0.5–2.0 (default 1.05), Silence 0–1 s,
  **ECAPA Encoder Steps 1–32** (the fourth bar, default 24).
- **Languages** — **Auto** (default; detects per paragraph: Unicode script
  ranges for ko/ja/ar/ru/el, stopword voting for de/en/es/fr/it/pt/nl/tr,
  engine-auto fallback) plus the design's 12.
- Engine/device status in the tagline row (e.g. `DirectML AMD Radeon 860M`).

## Text normalization

There are **two tiers**, and `app.py` imports the **quality** one:

- **`normalizer.py` (fast)** — the LoudFlow regex engine (deliberately NOT
  mT5 — see its docstring), tuned for DE + EN, microseconds per sentence,
  memoized. On a Spanish or Russian script it still reads numbers with
  English words, because its number core only speaks de/en.
- **`normalizer_quality.py` (quality, what actually runs)** — the same rules
  first, then a deterministic `num2words` pass that renders every remaining
  numeric and symbolic shape in the **actual language of the text**, across
  nine languages: ordinals, decades, roman numerals, ranges, scores,
  durations, versions, math signs, fractions, phone groups and units. Still
  not a language model — a 0.5B polish tier was tested and rejected
  (byte-identical output 36/36 for seconds of added latency).

The DE + EN table below describes the fast tier's core, which the quality
tier inherits:

| Input | Spoken (de) |
|---|---|
| `1920` | neunzehnhundertzwanzig (year form; en: "nineteen twenty") |
| `1914-1918` | neunzehnhundertvierzehn bis neunzehnhundertachtzehn |
| `die 1920er / in den 1990ern / the 1990s` | …zwanziger / …neunzigern / nineteen nineties |
| `am 15.06.1920` | am fünfzehnten Juni neunzehnhundertzwanzig (ordinals, dative after am/vom/zum/bis) |
| `um 14.30 Uhr` / `14:30` | vierzehn Uhr dreißig (Swiss dot-times are not decimals) |
| `CHF 12.50` / `Fr. 12.50` | zwölf Franken fünfzig (not "Komma fünf null") |
| `50%` | fünfzig Prozent |
| `2.4 GHz`, `500 m`, `3.5 kg` | units spelled out |
| `d.h., ggf., St. Gallen / St. Louis` | abbreviations per language |

Everything is pinned by `tests/test_normalizer.py` (17 tests) and
`tests/test_detect.py` (language auto-detection) — plain
`python tests/test_normalizer.py`, no pytest needed. The normalizer never
raises: any error returns the original text.

## API

| Endpoint | What |
|---|---|
| `GET  /api/status` | engine + ECAPA state, device, languages, default steps |
| `GET  /api/voices` | registry |
| `POST /api/voices/upload` | multipart `file`, `vtype=uploaded\|recorded`, optional `name` |
| `POST /api/voices/import_style` | import a ready Supertonic style JSON (+ optional reference audio) as a voice tab |
| `PATCH/DELETE /api/voices/{id}` | rename / remove (files cleaned up) |
| `GET  /api/voices/{id}/audio` | reference clip |
| `GET  /api/voices/{id}/style` | export the style JSON (cloned, imported AND built-in voices) |
| `POST /api/voices/{id}/clone` | `ecapa_steps` → smart-init, saves Supertonic-compatible style JSON |
| `POST /api/synthesize` | `voice_id,text,lang('auto' or code),steps,speed,silence,ecapa_steps` → **streaming** WAV; headers `X-SYNTH-ID/X-SR/X-CHUNKS/X-LANG/X-NORMALIZED/X-AUTO-CLONED/…` |
| `GET  /api/synthesize/stats/{sid}` | live while streaming (`chunks_done`, `infer_s`, `audio_s`, `ttfa_s`) and final (`rtf`, `error`) |
| `POST /api/benchmark/ecapa_steps` | steps sweep with verified similarity per value |

Cloned styles are standard Supertonic voice-style JSONs
(`data/voices/style_*.json`) — they load in LoudFlow too, so a side-by-side
listening comparison against a `paid_parity` clone of the same reference is
just a matter of pointing both UIs at the same file pair.

## Files

```
app.py              FastAPI server: streaming synthesis, auto-language, registry routes
engine.py           Supertonic 3 wrapper (from LoudFlow, + silence_duration)
ecapa.py            ECAPA encoder, smart-init clone, verification, style JSON
normalizer.py       DE/EN TTS normalizer, fast tier (years, dates, money, %, units, …)
normalizer_quality.py  quality tier — num2words across 9 languages (what app.py imports)
prosody.py          pauses, breath debt, edge trim, room tone, rate scale
mastering.py        scipy rebuild of the LoudFlow Pedalboard chain
voiceforge.py       style-JSON blending / import helpers
voices.py           persistent voice registry + built-in voice seeding
benchmark_steps.py  ECAPA-steps convergence benchmark (offline)
tests/              test_normalizer.py, test_detect.py (plain python, no pytest)
static/             index.html + app.js (pixel-perfect UI) + fonts/ (local)
data/refs           uploaded/recorded reference clips
data/voices         registry.json + cloned style JSONs
data/cache          voice_embeddings.npz, ECAPA model copy, benchmark results
```
