# ECAPAFlow — Full Technical Specification of Frontend/Backend Interaction

**Purpose**: this is the final reference before building **QwenFlow**, a
sibling project reusing ECAPAFlow's UI layer against a different backend
engine. Assume zero access to the ECAPAFlow codebase — every behavior
needed to reconstruct an identical UI and interaction model is written out
explicitly below, including exact selectors, payload shapes, formulas, and
failure modes.

> **Note on source documents**: this spec consolidates and supersedes
> `SIBLING_PROJECT_BOOTSTRAP.md`. A file named `STREAMING_DEEPDIVE.md` was
> requested as a second source but does not exist anywhere in this project
> (checked via file search) — §3 below is instead built fresh from a
> line-by-line read of `app.py`'s `_stream()` generator and `app.js`'s
> `regenerate()`/`createStreamPlayer()`, so it should be treated as the
> authoritative streaming reference going forward.

Two engines are wrapped by this app, and this document treats them as two
independent seams:
- **TTS engine** (`engine.py`, `SupertonicEngine`) — text → speech audio.
- **Speaker encoder** (`ecapa.py`, ECAPA-TDNN) — reference audio → a
  192-dim embedding used to pick/blend a voice.

Nothing about the UI, the HTTP contract, or the state-management model
below is specific to either engine's internals — both are swappable behind
the function boundaries described in the original bootstrap report, which
is not repeated here.

---

## 1. Complete UI Inventory

The page is a single static HTML file (`static/index.html`) with **no
framework** — plain inline CSS and a single `<script src="/static/app.js">`.
All dynamic behavior is vanilla JS: one global `state` object plus
hand-written `render*()` functions that rebuild DOM subtrees with
`innerHTML`/`createElement`. There is no virtual DOM and no build step.

Below, every interactive element is documented with: selector, default
state on load, every state it can be in and the exact visual signal for
each, the backend call it makes (if any) with full request/response
shapes, and precisely what happens on success / failure / timeout.

### 1.1 Voice tab pill (one per voice, dynamic — container `#voice-tabs`)

- **Selector**: dynamically created `<div>` per voice inside `#voice-tabs`
  (no per-voice fixed ID — identify by iteration order / `voice.id` in JS
  memory).
- **Default on load**: empty container until `GET /api/voices` resolves;
  `#empty-state` is shown (`display:block`) and `#main-panel` hidden
  (`display:none`) if the resulting list is empty.
- **States**:
  - *Active* tab: border `rgba(255,255,255,0.85)`, background
    `rgba(255,255,255,0.07)`, text `#f8f8f4`.
  - *Inactive* tab: border `rgba(255,255,255,0.14)`, background `#0e0e0e`,
    text `#c8c8c2`.
  - *Cloned* (`voice.cloned === true`): a checkmark SVG is appended after
    the preview button; absent otherwise.
  - *Has verified similarity* (`clone_info.verified_sim_pct != null`):
    pill gets a `title` tooltip `verified ECAPA similarity {pct}%`.
- **Backend**: none directly on the pill itself — clicking the pill body
  calls `selectVoice(id)`, a pure client-side state switch (sets
  `activeVoiceId`, stops any playing audio, clears the result player, and
  re-renders title/source-button/clone-button — no network call).
- **On success/failure/timeout**: N/A (no network call from the pill body
  click itself).

#### 1.1.a Preview play/pause button (nested inside each pill)
- **Selector**: a `<span>` with no fixed ID, identified in JS by
  `state.previewPlayingId`/`state.previewLoadingId` matching the voice ID.
- **Default**: idle — play triangle icon, 8px.
- **States**: idle (▶) → loading (▶ + CSS `animation: stbPulse 1s
  ease-in-out infinite`, triggered the instant the click handler fires,
  before any network response) → playing (⏸, pulse removed) → back to
  idle on `ended`/`error`.
- **Backend**: `GET /api/voices/{voice_id}/preview` — fetched via a native
  `<audio src="...">` element, **not** `fetch()`.
  - Request: no body, no query params.
  - Success: `200`, `audio/wav` bytes (a synthesized self-introduction,
    cached server-side; falls back to the raw reference clip if the voice
    has no style yet).
  - Failure: server can return `503` (engine still loading) or `404` (no
    style yet) — **but the frontend cannot read either status or body**,
    because native `<audio>` elements only expose a generic `error` event,
    not the response JSON. The UI reacts identically to any failure.
- **On success**: icon flips to pause; on natural end, flips back to idle.
- **On failure**: toast `"Preview not ready — the engine may still be
  loading."`; state resets to idle (`previewLoadingId`/`previewPlayingId`
  both cleared).
- **On timeout**: **not implemented**. A native `<audio>` element has no
  timeout of its own; if the server never responds (hung request), the
  button stays stuck in the pulsing "loading" state indefinitely with no
  escape hatch — the only recovery is clicking the same button again
  (which calls `stopAllAudio()` first, discarding the stuck `<audio>`
  instance) or reloading the page.

#### 1.1.b Language-tag `<select>` (nested inside each pill)
- **Default**: value `""` ("Any"), dim text `#5c5c56`.
- **States**: untagged (dim `#5c5c56`) vs. tagged (bright `#e8e8e2`,
  showing the language name). This tag is **the app's own organizational
  label**, not a claim about the underlying model — it only drives "Auto
  Voice" mode routing.
- **Backend**: `PATCH /api/voices/{voice_id}` (multipart form).
  - Request body: `lang` = language code string, or the literal string
    `"any"` to clear the tag.
  - Success `200`: `{"voice": {...full updated registry entry...}}`.
  - Failure `400`: `{"detail": "nothing to update"}` (unreachable from
    this control — `lang` is always sent); `404`:
    `{"detail": "voice not found"}`.
- **On success**: the change was already applied *optimistically* before
  the request even resolves (the `<select>`'s value updates instantly on
  `change`); the server response is not re-applied on top.
- **On failure**: the optimistic change is **rolled back** (`v.lang =
  prev`), the pill re-renders, and a toast shows `"Could not set the voice
  language tag."`.
- **On timeout**: none implemented — the `await fetch(...)` has no client
  timeout; a hung request leaves the (already-applied) optimistic value
  showing indefinitely with no rollback until the promise eventually
  settles one way or the other.

#### 1.1.c Close (×) button (nested inside each pill)
- **Backend**: `DELETE /api/voices/{voice_id}`.
  - Success `200`: `{"ok": true}`. Failure `404`:
    `{"detail": "voice not found"}`.
- **On success or failure — indistinguishable**: the call is wrapped in a
  bare `try { await fetch(...) } catch {}` with **no `.ok` check at all**.
  The voice is removed from the local `state.voices` array unconditionally
  regardless of what the server actually did. A failed delete (e.g. the
  voice was already gone) has zero visible difference from a successful
  one.
- **On timeout**: none — same bare try/catch swallows a hang exactly like
  any other outcome once (if ever) it resolves; the local removal is
  effectively immediate and does not wait in any meaningful way for the
  network call besides the awaited promise.

### 1.2 "+ Upload Voice" pill + hidden file input
- **Selector**: pill has no fixed ID; hidden input is `#file-input`
  (`<input type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.json">`).
- **Default**: empty, no file chosen.
- **States**: none beyond the file-picker's native OS dialog.
- **Backend** — branches on file extension:
  - `*.json` → `POST /api/voices/import_style` (multipart form: `style`
    = the file; optional `ref_audio`, `name`).
    - Success `200`: `{"voice": {...}}`.
    - Failure `400`: `{"detail": "not a valid JSON file"}` or
      `{"detail": "not a Supertonic voice-style JSON (missing
      style_ttl.data/dims)"}` (or `style_dp.data/dims`).
  - anything else → `POST /api/voices/upload` (multipart form: `file`,
    `vtype="uploaded"`).
    - Success `200`: `{"voice": {...}, "warning": "<string, may be
      empty>"}`.
    - Failure `400`: `{"detail": "empty upload"}`, or one of the
      reference-quality rejections in §4.
- **On success**: `refreshVoices()` re-syncs the full list from the
  server, then `selectVoice(newVoiceId)` makes the new tab active
  immediately. If a non-fatal `warning` string came back (e.g. "quiet
  reference"), a toast shows `"Voice added — note: {warning}"` (4500ms).
  A `.json` import shows `"1:1 voice style imported — this tab now speaks
  with exactly this trained voice."` (4200ms).
- **On failure**: toast with the server's `detail` string (4500ms); the
  file input is cleared (`e.target.value=''`) either way so the same file
  can be re-picked; no voice is added; no reload needed.
- **On timeout**: none — indefinite await.

### 1.3 Record Voice pill (toggles to a "Recording…" pill)
- **Default**: mic icon, label "Record Voice".
- **States**: idle → recording (solid border `rgba(255,255,255,0.45)`,
  text `#f0f0ec`, a pulsing white dot, label `Recording 0:0X — tap to
  stop`, live-updating every second, **hard-stops automatically at 45
  seconds**) → back to idle once the upload completes.
- **Backend chain**: `navigator.mediaDevices.getUserMedia({audio:true})`
  (browser API, not the ECAPAFlow server) records via `MediaRecorder`;
  on stop, the recorded Blob (WebM/Opus in Chrome) is decoded and
  re-encoded to mono 16-bit WAV client-side (`blobToWav()`, using
  `AudioContext.decodeAudioData`), then POSTed to `POST
  /api/voices/upload` with `vtype="recorded"` — same endpoint/shapes as
  §1.2's non-JSON branch.
- **On success**: same as §1.2 (auto-selects the new tab); recorded-voice
  display names auto-number as "Recorded Voice", "Recorded Voice (2)", etc.
- **On failure**:
  - Mic permission denied → toast `"Microphone access denied — allow it
    to record a reference."` (no server call ever made).
  - Recording produced 0 bytes → toast `"Recording was empty."`.
  - Upload rejected by the server's audio-quality gate → toast with the
    server's `detail` string.
- **On timeout**: none.

### 1.4 Voice title (display + inline edit)
- **Selector**: container `#title-block`.
- **Default**: view mode — 30px bold text = active voice's name, a pencil
  icon button beside it.
- **States**: view ↔ edit. Clicking the pencil swaps the text for an
  auto-focused, auto-selected `<input>` (underlined, no box). `Enter` or
  blur commits; `Escape` cancels without saving.
- **Backend**: `PATCH /api/voices/{voice_id}` (multipart, field `name`) —
  only sent if the trimmed value is non-empty **and** different from the
  current name.
- **On success**: not distinguished from failure (see below) — the local
  name was already updated in `state` before the request was even sent.
- **On failure**: **silently ignored**. The code is `try { await
  fetch(...) } catch {}` with no `.ok` check and the comment "rename
  persists on next sync" — a failed rename shows no error at all; the UI
  keeps displaying the new (unsaved) name until the next
  `refreshVoices()` call quietly reverts it with zero user-facing
  explanation.
- **On timeout**: none.

### 1.5 "Source Voice" / "Preview" button
- **Selector**: `#source-btn` (button), `#source-label` (text),
  `#source-icon` (icon).
- **Default**: hidden (`display:none`) if the active voice has no
  playable audio at all (no reference file and not a built-in voice).
  When visible: label = `"Source Voice {duration}"` for
  uploaded/recorded/imported voices, `"Preview {duration}"` for built-ins.
- **States**: hidden / visible; within visible: idle (play icon) ↔
  playing (pause icon).
- **Backend**: `GET /api/voices/{voice_id}/audio` via native `<audio>`
  (not `fetch()` — same JSON-blindness caveat as §1.1.a).
  - Success: `200`, raw audio bytes (`audio/wav`, `audio/mpeg`,
    `audio/flac`, `audio/ogg`, `audio/mp4`, or `audio/webm` depending on
    the stored file's extension); or, if the voice has no reference file
    at all, the server transparently serves the synthesized self-intro
    preview instead.
  - Failure: `404` `{"detail": "voice not found"}` or `{"detail":
    "reference audio missing"}` — again invisible to the frontend beyond
    a generic `error` event.
- **On success**: icon flips to pause; flips back on natural end.
- **On failure**: toast `"Could not play the reference audio."`.
- **On timeout**: none.

### 1.6 Script textarea
- **Selector**: `#script`.
- **Default value**: the canned "Chatbot" sample paragraph (since the
  default content-type tab is "Chatbot" — see §1.8).
- **States**: plain text input; hard `maxlength="100000"` HTML attribute
  is the only client-side limit (mirrors the server's `MAX_TEXT_CHARS =
  100000`). No visual "error" state exists for exceeding it — the browser
  simply refuses further keystrokes past the limit.
- **Backend**: none directly — its value becomes the `text` field of the
  next `POST /api/synthesize` call only.

### 1.7 Character counter
- **Selector**: `#char-count` (pure text, no interactivity) inside a
  `{count}/100000` display, updated on every `input` event on `#script`.

### 1.8 Content-type tabs (Freeform / Announcement / Chatbot / Article / Podcast)
- **Selector**: five plain `<div>`s inside `#content-types`, no fixed IDs.
- **Default active**: "Chatbot".
- **States**: active — bold (600), underlined, `#f8f8f4`; inactive —
  normal weight, no underline, `#84847c`.
- **Backend**: **none**. Clicking a tab replaces the textarea's content
  with a hardcoded client-side sample string for that type (`Freeform`'s
  sample is the empty string, i.e. it clears the textarea). This
  **unconditionally discards** whatever the user had already typed — no
  confirmation prompt.

### 1.9 Language toggle + dropdown
- **Selector**: `#lang-toggle` (trigger row), `#lang-name` (current
  label), `#lang-menu` (floating panel).
- **Default**: "Auto" selected, menu closed.
- **States**: menu closed / open (toggles on click; also auto-closes on
  any click outside the menu, via a document-level listener).
- **Backend**: none on selection itself. However, the **list of options**
  is backend-influenced: the client seeds a 12-language + "Auto" array
  hardcoded in JS, then **unconditionally overwrites it** with whatever
  `GET /api/status`'s `languages` array contains the first time that
  request resolves (in the shipped app these match exactly, but a
  sibling project's backend controls this list). The selected language
  name is converted to a code (`langCode()`) and sent as the `lang` field
  only when Regenerate is clicked.

### 1.10 Mode buttons — Speed / Medium / Quality
- **Selector**: three `<button>`s inside `#mode-buttons`, no fixed IDs.
- **Default active**: whichever preset's `steps` value equals
  `state.quality` (default `16` → "Medium" active out of the box, unless
  a `localStorage` preference overrides it — see §2).
- **States**: active (bright border `rgba(255,255,255,0.85)`, background
  tint) vs. inactive (`rgba(255,255,255,0.16)` border, `#0e0e0e`
  background). Subtitle text under the label starts as just `"{steps}
  steps"` and gains a `" · ×{rtf}"` suffix once real timing history
  exists for that step count (see §5 for exactly how/when that number
  appears — it is not present on a cold server with no prior syntheses).
- **Backend**: none on click itself (sets `state.quality`, which then
  moves the Quality slider — see §1.12 — and calls `savePrefs()`). The
  **contents** of the 3-item array (labels/step counts) are seeded
  client-side then overwritten by `GET /api/status`'s `modes` field on
  first load.
- **On failure of the underlying `/api/status` call**: silent degrade —
  the subtitle simply never gains its `×RTF` suffix; no error is shown.

### 1.11 Auto Voice toggle
- **Selector**: `#auto-voice-toggle`, `#auto-voice-dot`,
  `#auto-voice-label`.
- **Default**: OFF — dim dot `rgba(255,255,255,0.25)`, label "Auto Voice
  OFF" (overridable by a saved `localStorage` preference).
- **States**: ON (bright dot `#f6f6f2`, "Auto Voice ON") / OFF.
- **Backend**: none on click — only changes the `auto_voice` field
  (`"1"`/`"0"`) of the *next* `POST /api/synthesize` call.

### 1.12 Streaming toggle
- **Selector**: `#stream-toggle`, `#stream-dot`, `#stream-label`.
- **Default**: ON — bright dot, "Streaming ON" (overridable by saved
  preference).
- **States/backend**: identical pattern to §1.11 — only affects the
  `streaming` field of the next synthesize call.

### 1.13 Custom sliders — Quality, Speed, Silence Duration, ECAPA Encoder Steps
- **Selectors**: none fixed — built dynamically into `#sliders` (Quality,
  Speed, Silence) and `#clone-sliders` (ECAPA Encoder Steps); each has an
  internal `{fill, knob, value}` DOM handle kept only in a JS array, not a
  queryable ID.
- **Defaults**: Quality `16` (or `localStorage` override), Speed `1.05`,
  Silence `0.30`, ECAPA Encoder Steps `24` — this last one is **replaced
  once** by the server's `default_ecapa_steps` value the first time
  `GET /api/status` resolves, which literally rebuilds all four slider DOM
  nodes (`buildSliders()` runs again).
- **Ranges**: Quality `[4, 64]` step `1`; Speed `[0.5, 2.0]` step `0.05`;
  Silence `[0, 1.0]` step `0.05`; ECAPA Encoder Steps `[1,
  maxEcapaSteps]` step `1` (`maxEcapaSteps` default `32`, also
  server-supplied).
- **States**: value only — sliders have **no disabled/error state at
  all**. They remain interactive even mid-generation or mid-clone;
  changing a value while a request is in flight never affects that
  in-flight request, only the next one.
- **Interaction model**: pointer-drag on a 150px track
  (`pointerdown`/`pointermove`/`pointerup`), not a native `<input
  type=range>`.
- **Backend**: none directly — values are only read into the next
  Regenerate/Clone request body.

### 1.14 Clone status text
- **Selector**: `#clone-status` — pure display, no interactivity. Text is
  computed from the active voice's `type`/`clone_info` (see §5's metric
  table for the exact string components: timing, est./verified
  similarity %, top-3 blend voice codes).

### 1.15 "Clone Voice" button
- **Selector**: `#clone-btn`.
- **Default label**: `"Clone Voice"` for a never-cloned uploaded/recorded
  voice with nothing in progress.
- **States** (checked in this priority order client-side):
  1. `state.isCloning` → label `"Cloning…"`, `opacity:0.7`.
  2. active voice `type === 'builtin'` → label `"Native Voice ✓"`,
     `opacity:1` — clicking is a **client-side no-op**: a toast
     (`"This is a native Supertonic voice — nothing to clone."`) fires
     and the request is never sent.
  3. active voice `type === 'imported'` → label `"Imported 1:1 ✓"` — same
     client-side no-op pattern (`"This is an imported 1:1 trained
     voice — nothing to re-clone."`).
  4. `voice.cloned === true` (already cloned) → label `"Voice Cloned ✓"`.
  5. else → `"Clone Voice"`.
- **Backend**: `POST /api/voices/{voice_id}/clone` (multipart form, field
  `ecapa_steps` = the current ECAPA slider value).
  - Success `200`: `{"voice": {...updated registry entry...}, "clone":
    {"est_sim_pct": <float 0-100>, "ranking": [{"voice": <str>, "sim":
    <float>, "weight": <float|null>}, ... all 10 built-ins, sorted desc],
    "effective_steps": <int>, "timings_ms": {"embed": <float>, "rank":
    <float>, "blend": <float>, "total": <float>}}}`.
    - **Shape nuance**: this immediate response's `clone.ranking` has
      **all 10** entries; what gets *persisted* into the voice registry
      (and is what a later `GET /api/voices` returns as
      `voice.clone_info.ranking`) is **truncated to the top 5**.
  - Failure `503`: `{"detail": "Supertonic 3 engine still loading"}`
    (also pre-empted client-side — see §4).
  - Failure `404`: `{"detail": "voice not found"}`.
  - Failure `400`: `{"detail": "built-in voice — nothing to clone"}` or
    `{"detail": "this voice has no reference audio to clone from
    (imported 1:1 style)"}` — both are defense-in-depth; the shipped UI
    pre-empts them client-side (states 2/3 above), so these are normally
    only reachable via a direct API call or a race where the voice's type
    changed between render and click.
  - Failure `404`: `{"detail": "reference audio missing: {filename}"}`
    (the reference file was deleted from disk after upload).
  - Failure `500`: `{"detail": "clone failed: {ExceptionType}:
    {message}"}` — the one true catch-all.
- **On success**: `refreshVoices()` re-syncs the whole list; a toast shows
  `"Voice cloned locally in {secs}s — {steps} ECAPA steps, est.
  similarity {pct}% (blend {top3}). Nothing uploaded or purchased."`
  (4200ms). A second refresh is scheduled 6 seconds later
  (`scheduleVerifiedRefresh()`) to pick up the server's **asynchronously
  computed** `verified_sim_pct` once its background verification thread
  finishes.
- **On failure**: toast with the exact server `detail` string;
  `state.isCloning` is reset in a `finally` block; button returns to its
  pre-click label. Fully recoverable, no reload needed.
- **On timeout**: **none implemented**. Since cloning is normally
  sub-2-seconds (per the project's own benchmarks), a genuine hang here
  leaves the button stuck on `"Cloning…"` **indefinitely with no Cancel
  control** — unlike Regenerate, there is no Stop-equivalent for cloning.

### 1.16 Regenerate / Stop button (+ spinner)
- **Selector**: `#regen-btn`, `#regen-spinner` (SVG, hidden by default),
  `#regen-label` (default text "Regenerate").
- **States**:
  1. Idle: label "Regenerate", spinner hidden.
  2. Cloning phase: label "Cloning voice…", spinner visible — shown only
     when the client *predicts* (mirroring the server's own check: voice
     is not built-in and not yet cloned) that an auto-clone is about to
     run before any audio can start.
  3. Generating phase: label "Stop", spinner visible.
- **The same DOM button is overloaded** — its click handler branches on
  `state.isRegenerating`: if true, it calls `stopGeneration()` (aborts the
  in-flight request) instead of starting a new one. There is no separate
  Stop element.
- **Backend**: `POST /api/synthesize` — full contract in §3.
- **On success**: playback proceeds per §3; RTF/duration/download button
  update as the stream progresses and completes.
- **On failure**: toast with the error message (server `detail`, or the
  raw JS error message for a network-level failure); state fully reset,
  recoverable without reload.
- **On abort** (Stop clicked): resolved **silently** — no toast at all;
  this is the one outcome explicitly treated as expected user behavior,
  not an error (`if (!(err && err.name === 'AbortError')) showToast(...)`).
- **On timeout**: **none implemented anywhere** — no client-side
  `AbortController` timeout, no server-side request timeout. A request
  will wait indefinitely for a response unless the user manually clicks
  Stop.

### 1.17 Result play/pause button
- **Selector**: `#result-play-btn`, `#result-icon`.
- **Default behavior before any generation**: clicking it shows toast
  `"Hit Regenerate first to synthesize this script."` and does nothing
  else (no `player` object exists yet).
- **States**: play icon (paused/not started) ↔ pause icon (playing).
- **Backend**: none — purely controls the already-buffered client-side
  Web Audio PCM player from the last synthesize stream.

### 1.18 Result progress bar
- **Selectors**: `#result-buffered` (dim fill, width %), `#result-progress`
  (bright fill, width %) — both absolutely positioned inside a shared
  track.
- **Pure display**, refreshed every 150ms by a `setInterval` ticker while
  a player exists. **Not seekable** — no click/drag handler is attached to
  the track at all.

### 1.19 Result current-time / duration labels
- **Selectors**: `#result-current`, `#result-duration` — pure display,
  `mm:ss` format. Duration is prefixed with `~` while the exact length is
  still unknown (streaming in progress), and snaps to an exact value with
  no prefix once the stream finishes.

### 1.20 Result RTF label
- **Selector**: `#result-rtf` — pure display; `display:none` when no RTF
  value is available. Shows a live in-progress reading (`"RTF 0.31 ·
  4/9"` — chunks done/total) while generating, then the final measured
  RTF once done. Formulas in §5.

### 1.21 Download button
- **Selector**: `#download-btn` — hidden until the player has a fully
  finished stream with `>0` seconds of audio.
- **Click**: assembles the buffered PCM samples into a WAV `Blob`
  entirely client-side (`player.toWavBlob()`) and triggers a browser file
  download named `{voiceName}.wav`. **No backend call** — 100%
  client-side, works even after a Stop-triggered partial result.

### 1.22 Toast
- **Selector**: `#toast` — generic message surface, never directly
  user-triggered. Auto-hides after a per-message duration (default
  3200ms; a few specific messages extend this to 4200–5200ms for longer
  text). Only one toast is visible at a time — a new call replaces the
  current one and resets its hide timer; there is no stacking/queue.

### 1.23 Engine status text
- **Selector**: `#engine-status` — pure display, top-right of the
  "Supertonic 3 | Lightning Fast…" line.
- **States**: while `!engine.loaded` — shows the server's live status
  message (e.g. `"Loading Supertonic 3 model (first run downloads
  ~400MB)..."`, `"Warming up..."`) with a pulsing CSS animation
  (`stbPulse 1.6s ease-in-out infinite`); once loaded — shows the static
  device label (e.g. `"DirectML AMD Radeon(TM) 860M Graphics"`), animation
  removed.
- **Backend**: `GET /api/status`, polled by `pollStatus()` every 1500ms
  **only while not yet loaded** (self-terminating retry loop — see §2);
  once loaded, polling for this purpose stops (though `/api/status` is
  still called elsewhere, e.g. `refreshRtfHistory()` after each
  synthesis).

---

## 2. State Management

### 2.1 Client-side persisted state
**Exactly one** `localStorage` key is used: `ecapaflow_prefs`, storing
`{quality, streaming, autoVoice}` only (written by `savePrefs()` any time
one of those three changes). Everything else lives in a single in-memory
JS object (`state`) that is **fully reset on page reload**: active voice
selection, script text, content-type tab, language, speed, silence,
ECAPA-steps slider value (until the server's default overwrites it once),
playback state, live stats, RTF, etc. The list of voices itself is never
cached client-side across reloads — it is always re-fetched fresh from
`GET /api/voices` on init.

### 2.2 Concurrency / race handling (client-side)
- **`regenerate()`** guards itself: `if (state.isRegenerating) return;` —
  a second click while a request is in flight is a no-op, not a queue.
- **`cloneVoice()`** guards itself identically via `state.isCloning`.
- **Clicking Stop** while generating does not start a second request — it
  calls `stopGeneration()`, which aborts the in-flight fetch
  (`AbortController.abort()`) and finalizes the player as if the stream
  had ended normally.
- **Switching voice tabs mid-generation** (`selectVoice()`) calls
  `stopAllAudio()` + `clearResult()`. `clearResult()` *does* abort the
  in-flight `synthAbort` controller and destroy the player, but does
  **not** synchronously reset `state.isRegenerating`/`regenPhase` — those
  are only cleared inside `regenerate()`'s own `finally` block once the
  aborted fetch's promise actually settles. This is a genuine (harmless,
  cosmetic-only) race window: the Regenerate button can briefly still
  read "Stop" for a moment after the user has already switched tabs and
  triggered the abort.
- **`AbortController` is used for exactly one call**: `POST
  /api/synthesize`. No other fetch call in the app (upload, clone, PATCH,
  DELETE, status polling) is cancellable client-side once sent.
- **`pollStatus()`** is the one function with its own automatic retry
  loop: on any fetch failure, or while `s.loaded` is still false, it
  reschedules itself via `setTimeout(pollStatus, 1500)` indefinitely.
  `refreshVoices()` on failure is silently swallowed with no retry
  (comment: "backend not up yet").

### 2.3 Server-side state — genuinely global, not per-session
There is **no session, cookie, or auth concept anywhere** in the backend.
Every piece of server-side mutable state is a **module-level global**
shared by all connected clients on that one running process:

| State | Location | Persistence |
|---|---|---|
| `engine` (`SupertonicEngine` singleton) — model handle, voice-style cache, provider info, last-20 error log, load/loading flags | `engine.py` module scope | in-memory only, rebuilt on process restart |
| `ecapa_mod._state` / `_cache` — encoder handle, built-in voice embedding table | `ecapa.py` module scope | embedding table also mirrored to `data/cache/voice_embeddings.npz` |
| `_rtf_history` (`RtfHistory` instance) — per-steps timing model | `app.py` module scope | persisted to `data/cache/rtf_history.json`, read/updated on every chunk |
| `_SYNTH_STATS` (`OrderedDict`, capped at 32 entries) — per-synthesis live/final stats | `app.py` module scope | **in-memory only**, evicted oldest-first past 32 entries, lost on restart |
| voice registry | `voices.py` module scope (`_registry`) | persisted to `data/voices/registry.json`, cached in memory after first load |

**Concurrency implications**:
- `SupertonicEngine.synthesize()` takes a single `threading.Lock`
  (`self._lock`) around the actual inference call. Two simultaneous
  synthesize requests (different browser tabs, or genuinely different
  users) **serialize at the engine** — the second request's audio chunks
  simply arrive later, with **no queued-position feedback anywhere in the
  UI**; the user has no way to know they're waiting behind someone else's
  request.
- The voice registry uses a `threading.RLock` around all reads/writes,
  guaranteeing internal consistency, but there is **no optimistic-locking
  or conflict detection** — two clients renaming/tagging the same voice
  "simultaneously" resolve last-write-wins with no warning to either.
- `ecapa.py`'s **encoder load** is lock-guarded (one-time), but the actual
  embedding **inference calls** (`embed_reference`, `_embed_wav_16k_np`)
  are **not** wrapped in any lock — concurrent calls rely entirely on
  whatever thread-safety PyTorch happens to provide for concurrent forward
  passes on one shared model instance; this is not explicitly
  synchronized or ordered by this codebase.
- `_SYNTH_STATS` entries have no ownership/auth binding — any client that
  knows or guesses a 12-hex-character synthesis ID could poll its stats.

**Bottom line for QwenFlow**: this architecture assumes a single local
user (a personal-tool design, not multi-tenant). If QwenFlow needs
multi-user isolation, every one of the tables above needs session-scoping
added — none of it exists today.

---

## 3. Streaming — Full Lifecycle

This section traces one Regenerate click end-to-end, byte by byte.

### 3.1 Sequence of events

1. **Click** `#regen-btn` → `regenerate()` runs guard checks: is a
   generation already in flight? is there active-voice + non-empty text?
   is the engine loaded? Any failing check shows a toast and returns
   without any network call.
2. `stopAllAudio()` + `clearResult()` tear down any prior player, timers,
   and abort controller.
3. `state.isRegenerating = true`; `state.regenPhase` is predicted
   client-side (`'cloning'` if the voice is non-builtin and not yet
   cloned, else `'generating'`) purely so the button label can say
   "Cloning voice…" instead of just spinning on "Stop" during the phase
   where the server is synchronously auto-cloning before any bytes flow.
4. A new `AbortController` is created and stored in `synthAbort`.
5. **Request sent**: `POST /api/synthesize`, `multipart/form-data`:

   | field | value |
   |---|---|
   | `voice_id` | active voice's ID |
   | `text` | full script textarea contents |
   | `lang` | code resolved from the language dropdown (e.g. `"en"`, `"auto"`) |
   | `steps` | Quality slider value (int) |
   | `speed` | Speed slider value (float) |
   | `silence` | Silence Duration slider value (float) |
   | `ecapa_steps` | ECAPA Encoder Steps slider value (int) |
   | `streaming` | `"1"` or `"0"` |
   | `auto_voice` | `"1"` or `"0"` |

6. **Server-side, before any response bytes are sent** (`app.py`,
   `api_synthesize`):
   - Validates: engine loaded (else `503`), text non-empty (else `400`),
     voice exists (else `404`), coerces an invalid `lang` to `"na"`.
   - **Auto-clone check**: if the voice is not built-in and has never
     been cloned (or its style file is missing from disk), runs the full
     clone pipeline **synchronously, blocking the response** (this is the
     entire "Cloning voice…" phase — no partial signal reaches the client
     during this blocking window besides its own prediction from step 3).
   - Resolves the voice's TTS style object.
   - Splits the script into paragraphs (on line breaks), then into
     sentences per paragraph; runs the text normalizer per paragraph
     (numbers/dates/currency/markdown-stripping); if `lang == "auto"`,
     detects each paragraph's language independently via a stopword
     voting heuristic.
   - If Auto Voice is on, resolves a per-language voice override for each
     distinct language actually present in the script (falling back to
     the selected voice wherever no tagged voice exists for that
     language).
   - Computes a **predicted total duration** (`est_audio_s`) and a
     **predicted stall warning** (`will_likely_stall`) from the
     persistent RTF/CPS history model (see §5 for the exact formulas).
   - Registers an initial entry in `_SYNTH_STATS[sid]` (`done:false`,
     zeroed counters).
   - Returns a `StreamingResponse` (media type `audio/wav`) with these
     response **headers**, all set before the body starts streaming:

     | Header | Meaning |
     |---|---|
     | `X-SYNTH-ID` | 12-hex-char synthesis ID, used to poll `/api/synthesize/stats/{sid}` |
     | `X-SR` | sample rate (int, e.g. `44100`) |
     | `X-CHUNKS` | total sentence count planned |
     | `X-STEPS` | diffusion steps used |
     | `X-LANG` | resolved language label (e.g. `"auto:en,de"`) |
     | `X-DEVICE` | `"CUDA"` / `"DirectML"` / `"CPU"` |
     | `X-AUTO-CLONED` | `"1"` if step 6's blocking auto-clone ran, else `"0"` |
     | `X-CLONE-MS` | milliseconds spent auto-cloning (0 if none) |
     | `X-EST-SIM` | the voice's `est_sim_pct` if it has one, else empty |
     | `X-EST-AUDIO-S` | predicted total audio duration in seconds |
     | `X-WILL-STALL` | `"1"` if the model predicts generation will fall behind realtime for this script length at this steps value |
     | `X-NORMALIZED` | URL-encoded, 900-char-truncated normalized text (sent but **never read by the shipped frontend** — dead weight today, available for a sibling project to use) |

7. **Client receives headers** — this moment is used as a time-to-first-
   byte proxy (`ttfbMs`, minus any auto-clone time, since that's CPU work
   not network latency) to size how much audio to buffer before playback
   starts. `state.estDuration` is set from `X-EST-AUDIO-S`. A PCM stream
   player is created (`createStreamPlayer(sr, ttfbMs)`); a 150ms UI
   progress ticker and a 600ms stats-poll timer both start.
8. **Body streaming begins server-side**:
   1. First, exactly **44 raw bytes**: a hand-built WAV header
      (`RIFF`/`WAVE`/`fmt `/`data` chunks) with size fields **maxed out**
      (`0xFFFFFFFF`) since the total length is unknown ahead of time —
      the frontend strips these 44 bytes unconditionally and never
      inspects them.
   2. Then, for each sentence-chunk in order:
      - A **char budget** is computed for how many upcoming sentences to
        merge into this one engine call: the very first chunk in the
        whole stream is **always exactly one sentence** (fastest possible
        first audio); every subsequent chunk's budget is computed from
        the client's already-buffered playback runway and the persistent
        RTF/CPS model (`_stream_allowance()` — see §5); if streaming is
        OFF, the budget is simply the max chunk cap every time.
      - If this is not the very first chunk of the stream, a **silence
        filler** of raw PCM zero-bytes is yielded first: a short
        "sentence gap" if the next chunk is the same paragraph as the
        previous one, or a longer "paragraph pause" (0.6s) if it starts a
        new paragraph.
      - The engine synthesizes that merged chunk of text (produces a
        waveform + a measured pure-inference time).
      - The persistent RTF/CPS model is updated with this chunk's real
        measurement immediately (sharpens the estimate for the very next
        chunk of *this same* stream, and every future stream at this
        steps value).
      - The waveform is soft-limited (peaks capped at 0.85 of full scale)
        and converted to 16-bit PCM, **mono**, little-endian.
      - `_SYNTH_STATS[sid]` is updated in place with live progress
        (chunks done, cumulative inference seconds, cumulative audio
        seconds, time-to-first-audio once known, which voice(s) have
        actually been used so far).
      - The raw PCM bytes are yielded onto the wire.
   3. This repeats until every sentence has been synthesized, or the
      generator is terminated early (see §3.3).
9. **Client's read loop**: reads the raw byte stream incrementally via
   `res.body.getReader()`. It strips exactly the first 44 bytes once,
   carries over any odd trailing byte across network-chunk boundaries (so
   a 16-bit sample split across two TCP reads is never corrupted),
   converts each usable pair of bytes to a normalized `Float32` sample
   (`int16 / 32768`), and pushes the resulting `Float32Array` into the Web
   Audio PCM player. **If** streaming is ON and the buffered audio has
   reached an adaptively-sized start threshold (0.3–2.0 seconds, scaled
   from the measured `ttfbMs`) **and** playback hasn't started yet, it
   calls `player.start()` right there — this is the exact moment audible
   playback begins, typically well before the rest of the response has
   arrived.
10. **Loop exits** when `reader.read()` reports `done: true` — this
    happens whether the stream ended normally, ended early due to a
    caught server-side exception, or was terminated by a client abort
    (see §3.3 for how these are told apart). The client marks the player
    fully buffered (`p.finishStream()`); if streaming had been off (or
    the whole result was too short to ever cross the start threshold),
    playback starts now for the first time.
11. **Client fetches** `GET /api/synthesize/stats/{synthId}` one final
    time to get the authoritative final RTF and check the `error` field
    (see §3.3, §4). If Auto Voice was on and more than one distinct voice
    actually spoke, an informational toast lists which ones.
12. **`finally` block** (client): stops the stats-poll timer, resets
    `isRegenerating`, re-renders the Regenerate button/player/clone
    button, and triggers `GET /api/status` once more to refresh the mode
    buttons' measured-RTF hints with whatever the server's history model
    now looks like after this run.

### 3.2 Exact wire shape, in order
```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-SYNTH-ID: 4f9a2c1e08b3
X-SR: 44100
X-CHUNKS: 9
X-STEPS: 16
X-LANG: en
X-DEVICE: DirectML
X-AUTO-CLONED: 0
X-CLONE-MS: 0
X-EST-SIM: 51.3
X-EST-AUDIO-S: 8.4
X-WILL-STALL: 0
X-NORMALIZED: <url-encoded text, first 900 chars>
Access-Control-Expose-Headers: X-SYNTH-ID,X-SR,X-CHUNKS,X-STEPS,X-LANG,X-DEVICE,X-AUTO-CLONED,X-CLONE-MS,X-EST-SIM,X-EST-AUDIO-S,X-WILL-STALL,X-NORMALIZED

[44 bytes: fake WAV header, RIFF/WAVE/fmt/data, sizes = 0xFFFFFFFF]
[PCM16 mono bytes: chunk 1 — always exactly 1 sentence]
[optional silence filler PCM bytes: sentence-gap or paragraph-pause]
[PCM16 mono bytes: chunk 2 — adaptively sized]
[optional silence filler PCM bytes]
[PCM16 mono bytes: chunk 3]
... repeats until all sentences are synthesized or the stream is cut short ...
[connection closes — no trailer, no end-of-stream marker of any kind]
```
There is **no explicit end-of-stream sentinel** in the byte stream
itself — completion is signaled purely by the HTTP response body ending
(the TCP connection's normal close / chunked-transfer terminator, handled
transparently by `fetch`).

### 3.3 Distinguishing completion vs. mid-stream error vs. client abort
These three outcomes are **not** distinguishable from the raw byte stream
alone — all three simply end the response body. The frontend must always
cross-check `GET /api/synthesize/stats/{sid}` to know which actually
happened:

| Outcome | What the wire shows | How the frontend tells |
|---|---|---|
| **Normal completion** | Full stream, ends after the last sentence's audio | `reader.read()` returns `done:true`; the final stats poll shows `done:true, error:""` |
| **Mid-stream server error** | Stream simply stops early, with whatever audio was already sent — HTTP status was already committed to `200` long before the failure, so **no error status is ever possible** at this point | `reader.read()` still eventually returns `done:true` (the connection just closes) — the **only** signal is the final stats poll's non-empty `error` field, surfaced via a toast **after** the truncated audio has already played: `"Synthesis ended early: {error}"` |
| **Client abort** (Stop button, or tab close/navigate-away) | Same as above from the wire's perspective | The **browser-side** `fetch`/`reader.read()` promise **rejects** with `AbortError` — this is the one outcome visible synchronously as a JS exception, explicitly filtered out of the toast path (`if (!(err && err.name === 'AbortError')) ...`) |

Server-side, the streaming generator function wraps its per-chunk loop in
`try/except Exception/except GeneratorExit/finally`:
- A regular `Exception` during synthesis is caught, logged, and recorded
  as the `error` string in `_SYNTH_STATS[sid]` — the generator function
  then simply falls through to its `finally` block and ends (no more
  chunks are yielded), which is exactly why this looks identical to a
  normal completion on the wire.
- `GeneratorExit` is explicitly caught and **re-raised** (a no-op that
  exists only because the exception must propagate for Python generator
  semantics to work correctly) — this fires when the ASGI layer detects
  the client is gone and closes the generator at its next suspension
  point.
- The `finally` block **always** runs regardless of which path was taken,
  always writing a final `_SYNTH_STATS[sid]` entry with `done:true` and
  whatever partial numbers had accumulated — even if nobody will ever
  read it again (e.g. after a tab close).

### 3.4 What happens server-side if the tab is closed / navigation happens mid-stream
The in-flight chunk of speech being synthesized **at the exact moment of
disconnect is not preemptible** — Python cannot forcibly interrupt a
running synchronous call, so whatever `engine.synthesize()` call is
currently executing on that background thread **runs to completion**
regardless (wasted GPU/CPU work, not recoverable, not cancellable). Only
at the *next* `yield` statement does the generator actually receive
`GeneratorExit` and stop producing further chunks. There is:
- **No cancellation token** passed into the TTS engine call.
- **No watchdog thread** killing long-running synthesis.
- **No server-side request timeout** of any kind, anywhere in the app.

A very long script requested at Quality mode (32 steps) on a slow device,
left running after the user navigates away, will simply keep consuming
CPU/GPU for however long that specific in-flight chunk takes to finish,
then stop one chunk-boundary later — not instantly.

---

## 4. Error Handling, Exhaustively

Every distinct error condition the backend can produce, its exact HTTP
status + body, what the frontend does with it, and whether the app is
left usable without a reload.

| # | Condition | Endpoint | Status | Body | Frontend behavior | Recoverable without reload? |
|---|---|---|---|---|---|---|
| 1 | Empty file upload | `POST /api/voices/upload` | 400 | `{"detail":"empty upload"}` | toast(detail) | Yes |
| 2 | Reference too short | same | 400 | `{"detail":"reference too short (X.Xs); need at least 1.5s of clean speech"}` | toast(detail) | Yes |
| 3 | Reference too long | same | 400 | `{"detail":"reference too long (X.Xs); cap is 60s (30s recommended)"}` | toast(detail) | Yes |
| 4 | Reference effectively silent (peak) | same | 400 | `{"detail":"reference effectively silent (peak X.XXXX = X dBFS)"}` | toast(detail) | Yes |
| 5 | Reference effectively silent (RMS) | same | 400 | `{"detail":"reference effectively silent (RMS X.XXXXX = X dBFS)"}` | toast(detail) | Yes |
| 6 | Reference quiet/weak/noisy (**warning, not rejection**) | same | 200 | `{"voice": {...}, "warning": "quiet reference (peak X dBFS) \| weak signal (...) \| high spectral flatness (...)"}` | voice IS added; toast `"Voice added — note: {warning}"` | Yes (not even an error) |
| 7 | Import: invalid JSON | `POST /api/voices/import_style` | 400 | `{"detail":"not a valid JSON file"}` | toast(detail) | Yes |
| 8 | Import: missing style keys | same | 400 | `{"detail":"not a Supertonic voice-style JSON (missing style_ttl.data/dims)"}` (or `style_dp...`) | toast(detail) | Yes |
| 9 | Rename/tag with nothing to update | `PATCH /api/voices/{id}` | 400 | `{"detail":"nothing to update"}` | **unreachable from shipped UI** (both callers always send a field) | N/A |
| 10 | Voice not found (PATCH) | same | 404 | `{"detail":"voice not found"}` | for the language-tag select: rollback + toast; for title rename: **silently ignored, no check at all** | Yes |
| 11 | Voice not found (DELETE) | `DELETE /api/voices/{id}` | 404 | `{"detail":"voice not found"}` | **not checked at all** — voice removed from local state regardless | Yes (no visible difference from success) |
| 12 | Voice not found (audio/preview/style/clone) | various `/api/voices/{id}/...` | 404 | `{"detail":"voice not found"}` | varies — audio/preview surface only a generic `<audio>` error event; clone shows toast(detail) | Yes |
| 13 | Reference audio missing on disk | `POST /api/voices/{id}/clone` | 404 | `{"detail":"reference audio missing: {filename}"}` | toast(detail) | Yes |
| 14 | Clone: built-in voice | same | 400 | `{"detail":"built-in voice — nothing to clone"}` | pre-empted client-side; server path is defense-in-depth only | Yes |
| 15 | Clone: imported voice | same | 400 | `{"detail":"this voice has no reference audio to clone from (imported 1:1 style)"}` | pre-empted client-side; defense-in-depth only | Yes |
| 16 | Clone: engine loading | same | 503 | `{"detail":"Supertonic 3 engine still loading"}` | pre-empted client-side (own toast before the request is sent); server path is a race-window fallback | Yes |
| 17 | Clone: unexpected exception | same | 500 | `{"detail":"clone failed: {ExceptionType}: {message}"}` | toast(detail); `isCloning` reset in `finally` | Yes |
| 18 | Synthesize: engine loading | `POST /api/synthesize` | 503 | `{"detail":"Supertonic 3 engine still loading"}` | pre-empted client-side | Yes |
| 19 | Synthesize: empty text | same | 400 | `{"detail":"text required"}` | pre-empted client-side (own toast) | Yes |
| 20 | Synthesize: voice not found | same | 404 | `{"detail":"voice not found"}` | toast(detail) | Yes |
| 21 | Synthesize: cloned style missing on disk | same | 500 | `{"detail":"cloned style missing on disk"}` | toast(detail) | Yes |
| 22 | Synthesize: text empty after normalization (e.g. pure markdown/junk input) | same | 400 | `{"detail":"text is empty after normalization"}` | toast(detail) | Yes |
| 23 | Unknown synthesis ID | `GET /api/synthesize/stats/{sid}` | 404 | `{"detail":"unknown synthesis id"}` | silently ignored (`if (!r.ok) return;`) — practically unreachable in a normal single session (only past-32-syntheses eviction or a copy-pasted stale ID could trigger it) | Yes |
| 24 | Mid-stream synthesis exception | body of `POST /api/synthesize` | **200** (already committed) | no distinct wire signal — see §3.3 | toast `"Synthesis ended early: {error}"` shown *after* the truncated audio | Yes |
| 25 | Benchmark endpoint errors (503/404) | `POST /api/benchmark/ecapa_steps` | 503 / 404 | same shapes as clone | **no UI trigger exists for this endpoint at all** — API-only, dev tooling | N/A |
| 26 | Any unhandled server exception outside explicit `HTTPException`s | any | 500 | FastAPI's default `{"detail":"Internal Server Error"}` | picked up by the generic `detail \|\| fallback` pattern and toasted verbatim | Yes |
| 27 | Network-level failure (server down, DNS, CORS) | any endpoint | *(no HTTP response)* | `regenerate()`/`cloneVoice()`: toast(err.message); `pollStatus()`: auto-retries every 1500ms forever; `refreshVoices()`: silently swallowed, no retry, no toast | Yes for the first two; the void for `refreshVoices()` |
| 28 | Preview audio playback error | `GET /api/voices/{id}/preview` | any (403/404/503/network) | toast `"Preview not ready — the engine may still be loading."` — the underlying status/body is never inspected | Yes |
| 29 | Source audio playback error | `GET /api/voices/{id}/audio` | any | toast `"Could not play the reference audio."` | Yes |

**The one genuinely unrecoverable-without-external-intervention case**:
if the TTS engine itself fails to load at process startup (e.g. model
download failure), `engine.loaded` never becomes `true`. `pollStatus()`
retries forever, showing the pulsing failure message
(`engine.status = "Load failed: {e}"`), and Regenerate/Clone are both
permanently gated behind client-side `if (!state.engineLoaded)` checks
that just show a toast and refuse to even attempt a request. **Reloading
the page does not help** — the failure is a backend/process-level
condition, not a frontend one; the server process itself needs fixing and
restarting.

---

## 5. Metrics — Definition and Computation

| Metric (shown as) | Formula | Computed where | Real measurement, or estimate/seeded-default? |
|---|---|---|---|
| Final RTF (`#result-rtf` after completion, `X-...` not a header but derived post-stream) | `infer_total / audio_s` (both accumulated across every chunk of the stream) | `app.py`, `_stream()`'s `finally` block | **Real**, measured — `infer_total` is the literal sum of each chunk's wall-clock inference time returned by the TTS engine call; `audio_s` is derived from the actual PCM byte count sent. Falls back to `0.0` only if `audio_s` is `0` (nothing was synthesized) — not a placeholder standing in for a real number, just the honest value for "no audio produced." |
| Time-to-first-audio (`ttfa_s`, exposed via the stats endpoint, not directly labeled in the UI but feeds the client's buffering threshold) | wall-clock time from stream start to the moment the *first* chunk's synthesis returns | `app.py`, `_stream()` | **Real**, measured once per stream. |
| Live RTF while generating (`#result-rtf` during generation, and the mode-button "measured RTF" tooltip refresh) | `infer_s / audio_s` computed from the **in-progress, partial** numbers polled from `_SYNTH_STATS[sid]` every 600ms | `app.js`, `startStatsPolling()` | **Real**, just partial/instantaneous rather than final — same underlying real per-chunk measurements, just re-derived client-side from whatever has accumulated so far. |
| Client-side fallback RTF | `((wall-clock end - wall-clock start)/1000 - cloneMs/1000) / audioSeconds` | `app.js`, `regenerate()`, only used **if** the final stats fetch fails or hasn't marked `done` yet | **Real**, but a *different* real measurement than the server's (wall-clock including network/JS overhead, vs. the server's pure-inference sum) — a fallback to an alternate genuine measurement, never a fake/placeholder number. |
| Estimated blend similarity (`est_sim_pct`, e.g. "est 51%") | `dot(w, s_arr) * 100`, where `s_arr` = raw cosine similarities of the top-K built-in voices to the target embedding, `w` = their softmax weights (temperature 0.1) | `ecapa.py`, `smart_init_clone()` | **Real**, computed from real embeddings — but is explicitly an *estimate* by construction (a predicted, blend-weighted number), never presented as a ground-truth achieved result — the UI always labels it "est". |
| Verified similarity (`verified_sim_pct`, "verified X%") | `dot(embed(synthesized_sample), target_emb)` (both L2-normalized) — i.e. actually synthesize a sample with the finished blended voice and re-measure its cosine similarity to the original target embedding | `ecapa.py`, `verify_clone()` | **Real**, measured after the fact — this is the closest thing to ground truth the app produces, computed asynchronously a few seconds after cloning. |
| Per-built-in-voice similarity ranking (`ranking[].sim`) | `dot(target_embedding, voice_embedding)` for each of the 10 built-ins (both L2-normalized) | `ecapa.py`, `smart_init_clone()` | **Real**. |
| Mode-button subtitle RTF hint (`"16 steps · ×0.27"`) | predicted `infer_s` for one full-size (`CHUNK_CHARS`-length) steady-state chunk, from the persistent per-steps model: `fixed_s + rate_rtf * chunk_audio_s`, divided by `chunk_audio_s` | `app.py`, `_display_rtf()`, fed by `RtfHistory` (`app.py`) | **Modeled/predicted**, continuously re-fit from real per-chunk measurements after every synthesis at that steps value (`RtfHistory.update()`) — **but before any real synthesis has ever happened at a given steps value, it falls back to hardcoded seed constants**: `DEFAULT_FIXED_S = 1.0`, `DEFAULT_RATE_RTF = 0.15`, `DEFAULT_CPS = 14.0`. This is the **one metric in the app that can show a number derived from a hardcoded default rather than any real measurement on the current hardware** — it self-corrects to real data after exactly one synthesis at that steps count, and until then it is simply absent from the button (the `× {rtf}` suffix only appears once `rtf_history[steps]` exists as a valid row at all — see the frontend's `renderModes()`, which checks `row && typeof row.rtf === 'number'` before appending anything). |
| Predicted total duration while streaming (`"~0:12"`) | `total_chars / cps_seed + PARAGRAPH_PAUSE_S * (num_paragraphs - 1)` | `app.py`, `api_synthesize()` (`est_audio_s`) | **Modeled/predicted**, same historical-model dependency (and same seeded-default fallback) as the row above; the UI's own `~` prefix makes clear it's an estimate, snapping to the exact figure with no prefix once the stream actually finishes. |
| "Will likely stall" warning | boolean: `streaming AND est_audio_s > 3.0 AND est_infer_s > est_audio_s * 1.05` | `app.py`, `api_synthesize()` | Derived boolean from the same modeled estimates above, not a metric shown as a number — surfaces as a one-time toast warning. |

**Summary on the "no faked metrics" question**: nothing in this codebase
ever presents a hardcoded/fake number *as if* it were a real measurement.
The only seeded-default behavior is the RTF/duration prediction model's
cold-start constants (`DEFAULT_FIXED_S`/`DEFAULT_RATE_RTF`/`DEFAULT_CPS`),
which exist purely to make the *first-ever* prediction at a new steps
value reasonable before any real data exists — and every actual
similarity/RTF/TTFA number shown as a completed result is a genuine
post-hoc measurement, never a placeholder.

---

## 6. Dependency and Runtime Environment

### 6.1 Full pinned dependency list (`requirements.txt`)
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
```
`onnxruntime-gpu` is **not** listed as an installable extra — the
project's convention (per its own README) is: install `onnxruntime-gpu`
manually on a CUDA machine instead of `onnxruntime-directml`; `engine.py`
auto-detects whichever is actually importable at runtime (CUDA >
DirectML > CPU) and picks providers accordingly.

There is **no `package.json`** — the frontend has zero JavaScript/npm
dependencies of any kind; fonts are vendored `.woff2` files committed
directly under `static/fonts/`.

### 6.2 Categorized

| Category | Packages |
|---|---|
| **UI-only** (transport/serving, unrelated to either engine) | `fastapi>=0.100`, `uvicorn[standard]>=0.27`, `python-multipart>=0.0.7` (required specifically because every mutating endpoint uses multipart form bodies, not JSON) |
| **Speaker-encoder-only** (the ECAPA-TDNN seam) | `speechbrain>=1.0` (provides `EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")`), `torch>=2.1` (speechbrain's backend + direct tensor use in `ecapa.py`), `torchaudio>=2.1` (used narrowly for `Resample` only — audio *loading* deliberately avoids `torchaudio.load` in favor of `librosa.load`) |
| **TTS-engine-only** (separate seam, out of scope for "the ECAPA encoder" but relevant to a full engine swap) | `supertonic>=1.3.0`, `onnxruntime-directml` (Windows) / `onnxruntime-gpu` (CUDA, manual install) |
| **Shared by both engine layers** | `numpy>=1.26`, `soundfile>=0.12`, `librosa>=0.10` |

### 6.3 Exact launch command(s)
```bat
:: run.bat (Windows) — the actual shipped launcher
set PY=C:\Users\lauro\.venvs\localvoice\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" "%~dp0app.py"
pause
```
Equivalently, from any shell with the dependencies installed:
```
python app.py
```
This directly calls `uvicorn.run(app, host, port)` in-process (there is
**no** separate `uvicorn app:app` CLI invocation in normal use, though
`.claude/launch.json` also defines an alternate dev configuration that
does: `python -m uvicorn app:app --host 127.0.0.1 --port 7893`).

### 6.4 Environment variables (all optional; no `.env` file is used anywhere)
| Variable | Effect | Default |
|---|---|---|
| `PORT` or `ECAPAFLOW_PORT` | fixed port to bind | unset → auto-tries `7873`, then `7874`–`7876` if busy |
| `ECAPAFLOW_HOST` | bind address | `0.0.0.0` (LAN-reachable — prints a phone-usable LAN URL at startup) |
| `ECAPAFLOW_NO_BROWSER` | if `"1"`, suppresses the automatic browser-open 2 seconds after boot | unset (browser opens automatically) |

### 6.5 No other config files
Beyond the three environment variables above and the hardcoded constants
inside `app.py`/`ecapa.py`/`engine.py` (e.g. `DEFAULT_ECAPA_STEPS = 24`,
`MAX_TEXT_CHARS = 100000`, `CHUNK_CHARS = 300`), there is no `.env`,
`config.json`, `settings.yaml`, or equivalent anywhere in the project.
All *runtime-generated* state (voice registry, RTF history, cached
embeddings, cached previews, the downloaded speechbrain checkpoint) lives
under `data/`, created on demand — none of it needs to pre-exist for a
first run.
