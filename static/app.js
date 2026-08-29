/* ECAPAFlow frontend — pixel-perfect recreation of the Supertonic Voice
 * Builder design, wired to the real backend:
 *   regenerate() → POST /api/synthesize (real Supertonic 3 ONNX)
 *   cloneVoice() → POST /api/voices/{id}/clone (real ECAPA-TDNN smart-init)
 * plus upload/record reference clips (the design's 'uploaded'/'recorded'
 * voice types, which had no UI in the prototype).
 */
'use strict';

const ACCENT = '#FFFFFF';

const CONTENT_TYPES = [
  { key: 'freeform', label: 'Freeform' },
  { key: 'announcement', label: 'Announcement' },
  { key: 'chatbot', label: 'Chatbot' },
  { key: 'article', label: 'Article' },
  { key: 'podcast', label: 'Podcast' },
];

const CONTENT_SAMPLES = {
  freeform: '',
  announcement: 'Attention passengers: the 4:45 express to Central Station is now boarding on platform 3. Please have your tickets ready for inspection.',
  chatbot: 'Hi, thanks for contacting Customer Support. I can help you track an order, start a return, update your delivery address, or troubleshoot a product issue. To begin, share your order number, email, or phone number. If you cannot find it, I can search using your recent activity.',
  article: 'Researchers have unveiled a compact text-to-speech model that runs entirely on-device, eliminating the need for cloud servers while maintaining natural, expressive speech quality across dozens of languages.',
  podcast: 'Welcome back to the show. Today we are diving into the future of voice technology — how on-device AI is changing the way we build products, protect privacy, and ship faster than ever before.',
};

let LANGUAGES = [
  { name: 'Auto', code: 'auto' },
  { name: 'English', code: 'en' }, { name: 'Korean', code: 'ko' },
  { name: 'Spanish', code: 'es' }, { name: 'Portuguese', code: 'pt' },
  { name: 'French', code: 'fr' }, { name: 'Japanese', code: 'ja' },
  { name: 'German', code: 'de' }, { name: 'Arabic', code: 'ar' },
  { name: 'Italian', code: 'it' }, { name: 'Dutch', code: 'nl' },
  { name: 'Russian', code: 'ru' }, { name: 'Turkish', code: 'tr' },
];

// Quality-mode presets (server is the source of truth via /api/status).
// Each button shows the measured RTF for its step count once the backend
// has history for it, so the trade-off is visible before generating.
let MODES = [
  { key: 'speed', label: 'Speed', steps: 8 },
  { key: 'medium', label: 'Medium', steps: 16 },
  { key: 'quality', label: 'Quality', steps: 32 },
];
const MODE_HINTS = {
  speed: 'Best real-time factor — fastest generation, still a decent voice',
  medium: 'The sweet spot — best average quality, still well above real-time',
  quality: 'Maximum quality — generation speed does not matter',
  auto: 'Highest quality this machine still renders faster than real time — '
    + 'resolved per request from the measured RTF history, so it follows the '
    + 'hardware instead of a guess',
};

const state = {
  voices: [],
  activeVoiceId: null,
  isEditingTitle: false,
  contentType: 'chatbot',
  scriptText: CONTENT_SAMPLES.chatbot,
  languageOpen: false,
  language: 'Auto',   // detects the script language per paragraph server-side
  // Quality 16 per Lauro's ear (12 sounded worse to him — RTF 0.60 still
  // streams fine); speed 1.05 = the package's natural-pace default.
  quality: 16,
  speed: 1.05,
  silence: 0.30,
  ecapaSteps: 24,
  maxEcapaSteps: 32,
  streaming: true,        // ON: playback starts while the rest still renders
  autoVoice: false,       // ON: use a per-language-tagged voice per paragraph
  // Text normalizer tier: 'fast' = the regex engine (German/English,
  // microseconds); 'quality' = the same rules plus a num2words pass that
  // reads numbers, dates, currency, ordinals and units in the actual
  // language of the text, across nine languages. Neither is an LLM.
  normMode: 'fast',
  natural: true,          // ON: punctuation-length pauses, breaths, room tone,
                          // de-clicked seams, per-sentence rate variation
  master: true,           // ON: studio chain (HP, de-ess, presence shelf,
                          // soft compression, tiny-room reverb, auto level)
  resultSteps: null,      // steps the server actually used (Auto resolves it)
  rtfHistory: {},         // per-steps measured RTF/CPS from the backend
  sourcePlaying: false,
  previewPlayingId: null, // voice tab whose self-intro preview is playing
  previewLoadingId: null, // …or still being synthesized/fetched
  isRegenerating: false,
  isCloning: false,
  resultPlaying: false,
  resultPlayed: 0,        // seconds of the result actually heard
  resultBuffered: 0,      // seconds of audio received so far
  estDuration: null,      // server's total-duration estimate while streaming
  liveStats: null,        // {rtf, done, total} polled during generation
  resultRTF: null,
  engineLoaded: false,
  engineMessage: 'starting…',
  deviceLabel: '',
  recording: false,
  recordSeconds: 0,
};

const PREFS_KEY = 'ecapaflow_prefs';
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; }
  catch (e) { return {}; }
}
function savePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({
      quality: state.quality, streaming: state.streaming, autoVoice: state.autoVoice,
      natural: state.natural, master: state.master, normMode: state.normMode,
    }));
  } catch (e) { /* private mode — prefs just don't persist */ }
}

let sourceAudio = null;
let previewAudio = null;  // per-tab self-introduction preview
let player = null;        // PCM player (Web Audio), see createStreamPlayer
let synthAbort = null;    // AbortController for the in-flight /api/synthesize
let progressTimer = null;
let statsTimer = null;    // /api/synthesize/stats poll during generation
let toastTimer = null;
let mediaRecorder = null;
let recordChunks = [];
let recordTimer = null;
let recordStream = null;

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ utils */
function formatTime(totalSeconds) {
  const s = Math.max(0, Math.round(totalSeconds || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m + ':' + (r < 10 ? '0' : '') + r;
}

function getActiveVoice() {
  return state.voices.find((v) => v.id === state.activeVoiceId) || state.voices[0] || null;
}

function showToast(message, ms = 3200) {
  const t = $('toast');
  t.textContent = message;
  t.style.display = 'block';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.style.display = 'none'; }, ms);
}

function stopAllAudio() {
  if (sourceAudio) { sourceAudio.pause(); sourceAudio = null; }
  if (previewAudio) { previewAudio.pause(); previewAudio = null; }
  if (player) player.pause();
  state.sourcePlaying = false;
  state.resultPlaying = false;
  const hadPreview = state.previewPlayingId || state.previewLoadingId;
  state.previewPlayingId = null;
  state.previewLoadingId = null;
  if (hadPreview) renderVoices();
  renderSourceButton();
  renderPlayer();
}

/* ------------------------------------------------- PCM player (Web Audio)
 * Plays the 16-bit PCM streamed by /api/synthesize while the server is still
 * generating; every chunk is scheduled sample-exactly after the previous one,
 * so playback is gapless. If the buffer is thinning but hasn't run dry yet,
 * the about-to-play chunk is nudged slightly slower (imperceptible pitch
 * shift) to buy the server more time — WebRTC's NetEQ jitter buffer calls
 * this "preemptive expand" and uses it instead of silence whenever it can.
 * Silence-gap insertion is kept only as the last-resort fallback for when
 * synthesis has already fallen behind too far to disguise.
 * Pause/resume = suspending the AudioContext (freezes its clock, so the
 * played-time math stays exact across pauses). */
function createStreamPlayer(sampleRate, ttfbMs = 0, plan = {}) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  let ctx;
  try { ctx = new Ctx({ sampleRate, latencyHint: 'playback' }); }
  catch (e) { try { ctx = new Ctx({ sampleRate }); } catch (e2) { ctx = new Ctx(); } }
  const parts = [];        // Float32Array chunks, kept for replay
  let totalSamples = 0;
  let scheduled = 0;       // how many parts are already in the audio graph
  let sources = [];
  let started = false;
  let startAt = 0;         // ctx time the playback timeline started
  let nextAt = 0;          // ctx time the next chunk begins
  let gapSeconds = 0;      // silence/stretch overhead subtracted from played-time math
  let pendingGaps = [];    // recovery pauses booked for when they elapse
  let streamDone = false;
  let userPaused = false;  // distinguishes a user pause from a browser suspend
  let stalls = 0;
  let stallSeconds = 0;

  /* ---- live production-rate controller ---------------------------------
   * Ported from QuantFlow's streaming controller. The old rule here sized
   * the start buffer from measured TTFB and then reacted to a thinning
   * buffer; both are proxies. What actually decides whether this take
   * stalls is the rate THIS take is producing audio at, and that is
   * measurable directly.
   *
   * Two things make the measurement honest. The clock starts at the FIRST
   * CHUNK, not at the request — prefill, engine-lock wait and auto-clone
   * are time-to-first-audio, not production rate, and letting them into the
   * estimate would make the controller permanently pessimistic. And the
   * plan takes the WORST of three windows rather than an average: the
   * failure being defended against is a take whose rate is no longer what
   * it was when it started, and an average hides exactly that until the
   * buffer is already gone.
   */
  const RECENT_WINDOW_S = 0.8;   // long enough to be a rate, short enough to react
  const WORST_DECAY = 0.5;       // per CLOSED WINDOW, not per second
  const PRIOR_FADE_S = 3.0;      // server's guess is a floor that fades this fast
  const JITTER_S = Math.max(0.25, Math.min(1.2, (ttfbMs / 1000) * 1.5));
  const MAX_WAIT_S = Math.max(4, Math.min(20, (plan.estAudioS || 8) * 0.9));
  const RATE_FLOOR = 0.90;       // ~1.8 semitones at full deflection
  const RATE_GLIDE = 0.3;        // EMA per pushed chunk — inaudible per step
  const HARD_UNDERRUN_S = 0.02;
  const MIN_REBUFFER_S = 0.12;
  const MAX_REBUFFER_S = 1.5;
  // Share of the remaining audio that is speech. Pauses are pre-rendered
  // silence and cost no inference, so they must not be billed at (rtf - 1).
  const speechShare = (plan.estAudioS > 0 && plan.speechS > 0)
    ? Math.max(0.05, Math.min(1, plan.speechS / plan.estAudioS)) : 1;
  const rtfHi = plan.rtfHi > 0 ? plan.rtfHi : 0;

  let t1 = 0, a1 = 0;            // wall clock / audio seconds at chunk 1
  let winT0 = 0, winA0 = 0;      // current recent window
  let rtfRecent = 0, rtfWorst = 0;
  let rate = 1.0;                // current playbackRate target (glided)
  let lastArrival = 0, gapWorst = 0;   // inter-chunk arrival latency

  function measure() {
    const now = performance.now() / 1000;
    const audio = totalSamples / sampleRate;
    if (lastArrival) gapWorst = Math.max(now - lastArrival, gapWorst * WORST_DECAY);
    lastArrival = now;
    if (!t1) { t1 = now; a1 = audio; winT0 = now; winA0 = audio; return; }
    const wall = now - winT0;
    const made = audio - winA0;
    if (wall >= RECENT_WINDOW_S && made > 0.1) {
      rtfRecent = wall / made;
      rtfWorst = Math.max(rtfRecent, rtfWorst * WORST_DECAY);
      winT0 = now; winA0 = audio;
    }
  }

  // Seconds the next chunk is expected to take to arrive. QuantFlow's
  // controller has no term for this and does not need one: its generation
  // runs near RTF 1.0, so an average rate describes it. Here a chunk is one
  // whole model pass whose FIXED cost is multiple seconds at high step
  // counts — a stream can average RTF 0.5 and still leave the player dry for
  // three seconds waiting for chunk 2, which is exactly what happened on the
  // first real run of this controller (one 3.0 s stall, measured). The
  // cushion therefore has to cover arrival LATENCY as well as average rate.
  function chunkLatency() {
    const observed = Math.max(0, totalSamples / sampleRate - a1);
    const w = Math.max(0, 1 - observed / PRIOR_FADE_S);
    return Math.max(gapWorst, w * (plan.chunkLatS || 0));
  }

  function planRtf() {
    const now = performance.now() / 1000;
    const observed = Math.max(0, totalSamples / sampleRate - a1);
    const rtfCum = (t1 && observed > 0.25) ? (now - t1) / observed : 0;
    const live = Math.max(rtfCum, rtfRecent, rtfWorst);
    // +6%: production is not perfectly smooth, and the costs are asymmetric
    // — being early costs a fraction of a second of waiting, being late
    // costs an audible stall.
    const want = live * 1.06;
    // The server's prior is a FLOOR while there is little evidence, and its
    // weight decays to zero over PRIOR_FADE_S of observed audio. A history
    // row that has drifted can otherwise make the client buffer for ten
    // seconds while its own measurement says the machine is keeping up.
    const w = Math.max(0, 1 - observed / PRIOR_FADE_S);
    return Math.max(want || 0, w * 0.85 * (rtfHi || live));
  }

  // Seconds of audio that must be buffered before playback can start (or,
  // mid-stream, that the cushion is measured against).
  function demandCushion() {
    const produced = totalSamples / sampleRate;
    const remaining = Math.max(0, (plan.estAudioS || 0) - produced);
    const rtf = planRtf();
    const need = remaining * speechShare * Math.max(0, rtf - 1);
    // Whichever is worse: the rate the stream is running at, or the wait for
    // the single next chunk. Both are real ways to run dry.
    const worst = Math.max(need, chunkLatency() * 1.15);
    return Math.min(worst + JITTER_S, MAX_WAIT_S);
  }

  function scheduleNext() {
    if (ctx.state === 'closed') return;
    while (scheduled < parts.length) {
      const f32 = parts[scheduled++];
      const buf = ctx.createBuffer(1, f32.length, sampleRate);
      buf.getChannelData(0).set(f32);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      let cushion = nextAt - ctx.currentTime;
      const demand = streamDone ? 0 : demandCushion();

      if (cushion < HARD_UNDERRUN_S) {
        // Genuine underrun. One recovery pause, SIZED FROM WHAT THE REST OF
        // THE STREAM NEEDS instead of a fixed constant, and booked into the
        // metrics rather than silently absorbed.
        const wait = Math.min(MAX_REBUFFER_S, Math.max(MIN_REBUFFER_S, demand));
        const bumped = ctx.currentTime + wait;
        // Book the pause for WHEN IT ELAPSES, not when it is scheduled. The
        // pause is up to 1.5 s in the future; charging it to played-time
        // immediately made the position readout and the progress bar jump
        // backwards by that much at the moment of scheduling.
        pendingGaps.push({ at: bumped, amount: bumped - nextAt });
        stallSeconds += bumped - nextAt;
        stalls += 1;
        nextAt = bumped;
        rate = 1.0;   // the pause already bought what the glide was earning
        src.start(nextAt);
        nextAt += buf.duration;
      } else {
        // Rate glide: if the cushion is under what the rest of the stream
        // needs, playback simply slows down. NOTE this sets playbackRate,
        // which RESAMPLES — it shifts pitch, it does not time-stretch. At
        // the 0.90 floor that is ~1.8 semitones. Right trade against a hard
        // silence; wrong for music or a tonal language.
        let want = 1.0;
        if (demand > cushion) {
          const deficit = demand - cushion;
          const over = Math.max(1.0, demand + cushion);
          want = Math.max(RATE_FLOOR, 1 - deficit / over);
        }
        rate += (want - rate) * RATE_GLIDE;
        if (rate > 0.999) rate = 1.0;
        src.playbackRate.value = rate;
        src.start(nextAt);
        const played = buf.duration / rate;
        gapSeconds += played - buf.duration;
        nextAt += played;
      }
      sources.push(src);
    }
  }

  return {
    push(f32) {
      parts.push(f32);
      totalSamples += f32.length;
      measure();
      if (started) scheduleNext();
    },
    // What must be buffered before playback starts. Recomputed on every
    // check from the live rate, so it tightens as evidence arrives instead
    // of being decided once from a constant.
    startThresholdSeconds() { return demandCushion(); },
    planRtf() { return planRtf(); },
    stallInfo() { return { stalls, stallSeconds, rate }; },
    start() {
      if (started || ctx.state === 'closed') return;
      started = true;
      userPaused = false;
      startAt = nextAt = ctx.currentTime + 0.05;
      gapSeconds = 0;
      pendingGaps = [];
      if (ctx.state === 'suspended') ctx.resume().catch(() => {});
      scheduleNext();
    },
    restart() {
      for (const s of sources) { try { s.stop(); } catch (e) { /* done */ } }
      sources = [];
      scheduled = 0;
      started = false;
      this.start();
    },
    pause() { userPaused = true; if (ctx.state === 'running') ctx.suspend().catch(() => {}); },
    resume() { userPaused = false; if (ctx.state === 'suspended') ctx.resume().catch(() => {}); },
    // Watchdog: the browser (or audio driver) can suspend the context on its
    // own — e.g. power saving. If the user didn't pause, kick it back on.
    ensureRunning() {
      if (started && !userPaused && ctx.state !== 'running' && ctx.state !== 'closed') {
        ctx.resume().catch((e) => console.warn('audio resume failed:', e));
      }
    },
    ctxState() { return ctx.state; },
    // Production has finished: zero the cushion demand so playback runs at
    // exact rate instead of continuing to defend against generation that is
    // no longer happening.
    finishStream() { streamDone = true; rate = 1.0; },
    isStreamDone() { return streamDone; },
    // Assemble the buffered PCM parts into a complete downloadable WAV.
    toWavBlob() {
      const buf = new ArrayBuffer(44 + totalSamples * 2);
      const dv = new DataView(buf);
      const str = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
      str(0, 'RIFF'); dv.setUint32(4, 36 + totalSamples * 2, true); str(8, 'WAVE');
      str(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
      dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
      dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
      str(36, 'data'); dv.setUint32(40, totalSamples * 2, true);
      let off = 44;
      for (const p of parts) {
        for (let i = 0; i < p.length; i++, off += 2) {
          dv.setInt16(off, Math.max(-32768, Math.min(32767, Math.round(p[i] * 32767))), true);
        }
      }
      return new Blob([buf], { type: 'audio/wav' });
    },
    hasStarted() { return started; },
    totalSeconds() { return totalSamples / sampleRate; },
    playedSeconds() {
      if (!started) return 0;
      // Recovery pauses count only once the clock has actually reached them.
      while (pendingGaps.length && pendingGaps[0].at <= ctx.currentTime) {
        gapSeconds += pendingGaps.shift().amount;
      }
      return Math.min(Math.max(0, ctx.currentTime - startAt - gapSeconds), this.totalSeconds());
    },
    isPlaying() { return started && ctx.state === 'running' && !this.isEnded(); },
    isEnded() {
      return started && streamDone && totalSamples > 0 &&
             ctx.state === 'running' && ctx.currentTime >= nextAt - 0.005;
    },
    destroy() {
      for (const s of sources) { try { s.stop(); } catch (e) { /* done */ } }
      sources = [];
      try { ctx.close(); } catch (e) { /* already closed */ }
    },
  };
}

function startProgressTicker() {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = setInterval(() => {
    if (!player) return;
    player.ensureRunning();
    state.resultBuffered = player.totalSeconds();
    if (player.isEnded()) {
      state.resultPlaying = false;
      state.resultPlayed = 0;
    } else if (player.hasStarted()) {
      state.resultPlaying = player.isPlaying();
      state.resultPlayed = player.playedSeconds();
    }
    renderPlayer();
  }, 150);
}

function startStatsPolling(sid) {
  if (statsTimer) clearInterval(statsTimer);
  statsTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/synthesize/stats/${sid}`);
      if (!r.ok) return;
      const st = await r.json();
      state.liveStats = {
        rtf: st.audio_s > 0 ? st.infer_s / st.audio_s : null,
        done: st.chunks_done || 0,
        total: st.chunks || 0,
      };
      if (st.done) stopStatsPolling();
    } catch (e) { /* transient — next tick retries */ }
  }, 600);
}

function stopStatsPolling() {
  if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
}

/* ------------------------------------------------------------- icons (svg) */
const svgPlay = (size) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="currentColor" style="display:block"><polygon points="6 3 20 12 6 21 6 3"></polygon></svg>`;
const svgPause = (size) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="currentColor" style="display:block"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;
const svgCheck = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
const svgX = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
const svgPencil = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path></svg>`;
const svgPlus = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>`;
const svgMic = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line></svg>`;

/* ------------------------------------------------------------- voice tabs */
function renderVoices() {
  const row = $('voice-tabs');
  row.innerHTML = '';

  for (const v of state.voices) {
    const active = v.id === state.activeVoiceId;
    const pill = document.createElement('div');
    pill.style.cssText = `display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; border-radius: 999px; border: 1px solid ${active ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.14)'}; background: ${active ? 'rgba(255,255,255,0.07)' : '#0e0e0e'}; font-size: 14px; color: ${active ? '#f8f8f4' : '#c8c8c2'}; cursor: pointer; white-space: nowrap;`;
    if (v.cloned && v.clone_info && v.clone_info.verified_sim_pct != null) {
      pill.title = `verified ECAPA similarity ${v.clone_info.verified_sim_pct}%`;
    }
    // Tiny preview button: the voice introduces itself in its own style
    // (synthesized server-side and cached; falls back to the reference clip
    // for voices that were never cloned).
    const isPreviewing = state.previewPlayingId === v.id;
    const isPreviewLoading = state.previewLoadingId === v.id;
    const prev = document.createElement('span');
    prev.title = 'Preview — hear this voice introduce itself';
    prev.style.cssText = 'width: 20px; height: 20px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.35); display: flex; align-items: center; justify-content: center; flex-shrink: 0;'
      + (isPreviewLoading ? 'animation: stbPulse 1s ease-in-out infinite;' : '');
    prev.innerHTML = isPreviewing ? svgPause(8) : svgPlay(8);
    prev.addEventListener('click', (e) => { e.stopPropagation(); togglePreview(v.id); });
    pill.appendChild(prev);

    // Status badge — the one thing that tells the user whether this voice
    // can be used at all. A voice is usable ONLY when it is fully cloned.
    const st = v.status || (v.cloned ? 'ready' : 'new');
    const job = v.clone_job;
    if (st === 'ready') {
      const chk = document.createElement('span');
      chk.style.cssText = 'display: flex; line-height: 0;';
      chk.innerHTML = svgCheck;
      pill.appendChild(chk);
    } else {
      const badge = document.createElement('span');
      const look = {
        training: ['#f4c25a', 'cloning'],
        new: ['#8a8a84', 'not cloned'],
        failed: ['#e07b6a', 'failed'],
        'no-ref': ['#8a8a84', 'style only'],
      }[st] || ['#8a8a84', st];
      badge.textContent = st === 'training' && job
        ? `${Math.min(99, Math.round((job.elapsed_s / Math.max(1, job.budget_s)) * 100))}%`
        : look[1];
      badge.style.cssText = `font-size: 10px; font-family: 'JetBrains Mono', monospace; color: ${look[0]}; border: 1px solid ${look[0]}55; border-radius: 5px; padding: 1px 5px;`
        + (st === 'training' ? 'animation: stbPulse 1.6s ease-in-out infinite;' : '');
      pill.appendChild(badge);
    }
    const name = document.createElement('span');
    name.textContent = v.name;
    if (st !== 'ready') name.style.opacity = '0.6';
    pill.appendChild(name);

    // Language tag: the app's OWN organizational label, not a Supertonic
    // claim — Supertonic's voices are architecturally decoupled timbres
    // usable with any of its 31 languages ("no per-language fine-tuning",
    // per the official docs), so this only drives Auto Voice mode (pick
    // the voice tagged for a paragraph's language), it isn't a quality
    // ranking. Defaults to "Any" (untagged) for every voice.
    const langTag = document.createElement('select');
    langTag.title = "Tag this voice's language — your own organization, "
      + "used by Auto Voice mode. Supertonic voices aren't language-"
      + 'specific by design, so this is not a quality claim.';
    langTag.style.cssText = `background: transparent; border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; color: ${v.lang ? '#e8e8e2' : '#5c5c56'}; font-size: 11px; font-family: inherit; padding: 1px 2px; cursor: pointer; max-width: 62px;`;
    const anyOpt = document.createElement('option');
    anyOpt.value = '';
    anyOpt.textContent = 'Any';
    langTag.appendChild(anyOpt);
    for (const lng of LANGUAGES) {
      if (lng.code === 'auto') continue;
      const opt = document.createElement('option');
      opt.value = lng.code;
      opt.textContent = lng.name;
      langTag.appendChild(opt);
    }
    langTag.value = v.lang || '';
    langTag.addEventListener('click', (e) => e.stopPropagation());
    langTag.addEventListener('change', () => setVoiceLang(v.id, langTag.value));
    pill.appendChild(langTag);

    const close = document.createElement('span');
    close.style.cssText = 'display: flex; align-items: center; justify-content: center; color: #74746c; padding: 2px;';
    close.innerHTML = svgX;
    close.addEventListener('click', (e) => { e.stopPropagation(); removeVoice(v.id); });
    pill.appendChild(close);

    pill.addEventListener('click', () => selectVoice(v.id));
    row.appendChild(pill);
  }

  // Add-voice pills (upload / record) — backend voice types that had no UI
  // in the prototype; same pill geometry, dashed border to read as "new".
  const addPill = (iconHtml, label, onClick) => {
    const pill = document.createElement('div');
    pill.style.cssText = 'display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; border-radius: 999px; border: 1px dashed rgba(255,255,255,0.22); background: transparent; font-size: 14px; color: #a8a8a0; cursor: pointer; white-space: nowrap;';
    pill.innerHTML = `<span style="display:flex;line-height:0;">${iconHtml}</span><span>${label}</span>`;
    pill.addEventListener('mouseenter', () => { pill.style.borderColor = 'rgba(255,255,255,0.45)'; pill.style.color = '#e8e8e2'; });
    pill.addEventListener('mouseleave', () => { pill.style.borderColor = 'rgba(255,255,255,0.22)'; pill.style.color = '#a8a8a0'; });
    pill.addEventListener('click', onClick);
    row.appendChild(pill);
    return pill;
  };

  addPill(svgPlus, 'Upload Voice', () => $('file-input').click());

  if (state.recording) {
    const pill = addPill(
      `<span style="width: 8px; height: 8px; border-radius: 50%; background: #f6f6f2; display: inline-block; animation: stbPulse 1s ease-in-out infinite;"></span>`,
      `Recording ${formatTime(state.recordSeconds)} — tap to stop`,
      stopRecording
    );
    pill.style.borderStyle = 'solid';
    pill.style.borderColor = 'rgba(255,255,255,0.45)';
    pill.style.color = '#f0f0ec';
  } else {
    addPill(svgMic, 'Record Voice', startRecording);
  }

  const has = state.voices.length > 0;
  $('empty-state').style.display = has ? 'none' : 'block';
  $('main-panel').style.display = has ? 'block' : 'none';
}

function selectVoice(id) {
  state.activeVoiceId = id;
  state.isEditingTitle = false;
  stopAllAudio();
  clearResult();
  renderVoices();
  renderTitle();
  renderSourceButton();
  renderCloneButton();
}

async function removeVoice(id) {
  try { await fetch(`/api/voices/${id}`, { method: 'DELETE' }); } catch (e) { /* keep UI responsive */ }
  state.voices = state.voices.filter((v) => v.id !== id);
  if (state.activeVoiceId === id) {
    state.activeVoiceId = state.voices.length ? state.voices[0].id : null;
    stopAllAudio();
    clearResult();
  }
  renderVoices();
  renderTitle();
  renderSourceButton();
  renderCloneButton();
}

/* ------------------------------------------------------------------ title */
function renderTitle() {
  const block = $('title-block');
  block.innerHTML = '';
  const v = getActiveVoice();
  if (!v) return;

  if (state.isEditingTitle) {
    const input = document.createElement('input');
    input.value = v.name;
    input.style.cssText = "font-size: 30px; font-weight: 600; background: transparent; border: none; border-bottom: 1px solid rgba(255,255,255,0.3); color: #f6f6f2; font-family: inherit; outline: none; padding: 2px 0;";
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') commitTitle(input.value);
      if (e.key === 'Escape') { state.isEditingTitle = false; renderTitle(); }
    });
    input.addEventListener('blur', () => commitTitle(input.value));
    block.appendChild(input);
    input.focus();
    input.select();
  } else {
    const title = document.createElement('div');
    title.style.cssText = 'font-size: 30px; font-weight: 600; letter-spacing: -0.01em;';
    title.textContent = v.name;
    block.appendChild(title);

    const btn = document.createElement('button');
    btn.className = 'hov-text';
    btn.style.cssText = 'background: transparent; border: none; color: #74746c; cursor: pointer; padding: 6px; display: flex;';
    btn.innerHTML = svgPencil;
    btn.addEventListener('click', () => { state.isEditingTitle = true; renderTitle(); });
    block.appendChild(btn);
  }
}

async function commitTitle(raw) {
  const v = getActiveVoice();
  state.isEditingTitle = false;
  const name = (raw || '').trim();
  if (v && name && name !== v.name) {
    v.name = name;
    try {
      const fd = new FormData();
      fd.append('name', name);
      await fetch(`/api/voices/${v.id}`, { method: 'PATCH', body: fd });
    } catch (e) { /* rename persists on next sync */ }
  }
  renderTitle();
  renderVoices();
}

async function setVoiceLang(voiceId, code) {
  const v = state.voices.find((x) => x.id === voiceId);
  const prev = v ? v.lang : null;
  if (v) v.lang = code || null;  // optimistic, so the select reflects the pick instantly
  try {
    const fd = new FormData();
    fd.append('lang', code || 'any');
    const res = await fetch(`/api/voices/${voiceId}`, { method: 'PATCH', body: fd });
    if (!res.ok) throw new Error('failed');
  } catch (e) {
    if (v) v.lang = prev;  // roll back on failure
    renderVoices();
    showToast('Could not set the voice language tag.');
  }
}

/* --------------------------------------------------------- voice previews */
function togglePreview(id) {
  const wasThis = state.previewPlayingId === id || state.previewLoadingId === id;
  stopAllAudio();
  if (wasThis) return;
  state.previewLoadingId = id;   // first click may synthesize server-side
  renderVoices();
  const audio = new Audio(`/api/voices/${id}/preview`);
  previewAudio = audio;
  audio.addEventListener('playing', () => {
    if (previewAudio !== audio) return;
    state.previewLoadingId = null;
    state.previewPlayingId = id;
    renderVoices();
  });
  audio.addEventListener('ended', () => {
    if (previewAudio !== audio) return;
    state.previewPlayingId = null;
    renderVoices();
  });
  audio.addEventListener('error', () => {
    if (previewAudio !== audio) return;
    state.previewLoadingId = null;
    state.previewPlayingId = null;
    renderVoices();
    showToast('Preview not ready — the engine may still be loading.');
  });
  audio.play().catch(() => { /* surfaced via the error listener */ });
}

/* ----------------------------------------------------------- source voice */
function renderSourceButton() {
  const v = getActiveVoice();
  const btn = $('source-btn');
  // Native voices play a synthesized self-introduction as their source;
  // only style-only imported voices have nothing to play.
  const hasAudio = v && (v.ref_filename || v.type === 'builtin');
  if (!hasAudio) { btn.style.display = 'none'; return; }
  btn.style.display = 'inline-flex';
  const dur = v.duration_s ? ` ${formatTime(v.duration_s)}` : '';
  $('source-label').textContent = (v.type === 'builtin' ? 'Preview' : 'Source Voice') + dur;
  $('source-icon').innerHTML = state.sourcePlaying ? svgPause(9) : svgPlay(9);
}

function toggleSourcePlay() {
  const wasPlaying = state.sourcePlaying;
  stopAllAudio();
  if (wasPlaying) return;
  const v = getActiveVoice();
  if (!v) return;
  if (!v.ref_filename && v.type !== 'builtin') {
    showToast('This imported voice has no reference audio attached.');
    return;
  }
  sourceAudio = new Audio(`/api/voices/${v.id}/audio`);
  sourceAudio.addEventListener('ended', () => { state.sourcePlaying = false; renderSourceButton(); });
  sourceAudio.addEventListener('error', () => {
    state.sourcePlaying = false;
    renderSourceButton();
    showToast('Could not play the reference audio.');
  });
  sourceAudio.play().then(() => {
    state.sourcePlaying = true;
    renderSourceButton();
  }).catch(() => showToast('Could not play the reference audio.'));
}

/* ---------------------------------------------------------- content types */
function renderContentTypes() {
  const wrap = $('content-types');
  wrap.innerHTML = '';
  for (const ct of CONTENT_TYPES) {
    const active = ct.key === state.contentType;
    const el = document.createElement('div');
    el.style.cssText = `font-size: 15px; color: ${active ? '#f8f8f4' : '#84847c'}; font-weight: ${active ? 600 : 400}; text-decoration: ${active ? 'underline' : 'none'}; cursor: pointer; text-underline-offset: 5px;`;
    el.textContent = ct.label;
    el.addEventListener('click', () => {
      state.contentType = ct.key;
      const sample = CONTENT_SAMPLES[ct.key];
      if (sample !== undefined) {
        state.scriptText = sample;
        $('script').value = sample;
        renderCharCount();
      }
      renderContentTypes();
    });
    wrap.appendChild(el);
  }
}

/* -------------------------------------------------------------- languages */
function renderLanguageMenu() {
  $('lang-name').textContent = state.language;
  const menu = $('lang-menu');
  menu.style.display = state.languageOpen ? 'block' : 'none';
  menu.innerHTML = '';
  for (const lng of LANGUAGES) {
    const selected = lng.name === state.language;
    const item = document.createElement('div');
    item.style.cssText = `padding: 9px 12px; border-radius: 8px; font-size: 14px; color: ${selected ? '#f8f8f4' : '#c8c8c2'}; background: ${selected ? 'rgba(255,255,255,0.08)' : 'transparent'}; cursor: pointer;`;
    item.textContent = lng.name;
    item.addEventListener('click', () => {
      state.language = lng.name;
      state.languageOpen = false;
      renderLanguageMenu();
    });
    menu.appendChild(item);
  }
}

function langCode() {
  const l = LANGUAGES.find((x) => x.name === state.language);
  return l ? l.code : 'en';
}

/* ------------------------------------------------------------ mode buttons */
function renderModes() {
  const wrap = $('mode-buttons');
  if (!wrap) return;
  wrap.innerHTML = '';
  for (const m of MODES) {
    const active = state.quality === m.steps;
    const row = state.rtfHistory && state.rtfHistory[String(m.steps)];
    const rtf = row && typeof row.rtf === 'number' ? row.rtf : null;
    const btn = document.createElement('button');
    btn.className = 'hov-border35';
    btn.title = (MODE_HINTS[m.key] || '') + (rtf !== null ? ` · measured RTF ${rtf.toFixed(2)}` : '');
    btn.style.cssText = `display: inline-flex; flex-direction: column; align-items: center; gap: 2px; padding: 8px 18px; border-radius: 12px; border: 1px solid ${active ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.16)'}; background: ${active ? 'rgba(255,255,255,0.07)' : '#0e0e0e'}; color: ${active ? '#f8f8f4' : '#a8a8a0'}; font-size: 14px; font-weight: 600; font-family: inherit; cursor: pointer;`;
    const sub = m.steps === 0
      ? (state.resultSteps ? `${state.resultSteps} steps · adaptive` : 'under real time')
      : `${m.steps} steps${rtf !== null ? ' · ×' + rtf.toFixed(2) : ''}`;
    btn.innerHTML = `<span>${m.label}</span>`
      + `<span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 400; color: ${active ? '#c2c2ba' : '#74746c'};">`
      + `${sub}</span>`;
    btn.addEventListener('click', () => {
      state.quality = m.steps;
      savePrefs();
      const sl = SLIDERS.find((s) => s.key === 'quality');
      if (sl && sl.el) renderSlider(sl);
      renderModes();
    });
    wrap.appendChild(btn);
  }
}

function renderStreamToggle() {
  const btn = $('stream-toggle');
  if (!btn) return;
  $('stream-label').textContent = state.streaming ? 'Streaming ON' : 'Streaming OFF';
  $('stream-dot').style.background = state.streaming ? '#f6f6f2' : 'rgba(255,255,255,0.25)';
  btn.style.color = state.streaming ? '#f0f0ec' : '#84847c';
  btn.title = state.streaming
    ? 'Streaming ON: playback starts instantly while the rest still generates'
    : 'Streaming OFF: generate everything first, then play';
}

function renderAutoVoiceToggle() {
  const btn = $('auto-voice-toggle');
  if (!btn) return;
  $('auto-voice-label').textContent = state.autoVoice ? 'Auto Voice ON' : 'Auto Voice OFF';
  $('auto-voice-dot').style.background = state.autoVoice ? '#f6f6f2' : 'rgba(255,255,255,0.25)';
  btn.style.color = state.autoVoice ? '#f0f0ec' : '#84847c';
  btn.title = state.autoVoice
    ? "Auto Voice ON: each paragraph plays in whichever voice you've tagged "
      + 'for its language (tag a voice from its tab menu), falling back to '
      + 'the selected voice otherwise. Supertonic voices are not '
      + 'language-specific by design — tags are your own organization, not '
      + 'a quality claim.'
    : 'Auto Voice OFF: the selected voice is always used, in whatever language the script is';
}

function renderNaturalToggle() {
  const btn = $('natural-toggle');
  if (!btn) return;
  $('natural-label').textContent = state.natural ? 'Natural ON' : 'Natural OFF';
  $('natural-dot').style.background = state.natural ? '#f6f6f2' : 'rgba(255,255,255,0.25)';
  btn.style.color = state.natural ? '#f0f0ec' : '#84847c';
  btn.title = state.natural
    ? 'Natural ON: pauses scale with the punctuation that caused them, the '
      + 'speaker breathes before long phrases, the silence carries the '
      + "recording's own room tone, seams are de-clicked and each sentence "
      + 'varies its rate by a few percent. Costs no inference time.'
    : 'Natural OFF: one fixed gap after every sentence, digital silence, '
      + 'hard chunk seams — the reference to A/B against';
}

function renderNormToggle() {
  const btn = $('norm-toggle');
  if (!btn) return;
  const q = state.normMode === 'quality';
  $('norm-label').textContent = q ? 'Text: Quality' : 'Text: Fast';
  $('norm-dot').style.background = q ? '#f6f6f2' : 'rgba(255,255,255,0.25)';
  btn.style.color = q ? '#f0f0ec' : '#84847c';
  btn.title = q
    ? 'Quality: numbers, dates, times, currency, ordinals, ranges, fractions, '
      + 'roman numerals and units are read in the ACTUAL language of the text, '
      + 'across nine languages (num2words). Costs a fraction of a millisecond '
      + 'per sentence. Not an LLM.'
    : 'Fast: the German/English regex rules only. Microseconds per sentence. '
      + 'Other languages keep their digits, which the engine reads natively — '
      + 'switch to Quality to have them spoken as words.';
}

function renderMasterToggle() {
  const btn = $('master-toggle');
  if (!btn) return;
  $('master-label').textContent = state.master ? 'Studio ON' : 'Studio OFF';
  $('master-dot').style.background = state.master ? '#f6f6f2' : 'rgba(255,255,255,0.25)';
  btn.style.color = state.master ? '#f0f0ec' : '#84847c';
  btn.title = state.master
    ? 'Studio ON: 75 Hz high-pass, gentle de-esser, +2.5 dB presence shelf at '
      + '6 kHz, 2:1 soft compression, 3.5% tiny-room reverb and one auto-level '
      + 'move — the same polish LoudFlow gets from its Pedalboard chain, run '
      + 'continuously across chunks AND pauses'
    : 'Studio OFF: raw engine output, no EQ, no dynamics, no room';
}

/* ---------------------------------------------------------------- sliders */
const SLIDERS = [
  { key: 'quality', label: 'Quality', min: 4, max: 64, step: 1, fmt: (v) => String(v) },
  { key: 'speed', label: 'Speed', min: 0.5, max: 2.0, step: 0.05, fmt: (v) => v.toFixed(2) },
  { key: 'silence', label: 'Silence Duration', min: 0, max: 1.0, step: 0.05, fmt: (v) => v.toFixed(2) },
  // The fourth bar — ECAPA encoder steps: how many 3s windows of the
  // reference the speaker encoder averages when cloning. Lives inside the
  // Voice Cloning section. Quality-first default is calibrated by benchmark.
  { key: 'ecapaSteps', label: 'ECAPA Encoder Steps', min: 1, max: 32, step: 1, fmt: (v) => String(v), mount: 'clone-sliders' },
];

function buildSliders() {
  $('sliders').innerHTML = '';
  $('clone-sliders').innerHTML = '';
  for (const sl of SLIDERS) {
    const rowWrap = $(sl.mount || 'sliders');
    if (sl.key === 'ecapaSteps') sl.max = state.maxEcapaSteps;
    const col = document.createElement('div');
    col.style.cssText = 'display: flex; flex-direction: column; gap: 12px;';

    const label = document.createElement('div');
    label.style.cssText = 'font-size: 13px; color: #a8a8a0;';
    label.textContent = sl.label;
    col.appendChild(label);

    const row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 14px;';

    const track = document.createElement('div');
    track.style.cssText = 'position: relative; width: 150px; height: 16px; display: flex; align-items: center; cursor: pointer; touch-action: none;';
    track.innerHTML = `
      <div style="position: absolute; left: 0; right: 0; height: 2px; background: rgba(255,255,255,0.18); border-radius: 2px;"></div>
      <div data-fill style="position: absolute; left: 0; width: 0%; height: 2px; background: #f0f0ec; border-radius: 2px;"></div>
      <div data-knob style="position: absolute; left: 0%; width: 14px; height: 14px; margin-left: -7px; border-radius: 50%; background: #f6f6f2; border: 1px solid rgba(0,0,0,0.2);"></div>`;

    const value = document.createElement('div');
    value.style.cssText = "font-size: 14px; color: #f0f0ec; font-family: 'JetBrains Mono', monospace; min-width: 36px;";

    sl.el = { fill: track.querySelector('[data-fill]'), knob: track.querySelector('[data-knob]'), value };

    track.addEventListener('pointerdown', (e) => {
      const rect = track.getBoundingClientRect();
      const update = (clientX) => {
        let pct = rect.width ? (clientX - rect.left) / rect.width : 0;
        pct = Math.min(1, Math.max(0, pct));
        let val = sl.min + pct * (sl.max - sl.min);
        val = Math.round(val / sl.step) * sl.step;
        val = parseFloat(val.toFixed(2));
        state[sl.key] = val;
        renderSlider(sl);
        if (sl.key === 'quality') { savePrefs(); renderModes(); }
      };
      update(e.clientX);
      const onMove = (ev) => update(ev.clientX);
      const onUp = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
      };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
    });

    row.appendChild(track);
    row.appendChild(value);
    col.appendChild(row);
    rowWrap.appendChild(col);
    renderSlider(sl);
  }
}

function renderSlider(sl) {
  const v = state[sl.key];
  // Quality 0 is the Auto sentinel (steps resolved server-side, see MODES):
  // show the resolved value if the last render reported one, and park the
  // knob at whatever that value is instead of off the left end of the track.
  const shown = (sl.key === 'quality' && v === 0)
    ? (state.resultSteps || sl.min) : v;
  const pct = Math.max(0, Math.min(100,
    ((shown - sl.min) / (sl.max - sl.min)) * 100));
  sl.el.fill.style.width = pct + '%';
  sl.el.knob.style.left = pct + '%';
  sl.el.value.textContent = (sl.key === 'quality' && v === 0)
    ? (state.resultSteps ? 'auto ' + state.resultSteps : 'auto') : sl.fmt(v);
}

/* -------------------------------------------------------------- charcount */
function renderCharCount() {
  $('char-count').textContent = String(state.scriptText.length);
}

/* ------------------------------------------------------------- result bar */
function clearResult() {
  if (synthAbort) { synthAbort.abort(); synthAbort = null; }
  if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
  stopStatsPolling();
  if (player) { player.destroy(); player = null; }
  state.resultPlaying = false;
  state.resultPlayed = 0;
  state.resultBuffered = 0;
  state.estDuration = null;
  state.liveStats = null;
  state.resultRTF = null;
  renderPlayer();
}

function renderPlayer() {
  $('result-icon').innerHTML = state.resultPlaying ? svgPause(13) : svgPlay(13);
  $('result-icon').style.color = '#f6f6f2';

  // While generating, the bar's total is the server's duration estimate, so
  // the buffered (dim) fill visibly GROWS toward it chunk by chunk; once the
  // stream ends the total snaps to the exact length.
  const buffered = state.resultBuffered;
  const done = !player || player.isStreamDone();
  const total = Math.max(done ? buffered : Math.max(state.estDuration || 0, buffered), 0.01);
  $('result-current').textContent = formatTime(state.resultPlayed);
  $('result-duration').textContent = (done ? '' : '~') + formatTime(total);
  $('result-buffered').style.width = Math.min(100, (buffered / total) * 100) + '%';
  $('result-progress').style.width = Math.min(100, (state.resultPlayed / total) * 100) + '%';

  const rtfEl = $('result-rtf');
  if (state.isRegenerating && state.liveStats) {
    const ls = state.liveStats;
    const parts = [];
    if (ls.rtf !== null && isFinite(ls.rtf)) parts.push('RTF ' + ls.rtf.toFixed(2));
    if (ls.total) parts.push(ls.done + '/' + ls.total);
    rtfEl.textContent = parts.join(' · ');
    rtfEl.style.display = parts.length ? 'inline' : 'none';
  } else if (state.resultRTF !== null) {
    rtfEl.textContent = 'RTF ' + state.resultRTF.toFixed(2);
    rtfEl.style.display = 'inline';
  } else {
    rtfEl.style.display = 'none';
  }

  // Download appears once generation has finished (also after Stop — the
  // partial audio is a complete, saveable WAV). Never during generation:
  // the file would be incomplete.
  const canDownload = player && player.isStreamDone() && player.totalSeconds() > 0;
  $('download-btn').style.display = canDownload ? 'inline-flex' : 'none';
}

function downloadResult() {
  if (!player || !player.isStreamDone()) return;
  const v = getActiveVoice();
  const name = ((v && v.name) || 'voice').replace(/[^\w\-. ]+/g, '').trim() || 'voice';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(player.toWavBlob());
  a.download = `${name}.wav`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

function renderRegenButton() {
  // While generating, the button turns into a Stop control (spinner stays
  // on to signal that synthesis is still running). Auto-clone runs
  // synchronously on the server BEFORE any audio streams back, so without
  // this the button would just say "Stop" for however long cloning takes
  // with no indication anything besides "generating" is happening.
  $('regen-spinner').style.display = state.isRegenerating ? 'block' : 'none';
  if (!state.isRegenerating) {
    $('regen-label').textContent = 'Regenerate';
  } else {
    $('regen-label').textContent = state.regenPhase === 'cloning' ? 'Cloning voice…' : 'Stop';
  }
}

function stopGeneration() {
  // Abort the in-flight stream; the audio buffered so far stays playable
  // and the player ends it like a normally finished stream.
  if (synthAbort) { synthAbort.abort(); synthAbort = null; }
  if (player) player.finishStream();
  stopStatsPolling();
  state.estDuration = null;
  state.liveStats = null;
}

function renderCloneButton() {
  const v = getActiveVoice();
  const btn = $('clone-btn');
  const status = $('clone-status');
  const job = v && v.clone_job;
  // A clone in progress owns this whole panel: what it is doing, how far in,
  // how long is left, and what it has achieved so far. No spinner-with-no-
  // information, and no way to mistake a half-trained voice for a finished one.
  if (job && ['starting', 'smart-init', 'searching', 'verifying'].includes(job.phase)) {
    const pct = Math.min(99, Math.round((job.elapsed_s / Math.max(1, job.budget_s)) * 100));
    btn.textContent = `Cloning ${pct}% — Cancel`;
    btn.style.opacity = '1';
    const mins = Math.max(0, Math.round(job.eta_s / 60));
    const bits = [job.message || job.phase];
    if (job.best_sim) bits.push(`match ${job.best_sim.toFixed(1)}%`);
    if (job.baseline_sim) bits.push(`instant clone was ${job.baseline_sim.toFixed(1)}%`);
    bits.push(mins >= 1 ? `~${mins} min left` : `~${Math.max(1, Math.round(job.eta_s))}s left`);
    status.textContent = bits.join(' · ');
    return;
  }
  if (state.isCloning) {
    btn.textContent = 'Starting…';
    btn.style.opacity = '0.7';
  } else if (v && v.type === 'builtin') {
    btn.textContent = 'Native Voice ✓';
    btn.style.opacity = '1';
  } else if (v && v.type === 'imported') {
    btn.textContent = 'Imported 1:1 ✓';
    btn.style.opacity = '1';
  } else {
    btn.textContent = v && v.cloned ? 'Voice Cloned ✓' : 'Clone Voice';
    btn.style.opacity = '1';
  }

  if (state.isCloning) {
    status.textContent = 'starting…';
  } else if (v && v.type === 'builtin') {
    status.textContent = 'Supertonic 3 built-in ' + ((v.clone_info && v.clone_info.voice_code) || '');
  } else if (v && v.type === 'imported' && v.clone_info) {
    const ci = v.clone_info;
    const parts = ['imported'];
    if (ci.mode) parts.push(ci.mode);
    if (ci.iterations) parts.push(ci.iterations + ' iters');
    if (ci.trained_sim_pct != null) parts.push('trained ' + Math.round(ci.trained_sim_pct) + '%');
    if (ci.verified_sim_pct != null) parts.push('verified ' + Math.round(ci.verified_sim_pct) + '%');
    status.textContent = parts.join(' · ');
  } else if (v && v.cloned && v.clone_info) {
    const ci = v.clone_info;
    const parts = [];
    if (ci.verified_sim_pct != null) parts.push('match ' + Number(ci.verified_sim_pct).toFixed(1) + '%');
    else if (ci.est_sim_pct != null) parts.push('est ' + Math.round(ci.est_sim_pct) + '%');
    if (ci.improved_points != null) {
      parts.push((ci.improved_points >= 0 ? '+' : '') + Number(ci.improved_points).toFixed(1)
        + ' pts over instant');
    }
    if (ci.depth) parts.push(ci.depth + ' clone');
    if (ci.search_evals) parts.push(ci.search_evals + ' candidates');
    if (ci.search_seconds) parts.push(Math.round(ci.search_seconds / 60) + ' min');
    if (!parts.length && ci.timings_ms && ci.timings_ms.total != null) {
      parts.push((ci.timings_ms.total / 1000).toFixed(2) + 's');
    }
    if (ci.ranking && ci.ranking.length) parts.push(ci.ranking.slice(0, 3).map((r) => r.voice).join('+'));
    status.textContent = parts.join(' · ');
  } else if (v && (v.status === 'failed')) {
    status.textContent = 'cloning failed — press Clone Voice to try again';
  } else if (v) {
    status.textContent = 'not cloned yet — this voice cannot be used until it is';
  } else {
    status.textContent = '';
  }
}

/* -------------------------------------------------------------- regenerate */
async function regenerate() {
  if (state.isRegenerating) return;
  const v = getActiveVoice();
  const text = (state.scriptText || '').trim();
  if (!v) return;
  if (!text) { showToast('Type some text first.'); return; }
  if (!state.engineLoaded) { showToast('Supertonic 3 is still loading — one moment.'); return; }

  // A voice may only be used once it is fully cloned. Say so here rather
  // than letting the server refuse — the user should never press a button
  // that cannot work.
  const vst = v.status || (v.cloned ? 'ready' : 'new');
  if (vst !== 'ready') {
    if (vst === 'training' && v.clone_job) {
      const mins = Math.max(1, Math.round(v.clone_job.eta_s / 60));
      showToast(`“${v.name}” is still cloning — about ${mins} minute${mins === 1 ? '' : 's'} left. `
        + 'It unlocks automatically when it finishes.', 5000);
    } else if (vst === 'failed') {
      showToast(`Cloning “${v.name}” failed — press Clone Voice to try again.`, 5000);
    } else {
      showToast(`“${v.name}” has not been cloned yet. Press Clone Voice and choose how thorough it should be.`, 5500);
    }
    return;
  }

  stopAllAudio();
  clearResult();
  state.isRegenerating = true;
  state.regenPhase = 'generating';
  renderRegenButton();

  const controller = new AbortController();
  synthAbort = controller;
  const t0 = performance.now();
  const fetchStart = performance.now();

  try {
    const fd = new FormData();
    fd.append('voice_id', v.id);
    fd.append('text', text);
    fd.append('lang', langCode());
    fd.append('steps', String(state.quality));
    fd.append('speed', String(state.speed));
    fd.append('silence', String(state.silence));
    fd.append('ecapa_steps', String(state.ecapaSteps));
    fd.append('streaming', state.streaming ? '1' : '0');
    fd.append('auto_voice', state.autoVoice ? '1' : '0');
    fd.append('natural', state.natural ? '1' : '0');
    fd.append('master', state.master ? '1' : '0');
    fd.append('norm_mode', state.normMode || 'fast');

    const res = await fetch('/api/synthesize', { method: 'POST', body: fd, signal: controller.signal });
    const headersAt = performance.now();
    state.regenPhase = 'generating';
    renderRegenButton();
    if (!res.ok) {
      let detail = 'synthesis failed';
      try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
      throw new Error(detail);
    }

    const autoCloned = res.headers.get('X-AUTO-CLONED') === '1';
    const cloneMs = parseFloat(res.headers.get('X-CLONE-MS') || '0');
    const sr = parseInt(res.headers.get('X-SR') || '44100', 10) || 44100;
    const synthId = res.headers.get('X-SYNTH-ID');
    const estAudio = parseFloat(res.headers.get('X-EST-AUDIO-S') || '0');
    const willStall = res.headers.get('X-WILL-STALL') === '1';
    state.estDuration = estAudio > 0 ? estAudio : null;
    // What the server actually rendered at — in Auto mode this is the step
    // count it picked from its measured history, so surface it on the mode
    // button and the Quality slider instead of leaving them saying "auto".
    const usedSteps = parseInt(res.headers.get('X-STEPS') || '0', 10);
    if (usedSteps > 0 && usedSteps !== state.resultSteps) {
      state.resultSteps = usedSteps;
      renderModes();
      const qs = SLIDERS.find((s) => s.key === 'quality');
      if (qs && qs.el) renderSlider(qs);
    }
    // Time-to-first-byte (headers arriving is a proxy for it, since the
    // server sends headers before the body starts) — minus any auto-clone
    // time, which is CPU work, not network latency, and would otherwise
    // wrongly inflate the buffering threshold below. Used to size how much
    // to buffer before playback starts (see createStreamPlayer).
    const ttfbMs = Math.max(0, (headersAt - fetchStart) - cloneMs);

    // The server's opening bid: estimated total audio, how much of it is
    // speech (pauses cost no inference), and the RTF it expects at this step
    // count. All three are advisory — the player's own live measurement
    // overrides them within ~3 s of audio, which is what makes a wrong
    // server estimate cheap instead of a 10-second wait.
    const p = createStreamPlayer(sr, ttfbMs, {
      estAudioS: estAudio > 0 ? estAudio : 0,
      speechS: parseFloat(res.headers.get('X-SPEECH-S') || '0') || 0,
      rtfHi: parseFloat(res.headers.get('X-RTF-HI') || '0') || 0,
      chunkLatS: parseFloat(res.headers.get('X-CHUNK-LAT-S') || '0') || 0,
    });
    player = p;
    startProgressTicker();
    if (synthId) startStatsPolling(synthId);

    if (autoCloned) {
      showToast(`Voice cloned locally in ${(cloneMs / 1000).toFixed(2)}s — audio generated.`);
      refreshVoices(); // no await
      scheduleVerifiedRefresh();
    }
    if (willStall) {
      showToast('Heads up: at this quality, generation is slower than playback for a '
        + 'script this long — expect a brief pause partway through. Speed/Medium keep up.', 5200);
    }

    // The server streams the WAV while it synthesizes: 44-byte header, then
    // PCM per sentence chunk. Playback starts as soon as a little audio is
    // buffered; the rest keeps arriving behind the playhead.
    const reader = res.body.getReader();
    let received = 0;
    let headerSkipped = 0;
    let carry = null; // odd leftover byte when a network chunk splits a sample

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      let bytes = value;
      if (headerSkipped < 44) { // strip the 44-byte streaming WAV header
        const skip = Math.min(44 - headerSkipped, bytes.length);
        headerSkipped += skip;
        bytes = bytes.subarray(skip);
        if (!bytes.length) continue;
      }
      if (carry) {
        const merged = new Uint8Array(carry.length + bytes.length);
        merged.set(carry, 0);
        merged.set(bytes, carry.length);
        bytes = merged;
        carry = null;
      }
      const usable = bytes.length & ~1;
      if (bytes.length !== usable) carry = bytes.slice(usable);
      if (!usable) continue;
      const f32 = new Float32Array(usable / 2);
      const dv = new DataView(bytes.buffer, bytes.byteOffset, usable);
      for (let i = 0; i < f32.length; i++) f32[i] = dv.getInt16(i * 2, true) / 32768;
      p.push(f32);
      received += usable;
      // Streaming ON: instant start as soon as a little audio is buffered
      // (threshold adapts to measured network latency — see createStreamPlayer).
      // Streaming OFF: keep buffering silently; playback starts when done.
      if (state.streaming && !p.hasStarted() && p.totalSeconds() >= p.startThresholdSeconds()) {
        p.start();
        state.resultPlaying = true;
        renderPlayer();
      }
    }

    if (controller.signal.aborted) return;
    p.finishStream();
    state.estDuration = null; // exact length known now
    if (received > 0 && !p.hasStarted()) { // streaming OFF (or tiny result)
      p.start();
      state.resultPlaying = true;
      renderPlayer();
    }

    // Final RTF — the server measures pure inference time per chunk; fall
    // back to wall-clock if the stats lookup fails.
    let rtf = null;
    if (synthId) {
      try {
        const sres = await fetch(`/api/synthesize/stats/${synthId}`);
        if (sres.ok) {
          const st = await sres.json();
          if (st && st.done) {
            rtf = st.rtf;
            if (st.error) showToast(`Synthesis ended early: ${st.error}`);
            // Which voices actually played which paragraph — only known once
            // the stream finishes (headers go out before generation starts).
            if (state.autoVoice && Array.isArray(st.voices_used) && st.voices_used.length > 1) {
              showToast(`Auto Voice: switched between ${st.voices_used.join(', ')} by paragraph language.`, 4200);
            }
          }
        }
      } catch (e) { /* fall through to wall-clock */ }
    }
    if (rtf == null) {
      const audioSec = received / 2 / sr;
      if (audioSec > 0) rtf = ((performance.now() - t0) / 1000 - cloneMs / 1000) / audioSec;
    }
    if (rtf != null && isFinite(rtf)) state.resultRTF = rtf;
  } catch (err) {
    if (!(err && err.name === 'AbortError')) showToast(String(err.message || err));
  } finally {
    // Close the stream on EVERY exit path, not just the clean one. A dropped
    // connection (server killed, Wi-Fi gone on the phone-over-LAN use case)
    // used to skip finishStream() entirely, which left the transport stuck on
    // Pause, replay impossible and — worst — the Download button hidden, so
    // the perfectly good audio already in the buffer could not be saved. The
    // only way out was Regenerate, which destroys it.
    if (player) player.finishStream();
    state.estDuration = null;
    if (synthAbort === controller) synthAbort = null;
    state.isRegenerating = false;
    stopStatsPolling();
    state.liveStats = null;
    renderRegenButton();
    renderPlayer();
    renderCloneButton();
    refreshRtfHistory(); // updates the measured-RTF labels on the mode buttons
  }
}

async function refreshRtfHistory() {
  try {
    const res = await fetch('/api/status');
    const s = await res.json();
    if (s.rtf_history) {
      state.rtfHistory = s.rtf_history;
      renderModes();
    }
  } catch (e) { /* purely cosmetic — labels refresh next time */ }
}

function toggleResultPlay() {
  if (!player) { showToast('Hit Regenerate first to synthesize this script.'); return; }
  if (state.resultPlaying) {
    player.pause();
    state.resultPlaying = false;
    renderPlayer();
    return;
  }
  if (sourceAudio) { sourceAudio.pause(); sourceAudio = null; state.sourcePlaying = false; renderSourceButton(); }
  if (!player.hasStarted()) player.start();      // user starts before the autoplay gate
  else if (player.isEnded()) player.restart();   // replay from the top
  else player.resume();
  state.resultPlaying = true;
  renderPlayer();
}

/* ------------------------------------------------------------------ clone */
async function cloneVoice() {
  const v = getActiveVoice();
  if (!v || state.isCloning) return;
  if (v.type === 'builtin') {
    showToast('This is a native Supertonic voice — nothing to clone.');
    return;
  }
  if (v.type === 'imported') {
    showToast('This is an imported 1:1 trained voice — nothing to re-clone.');
    return;
  }
  if (!state.engineLoaded) { showToast('Supertonic 3 is still loading — one moment.'); return; }

  // Already running? Then this button is a Cancel.
  if (v.clone_job && ['starting', 'smart-init', 'searching', 'verifying'].includes(v.clone_job.phase)) {
    await fetch(`/api/voices/${v.id}/clone/cancel`, { method: 'POST' }).catch(() => {});
    showToast('Cloning cancelled — the voice was left as it was.');
    await refreshVoices();
    renderCloneButton();
    return;
  }

  const depth = await pickCloneDepth();
  if (!depth) return;

  state.isCloning = true;
  renderCloneButton();
  try {
    const fd = new FormData();
    fd.append('ecapa_steps', String(state.ecapaSteps));
    fd.append('depth', depth);
    const res = await fetch(`/api/voices/${v.id}/clone`, { method: 'POST', body: fd });
    if (!res.ok) {
      let detail = 'clone failed';
      try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
      throw new Error(detail);
    }
    const mins = Math.round((CLONE_DEPTHS.find((d) => d.key === depth) || {}).seconds / 60);
    showToast(`Cloning “${v.name}” — about ${mins} minute${mins === 1 ? '' : 's'}. `
      + 'It runs entirely on this machine, and the voice unlocks the moment it is done. '
      + 'You can keep using other voices meanwhile.', 6000);
    await refreshVoices();
    startClonePolling();
  } catch (err) {
    showToast(String(err.message || err));
  } finally {
    state.isCloning = false;
    renderCloneButton();
    renderVoices();
  }
}

// The depth choice, asked plainly rather than hidden in a settings panel:
// the whole point is that the user decides how long this takes.
let CLONE_DEPTHS = [
  { key: 'quick', label: 'Quick', seconds: 120, description: 'about two minutes' },
  { key: 'standard', label: 'Standard', seconds: 900, description: 'about fifteen minutes' },
  { key: 'deep', label: 'Deep', seconds: 3600, description: 'an hour' },
];

function pickCloneDepth() {
  return new Promise((resolve) => {
    const back = document.createElement('div');
    back.style.cssText = 'position: fixed; inset: 0; background: rgba(0,0,0,0.72); display: flex; align-items: center; justify-content: center; z-index: 90;';
    const card = document.createElement('div');
    card.style.cssText = 'background: #121212; border: 1px solid rgba(255,255,255,0.14); border-radius: 16px; padding: 26px 28px; max-width: 460px; color: #e8e8e2; font-size: 14px;';
    card.innerHTML = '<div style="font-size:17px;margin-bottom:8px;">How thorough should this clone be?</div>'
      + '<div style="color:#9a9a94;line-height:1.6;margin-bottom:18px;">The clone searches for the voice style whose synthesized speech best matches your recording, and checks the result on a sentence it never trained on. Longer means more candidates tried — nothing is uploaded, and you can cancel any time.</div>';
    for (const d of CLONE_DEPTHS) {
      const b = document.createElement('button');
      b.className = 'hov-border35';
      b.style.cssText = 'display:block;width:100%;text-align:left;margin-bottom:8px;padding:11px 14px;border-radius:10px;border:1px solid rgba(255,255,255,0.16);background:#0e0e0e;color:#f0f0ec;font-size:14px;font-family:inherit;cursor:pointer;';
      b.innerHTML = `<b style="font-weight:500;">${d.label}</b> <span style="color:#8a8a84;">— ${d.description}</span>`;
      b.addEventListener('click', () => { document.body.removeChild(back); resolve(d.key); });
      card.appendChild(b);
    }
    const cancel = document.createElement('button');
    cancel.className = 'hov-border35';
    cancel.style.cssText = 'margin-top:6px;padding:9px 14px;border-radius:10px;border:1px solid rgba(255,255,255,0.12);background:transparent;color:#8a8a84;font-size:13px;font-family:inherit;cursor:pointer;';
    cancel.textContent = 'Not now';
    cancel.addEventListener('click', () => { document.body.removeChild(back); resolve(null); });
    card.appendChild(cancel);
    back.appendChild(card);
    back.addEventListener('click', (e) => {
      if (e.target === back) { document.body.removeChild(back); resolve(null); }
    });
    document.body.appendChild(back);
  });
}

// While any clone runs, refresh often enough that the numbers visibly move.
let clonePollTimer = null;
function startClonePolling() {
  if (clonePollTimer) return;
  clonePollTimer = setInterval(async () => {
    const before = state.voices.filter((v) => v.status === 'training').map((v) => v.id);
    await refreshVoices();
    renderCloneButton();
    const stillTraining = state.voices.filter((v) => v.status === 'training');
    for (const id of before) {
      if (!stillTraining.some((v) => v.id === id)) {
        const done = state.voices.find((v) => v.id === id);
        if (done && done.status === 'ready') {
          const ci = done.clone_info || {};
          showToast(`“${done.name}” is ready — ${Number(ci.verified_sim_pct || 0).toFixed(1)}% match`
            + (ci.improved_points != null ? `, ${ci.improved_points >= 0 ? '+' : ''}${Number(ci.improved_points).toFixed(1)} points over the instant clone` : '')
            + '. You can use it now.', 7000);
        } else if (done && done.status === 'failed') {
          showToast(`Cloning “${done.name}” failed. Press Clone Voice to try again.`, 6000);
        }
      }
    }
    if (!stillTraining.length) { clearInterval(clonePollTimer); clonePollTimer = null; }
  }, 2000);
}

let verifiedRefreshTimer = null;
function scheduleVerifiedRefresh() {
  // The backend verifies achieved similarity asynchronously (one synthesis +
  // one embed); pick it up shortly after so pill tooltips carry real numbers.
  if (verifiedRefreshTimer) clearTimeout(verifiedRefreshTimer);
  verifiedRefreshTimer = setTimeout(async () => {
    await refreshVoices();
  }, 6000);
}

/* --------------------------------------------------------------- uploads */
async function uploadVoiceFile(file, vtype) {
  const fd = new FormData();
  fd.append('file', file, file.name);
  fd.append('vtype', vtype);
  const res = await fetch('/api/voices/upload', { method: 'POST', body: fd });
  if (!res.ok) {
    let detail = 'upload failed';
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

async function onFilePicked(e) {
  const file = e.target.files && e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    let data;
    if (/\.json$/i.test(file.name)) {
      // A ready-made Supertonic voice-style JSON (Colab paid_parity clone,
      // Voice Builder purchase, LoudFlow clone) — imported 1:1, no blending.
      const fd = new FormData();
      fd.append('style', file, file.name);
      const res = await fetch('/api/voices/import_style', { method: 'POST', body: fd });
      if (!res.ok) {
        let detail = 'import failed';
        try { detail = (await res.json()).detail || detail; } catch (e2) { /* not json */ }
        throw new Error(detail);
      }
      data = await res.json();
      showToast('1:1 voice style imported — this tab now speaks with exactly this trained voice.', 4200);
    } else {
      data = await uploadVoiceFile(file, 'uploaded');
      if (data.warning) showToast(`Voice added — note: ${data.warning}`, 4500);
    }
    await refreshVoices();
    selectVoice(data.voice.id);
  } catch (err) {
    showToast(String(err.message || err), 4500);
  }
}

/* -------------------------------------------------------------- recording */
async function startRecording() {
  if (state.recording) return;
  try {
    recordStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showToast('Microphone access denied — allow it to record a reference.');
    return;
  }
  recordChunks = [];
  mediaRecorder = new MediaRecorder(recordStream);
  mediaRecorder.addEventListener('dataavailable', (e) => { if (e.data.size) recordChunks.push(e.data); });
  mediaRecorder.addEventListener('stop', finishRecording);
  mediaRecorder.start();
  state.recording = true;
  state.recordSeconds = 0;
  recordTimer = setInterval(() => {
    state.recordSeconds += 1;
    if (state.recordSeconds >= 45) { stopRecording(); return; }
    renderVoices();
  }, 1000);
  renderVoices();
}

function stopRecording() {
  if (!state.recording) return;
  if (recordTimer) { clearInterval(recordTimer); recordTimer = null; }
  state.recording = false;
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (recordStream) { recordStream.getTracks().forEach((t) => t.stop()); recordStream = null; }
  renderVoices();
}

async function finishRecording() {
  const blob = new Blob(recordChunks, { type: mediaRecorder ? mediaRecorder.mimeType : 'audio/webm' });
  recordChunks = [];
  if (!blob.size) { showToast('Recording was empty.'); return; }
  try {
    const wavBlob = await blobToWav(blob);
    const file = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
    const data = await uploadVoiceFile(file, 'recorded');
    await refreshVoices();
    selectVoice(data.voice.id);
    if (data.warning) showToast(`Voice recorded — note: ${data.warning}`, 4500);
  } catch (err) {
    showToast(String(err.message || err), 4500);
  }
}

async function blobToWav(blob) {
  // Decode whatever MediaRecorder produced (webm/opus in Chrome) with the
  // browser's own decoder, then re-encode to mono 16-bit WAV so the backend
  // needs no ffmpeg.
  const arr = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(arr);
  const ch = buf.numberOfChannels > 1
    ? mixToMono(buf)
    : buf.getChannelData(0);
  const sr = buf.sampleRate;
  ctx.close();

  const out = new ArrayBuffer(44 + ch.length * 2);
  const view = new DataView(out);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + ch.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, ch.length * 2, true);
  for (let i = 0; i < ch.length; i++) {
    const s = Math.max(-1, Math.min(1, ch[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([out], { type: 'audio/wav' });
}

function mixToMono(buf) {
  const len = buf.length;
  const mono = new Float32Array(len);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const data = buf.getChannelData(c);
    for (let i = 0; i < len; i++) mono[i] += data[i] / buf.numberOfChannels;
  }
  return mono;
}

/* ------------------------------------------------------------------- sync */
async function refreshVoices() {
  try {
    const res = await fetch('/api/voices');
    const data = await res.json();
    const prevActive = state.activeVoiceId;
    state.voices = data.voices;
    if (!state.voices.find((v) => v.id === prevActive)) {
      state.activeVoiceId = state.voices.length ? state.voices[0].id : null;
    }
    renderVoices();
    renderTitle();
    renderSourceButton();
    renderCloneButton();
  } catch (e) { /* backend not up yet */ }
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const s = await res.json();
    state.engineLoaded = !!s.loaded;
    state.deviceLabel = s.device || '';
    state.engineMessage = s.message || '';
    if (Array.isArray(s.languages) && s.languages.length) LANGUAGES = s.languages;
    if (Array.isArray(s.modes) && s.modes.length) MODES = s.modes;
    if (s.rtf_history) state.rtfHistory = s.rtf_history;
    renderModes();
    if (s.default_ecapa_steps && !pollStatus._stepsInit) {
      state.ecapaSteps = s.default_ecapa_steps;
      state.maxEcapaSteps = s.max_ecapa_steps || 32;
      pollStatus._stepsInit = true;
      buildSliders();
    }
    const el = $('engine-status');
    if (s.loaded) {
      el.textContent = state.deviceLabel;
      el.style.animation = 'none';
    } else {
      el.textContent = state.engineMessage;
      el.style.animation = 'stbPulse 1.6s ease-in-out infinite';
      setTimeout(pollStatus, 1500);
    }
  } catch (e) {
    setTimeout(pollStatus, 1500);
  }
}

/* ------------------------------------------------------------------- init */
function init() {
  const prefs = loadPrefs();
  if (typeof prefs.quality === 'number' && prefs.quality >= 0) state.quality = prefs.quality;
  if (typeof prefs.streaming === 'boolean') state.streaming = prefs.streaming;
  if (typeof prefs.autoVoice === 'boolean') state.autoVoice = prefs.autoVoice;
  if (typeof prefs.natural === 'boolean') state.natural = prefs.natural;
  if (typeof prefs.master === 'boolean') state.master = prefs.master;
  if (prefs.normMode === 'fast' || prefs.normMode === 'quality') state.normMode = prefs.normMode;

  $('script').value = state.scriptText;
  $('script').addEventListener('input', (e) => {
    state.scriptText = e.target.value;
    renderCharCount();
  });

  $('lang-toggle').addEventListener('click', (e) => {
    e.stopPropagation();
    state.languageOpen = !state.languageOpen;
    renderLanguageMenu();
  });
  document.addEventListener('click', (e) => {
    if (state.languageOpen && !$('lang-menu').contains(e.target)) {
      state.languageOpen = false;
      renderLanguageMenu();
    }
  });

  $('source-btn').addEventListener('click', toggleSourcePlay);
  $('regen-btn').addEventListener('click', () => {
    if (state.isRegenerating) stopGeneration();
    else regenerate();
  });
  $('result-play-btn').addEventListener('click', toggleResultPlay);
  $('download-btn').addEventListener('click', downloadResult);
  $('clone-btn').addEventListener('click', cloneVoice);
  $('file-input').addEventListener('change', onFilePicked);
  $('stream-toggle').addEventListener('click', () => {
    state.streaming = !state.streaming;
    savePrefs();
    renderStreamToggle();
  });
  $('auto-voice-toggle').addEventListener('click', () => {
    state.autoVoice = !state.autoVoice;
    savePrefs();
    renderAutoVoiceToggle();
  });
  $('natural-toggle').addEventListener('click', () => {
    state.natural = !state.natural;
    savePrefs();
    renderNaturalToggle();
  });
  $('master-toggle').addEventListener('click', () => {
    state.master = !state.master;
    savePrefs();
    renderMasterToggle();
  });
  $('norm-toggle').addEventListener('click', () => {
    state.normMode = state.normMode === 'quality' ? 'fast' : 'quality';
    savePrefs();
    renderNormToggle();
  });

  buildSliders();
  renderModes();
  renderStreamToggle();
  renderAutoVoiceToggle();
  renderNaturalToggle();
  renderMasterToggle();
  renderNormToggle();
  renderContentTypes();
  renderLanguageMenu();
  renderCharCount();
  renderRegenButton();
  renderPlayer();
  renderVoices();

  refreshVoices();
  pollStatus();
}

init();
