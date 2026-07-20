// @ts-nocheck
// Port web/js/playback.js — phát lại đoạn audio từng từ (▶ .phoneme-play) + phát mẫu
// TTS (🔊 .tts-play). Listener DELEGATED gắn 1 lần ở document → hoạt động với DOM do
// React inject (dangerouslySetInnerHTML). Kích hoạt các nút ▶/🔊 ở cả exam (M1) lẫn
// grading (M2). currentAccent + playbackUrl là module-level, sync từ React.

import { apiBase } from '../../lib/api';
import { hasHangul } from '../../lib/config';

let currentAccent = 'default';
export function setPlaybackAccent(a) {
  currentAccent = a || 'default';
}
// URL Blob của file single đang xem (nút ▶ dùng khi không có data-src riêng).
let _playbackUrl: string | null = null;
export function setPlaybackUrl(url) {
  _playbackUrl = url || null;
}
function playbackUrl() {
  return _playbackUrl;
}

// ── Phát lại đoạn audio của 1 TỪ (nút ▶) ──
let wordAudio = null;
let wordPlayToken = 0;
let wordStopTimer = null;
let wordGainNode = null;
let wordAudioCtx = null;
const WORD_SEGMENT_GAIN = 1;

function ensureWordGain() {
  if (wordGainNode) return;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  wordAudioCtx = new Ctx();
  const source = wordAudioCtx.createMediaElementSource(wordAudio);
  wordGainNode = wordAudioCtx.createGain();
  wordGainNode.gain.value = WORD_SEGMENT_GAIN;
  source.connect(wordGainNode).connect(wordAudioCtx.destination);
}

function playWordSegment(start, end, srcUrl) {
  const url = srcUrl || playbackUrl();
  if (!url) return;
  if (!wordAudio) {
    wordAudio = new Audio();
    ensureWordGain();
  }
  if (wordAudioCtx && wordAudioCtx.state === 'suspended') wordAudioCtx.resume();
  const myToken = ++wordPlayToken;
  if (wordStopTimer) {
    clearTimeout(wordStopTimer);
    wordStopTimer = null;
  }
  wordAudio.pause();
  if (wordAudio.src !== url) wordAudio.src = url;
  const from = Math.max(0, start);
  const stopMs = Math.max(0, end - start) * 1000;
  const begin = () => {
    wordAudio.currentTime = from;
    const p = wordAudio.play();
    if (p && typeof p.catch === 'function') p.catch(() => {});
    wordStopTimer = setTimeout(() => {
      if (myToken === wordPlayToken) wordAudio.pause();
    }, stopMs);
  };
  if (wordAudio.readyState >= 1) begin();
  else wordAudio.addEventListener('loadedmetadata', begin, { once: true });
}

// ── Phát "phát âm đúng" của 1 TỪ (nút 🔊 — Piper TTS qua /tts) ──
let ttsAudio = null;
const TTS_AUDIO_VERSION = 'v6'; // phải khớp CACHE_VERSION ở src/tts.py

// `ipa` (tuỳ chọn): chuỗi IPA tham chiếu ĐANG hiển thị của từ. Server đọc đúng chuỗi
// này khi bật TTS_IPA_SYNTH (khớp homograph/viết tắt/dạng trích dẫn) và tự fallback
// về `text` nếu tắt hoặc IPA không map được — nên luôn gửi kèm cả hai.
// `accent` (tuỳ chọn): ép giọng đọc cụ thể ('us' | 'gb') — dùng khi 1 chỗ hiện CẢ hai
// phiên âm UK và US, mỗi nút 🔊 đọc đúng giọng của nó. Rỗng → theo accent đang chọn.
function playWordTts(word, ipa = '', accent = '') {
  if (!word) return;
  if (hasHangul(word)) {
    if (!window.speechSynthesis) {
      alert('Trình duyệt không hỗ trợ đọc văn bản (Web Speech API) — chưa phát được audio mẫu tiếng Hàn.');
      return;
    }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(word);
    u.lang = 'ko-KR';
    u.rate = 0.85;
    window.speechSynthesis.speak(u);
    return;
  }
  const acc = accent || currentAccent;
  const ipaParam = ipa ? `&ipa=${encodeURIComponent(ipa)}` : '';
  const url = `${apiBase()}/tts?text=${encodeURIComponent(word)}${ipaParam}&accent=${encodeURIComponent(acc)}&v=${TTS_AUDIO_VERSION}`;
  if (!ttsAudio) ttsAudio = new Audio();
  ttsAudio.pause();
  ttsAudio.src = url;
  const p = ttsAudio.play();
  if (p && typeof p.catch === 'function') {
    p.catch((err) => {
      if (err && (err.name === 'AbortError' || err.name === 'NotAllowedError')) return;
      alert('Chưa phát được audio mẫu — server có thể chưa cài voice TTS (xem README: TTS_VOICE_US/GB).');
    });
  }
}

// ── Cài delegated listener 1 lần ──
let _installed = false;
export function installPlaybackHandlers() {
  if (_installed) return;
  _installed = true;
  document.addEventListener('click', (e) => {
    const t = e.target instanceof Element ? e.target : null;
    if (!t) return;
    const playBtn = t.closest('.phoneme-play');
    if (playBtn) {
      e.preventDefault();
      const start = parseFloat(playBtn.dataset.start);
      const end = parseFloat(playBtn.dataset.end);
      if (Number.isFinite(start) && Number.isFinite(end)) playWordSegment(start, end, playBtn.dataset.src || null);
      return;
    }
    const ttsBtn = t.closest('.tts-play');
    if (ttsBtn) {
      e.preventDefault();
      playWordTts(ttsBtn.dataset.word || '', ttsBtn.dataset.ipa || '', ttsBtn.dataset.accent || '');
    }
  });
}
