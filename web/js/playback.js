'use strict';

// Phát lại đoạn audio từng từ (▶) + phát mẫu TTS (🔊). Listener delegated gắn 1 lần.

// ── Phát lại đoạn audio của 1 TỪ (nút ▶ ở Pronunciation detail) ────────
// 1 thẻ <audio> ẩn dùng chung + generation token: mỗi lần bấm tăng token, huỷ timer
// dừng cũ, pause→seek→play đoạn mới, rồi đặt setTimeout dừng (guard bằng token để
// timer cũ thành no-op). Dừng bằng setTimeout chứ KHÔNG timeupdate: timeupdate chỉ
// ~4 lần/giây nên với đoạn từ 300–700ms dễ phát lố; audio từ Blob cục bộ không
// buffering nên setTimeout theo độ dài cố định ổn định + chính xác trên mọi trình duyệt.
// `start/end` từ backend ĐÃ đệm + clamp theo từ liền kề (xem _pad_and_clamp_windows) →
// FE phát VERBATIM, KHÔNG tự đệm (đệm là việc của backend vì chỉ nó biết ranh giới từ kề).
let wordAudio = null;
let wordPlayToken = 0;
let wordStopTimer = null;
let wordGainNode = null;
let wordAudioCtx = null;

// Bản ghi của người học thường nhỏ hơn hẳn audio mẫu TTS (🔊) → khuếch đại bằng
// Web Audio API GainNode (thẻ <audio>.volume bị giới hạn tối đa 1.0, không đủ to).
const WORD_SEGMENT_GAIN = 1;

function ensureWordGain() {
    if (wordGainNode) return;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return; // trình duyệt không hỗ trợ → phát bình thường, không khuếch đại
    wordAudioCtx = new Ctx();
    const source = wordAudioCtx.createMediaElementSource(wordAudio);
    wordGainNode = wordAudioCtx.createGain();
    wordGainNode.gain.value = WORD_SEGMENT_GAIN;
    source.connect(wordGainNode).connect(wordAudioCtx.destination);
}

function playWordSegment(start, end, srcUrl) {
    // srcUrl: audio câu cụ thể (kết quả cả đề — mỗi câu một Blob). Bỏ trống → dùng
    // Blob single global (playbackUrl) như cũ.
    const url = srcUrl || playbackUrl();
    if (!url) return;
    if (!wordAudio) { wordAudio = new Audio(); ensureWordGain(); }
    if (wordAudioCtx && wordAudioCtx.state === 'suspended') wordAudioCtx.resume();
    const myToken = ++wordPlayToken;
    if (wordStopTimer) { clearTimeout(wordStopTimer); wordStopTimer = null; }
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
    // currentTime chỉ set được khi metadata đã sẵn sàng (file mới load lần đầu).
    if (wordAudio.readyState >= 1) begin();
    else wordAudio.addEventListener('loadedmetadata', begin, { once: true });
}

// Delegated: bắt mọi nút .phoneme-play (panel dựng lại mỗi lần render), gắn 1 lần.
document.addEventListener('click', e => {
    const btn = e.target instanceof Element ? e.target.closest('.phoneme-play') : null;
    if (!btn) return;
    e.preventDefault();
    const start = parseFloat(btn.dataset.start);
    const end = parseFloat(btn.dataset.end);
    if (Number.isFinite(start) && Number.isFinite(end)) playWordSegment(start, end, btn.dataset.src || null);
});

// ── Phát "phát âm đúng" của 1 TỪ (nút 🔊) ──────────────────────────────
// Khác playWordSegment (phát lại Blob của chính người học): đây là audio THAM CHIẾU
// do server tổng hợp (Piper TTS) qua GET /tts. Phát CẢ file nên không cần token/seek/
// setTimeout — chỉ pause→đổi src→play. Giọng theo `currentAccent` (default→US ở backend).
// 1 thẻ <audio> ẩn dùng chung (tách khỏi wordAudio để hai nút không cắt nhau).
let ttsAudio = null;

// Phải khớp CACHE_VERSION ở src/tts.py. URL /tts được server cache 24h (max-age),
// nên bump version chỉ ở đĩa server KHÔNG đủ — trình duyệt vẫn phát WAV cũ theo URL cũ.
// Đưa version vào URL → đổi audio (vd fix mất /s/) là đổi cache-key trình duyệt luôn.
const TTS_AUDIO_VERSION = 'v4';

function playWordTts(word) {
    if (!word) return;
    const url = `${apiBase()}/tts?text=${encodeURIComponent(word)}&accent=${encodeURIComponent(currentAccent)}&v=${TTS_AUDIO_VERSION}`;
    if (!ttsAudio) ttsAudio = new Audio();
    ttsAudio.pause();
    ttsAudio.src = url;
    const p = ttsAudio.play();
    if (p && typeof p.catch === 'function') {
        p.catch(err => {
            // Bấm từ khác khi từ trước còn đang tải → load bị ngắt (AbortError) hoặc
            // play() bị pause() chen ngang: KHÔNG phải lỗi server → bỏ qua, không báo.
            if (err && (err.name === 'AbortError' || err.name === 'NotAllowedError')) return;
            // Còn lại (NotSupportedError = src tải lỗi: 503 chưa cài voice / lỗi mạng) → báo nhẹ.
            alert('Chưa phát được audio mẫu — server có thể chưa cài voice TTS (xem README: TTS_VOICE_US/GB).');
        });
    }
}

// Delegated: bắt mọi nút .tts-play (panel dựng lại mỗi lần render), gắn 1 lần.
document.addEventListener('click', e => {
    const btn = e.target instanceof Element ? e.target.closest('.tts-play') : null;
    if (!btn) return;
    e.preventDefault();
    playWordTts(btn.dataset.word || '');
});
