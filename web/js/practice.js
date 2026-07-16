'use strict';

// Popup luyện từ (ELSA-style): click từ trong Pronunciation detail → popup hiện
// từ + IPA + % + bảng phoneme (đúng/sai + tip tiếng Việt + 🔊) + định nghĩa/ví dụ
// (LLM, cache server) + mic ghi âm chấm lại RIÊNG từ đó qua /grade
// (text=<từ>, mode=mock_test, no_ai=true — không gọi LLM, không lưu history).
//
// Modal duy nhất tự dựng + append <body> lần đầu mở → dùng được từ mọi tab
// (single/batch/exam/history đều render qua phonemeErrorsHtml của render.js).

const practiceState = {
    data: null,          // {word, ipa, accuracy, skip_reason, phonemes} từ data-practice
    recorder: null,
    stream: null,
    chunks: [],
    recording: false,
    grading: false,
    playUrl: null,       // object URL của lần ghi âm gần nhất → nút "Nghe lại bạn vừa nói"
    playAudio: null,     // Audio element tái dùng (tránh chồng tiếng khi bấm liên tiếp)
};

// Cache word-info trong phiên (server cũng cache SQLite — đây chỉ đỡ round-trip).
const wordInfoCache = new Map();

// ── Modal skeleton (dựng 1 lần) ─────────────────────────────────────────
function ensurePracticeModal() {
    let overlay = document.getElementById('practice-modal');
    if (overlay) return overlay;
    overlay = document.createElement('div');
    overlay.id = 'practice-modal';
    overlay.className = 'practice-overlay hidden';
    overlay.innerHTML = `
        <div class="practice-modal" role="dialog" aria-modal="true" aria-label="Luyện tập từ vựng">
            <div class="practice-head">
                <h3 class="practice-head__word" id="practice-word"></h3>
                <button type="button" class="practice-bookmark" id="practice-bookmark" title="Lưu từ để luyện tập">☆</button>
                <span class="practice-head__spacer"></span>
                <div class="practice-ring" id="practice-ring"><div class="practice-ring__inner" id="practice-pct">–</div></div>
                <button type="button" class="practice-close" id="practice-close" title="Đóng" aria-label="Đóng">✕</button>
            </div>
            <div class="practice-ipa" id="practice-ipa"></div>
            <div class="practice-info" id="practice-info"></div>
            <div id="practice-phonemes"></div>
            <div class="practice-rec">
                <div class="practice-rec__hint">Chạm để nói — chấm lại riêng từ này</div>
                <button type="button" class="practice-mic" id="practice-mic" title="Ghi âm luyện tập">🎙️</button>
                <div class="practice-status" id="practice-status"></div>
                <button type="button" class="practice-replay hidden" id="practice-replay">▶ Nghe lại bạn vừa nói</button>
            </div>
        </div>`;
    document.body.appendChild(overlay);
    overlay.addEventListener('click', e => { if (e.target === overlay) closePracticePopup(); });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && !overlay.classList.contains('hidden')) closePracticePopup();
    });
    overlay.querySelector('#practice-close').addEventListener('click', closePracticePopup);
    overlay.querySelector('#practice-mic').addEventListener('click', togglePracticeRecording);
    overlay.querySelector('#practice-bookmark').addEventListener('click', togglePracticeBookmark);
    overlay.querySelector('#practice-replay').addEventListener('click', playPracticeReplay);
    return overlay;
}

// ── % chính xác hiển thị: (ok + low) / non-skipped — khớp ngưỡng isSignificant
//    của render.js (severity 'low' = recognizer noise, không tính lỗi với người học).
function practicePct(phonemes) {
    const scored = (phonemes || []).filter(p => p.status !== 'skipped');
    if (!scored.length) return null;
    const pass = scored.filter(p => p.status === 'ok' || p.severity === 'low').length;
    return Math.round(100 * pass / scored.length);
}

function practiceRingUpdate(pct) {
    const ring = document.getElementById('practice-ring');
    const label = document.getElementById('practice-pct');
    if (pct == null) { label.textContent = '–'; ring.style.setProperty('--pct', 0); return; }
    const color = pct >= 80 ? '#16a34a' : pct >= 50 ? '#f59e0b' : '#dc2626';
    ring.style.setProperty('--pct', pct);
    ring.style.setProperty('--ring-color', color);
    label.textContent = `${pct}%`;
}

// ── Render phoneme: dải chip màu + thẻ chi tiết CHỈ cho âm sai ──────────
// (thay bảng "Âm thanh | Bạn đã nói" cũ — từ đọc đúng hết không còn ra 13 hàng
// "Chính xác" lặp lại; âm đúng = chip xanh, sai = chip cam + thẻ tip, xám = chưa chấm).
function practiceIsBad(p) {
    return p.status !== 'skipped' && !(p.status === 'ok' || p.severity === 'low');
}

function practicePhonemesHtml(phonemes, isKo) {
    // GB chỉ đổi HIỂN THỊ (data-practice giữ symbol gốc US) — tái dùng transform
    // của render.js qua object từ giả. Từ tiếng Hàn: không có giọng Anh/Mỹ → bỏ qua.
    const disp = (currentAccent === 'gb' && !isKo && typeof toBritishWord === 'function')
        ? toBritishWord({ phonemes }).phonemes : phonemes;
    const shown = (disp || []).filter(p => !p._hidden);
    if (!shown.length) return '';

    // Chip có từ ví dụ → kiêm luôn nút nghe (delegated .tts-play của playback.js).
    const chips = shown.map(p => {
        const cls = p.status === 'skipped' ? 'skip' : practiceIsBad(p) ? 'bad' : 'ok';
        const info = phonemeTip(p.symbol);
        const sym = `/${escapeHtml(p.symbol)}/`;
        return info && info.example
            ? `<button type="button" class="practice-chip ${cls} tts-play" data-word="${escapeHtml(info.example)}" title="Nghe âm này trong từ “${escapeHtml(info.example)}”">${sym}</button>`
            : `<span class="practice-chip ${cls}">${sym}</span>`;
    }).join('');

    const bad = shown.filter(practiceIsBad);
    const skipped = shown.filter(p => p.status === 'skipped').length;
    let detail = '';
    if (bad.length) {
        const cards = bad.map(p => {
            const info = phonemeTip(p.symbol);
            const tts = info && info.example
                ? `<button type="button" class="tts-play practice-fix__play" data-word="${escapeHtml(info.example)}" title="Nghe âm này trong từ “${escapeHtml(info.example)}”">🔊 ${escapeHtml(info.example)}</button>`
                : '';
            const heard = p.status === 'del' ? '∅ thiếu âm' : `/${p.heard ?? '?'}/`;
            const tip = info ? info.tip
                : 'Nghe mẫu và bắt chước khẩu hình — chú ý vị trí lưỡi và môi.';
            return `<div class="practice-fix">
                <div class="practice-fix__row">
                    <span class="practice-fix__target">/${escapeHtml(p.symbol)}/</span>
                    <span class="practice-fix__label">bạn nói</span>
                    <span class="practice-fix__heard">${escapeHtml(heard)}</span>
                    ${tts}
                </div>
                <div class="practice-tip">${escapeHtml(tip)}</div>
            </div>`;
        }).join('');
        detail = `<div class="practice-fix-list__title">Cần cải thiện</div>${cards}`;
    } else {
        detail = '<div class="practice-allok">🎉 Tất cả các âm đều chính xác!</div>';
    }
    const skipNote = skipped > 0
        ? '<div class="practice-skipnote">Âm màu xám chưa chấm được ở lần nói này — bấm mic thử lại.</div>' : '';
    return `<div class="practice-chips">${chips}</div>${detail}${skipNote}`;
}

// IPA hiển thị của popup: dựng TỪ phonemes (kèm trọng âm display_stress, đã áp GB
// như dải chip) để TRÙNG với Pronunciation detail. Không có phonemes (vd gợi ý luyện
// từ chưa chấm) → fallback chuỗi d.ipa từ backend (nay cũng đã kèm nhấn âm).
function practiceIpaString(d) {
    const isKo = typeof hasHangul === 'function' && hasHangul(d.word);
    const disp = (currentAccent === 'gb' && !isKo && typeof toBritishWord === 'function')
        ? toBritishWord({ phonemes: d.phonemes || [] }).phonemes : (d.phonemes || []);
    return (typeof ipaStressString === 'function' ? ipaStressString(disp) : '') || (d.ipa || '');
}

function renderPracticeBody() {
    const d = practiceState.data;
    const isKo = typeof hasHangul === 'function' && hasHangul(d.word);
    document.getElementById('practice-word').textContent = d.word;
    const ipaStr = practiceIpaString(d);
    const ipaDisp = ipaStr ? `/${ipaStr}/` : '';
    document.getElementById('practice-ipa').innerHTML = `${escapeHtml(ipaDisp)}
        <button type="button" class="tts-play" data-word="${escapeHtml(d.word)}" title="Nghe phát âm chuẩn">🔊</button>`;
    document.getElementById('practice-phonemes').innerHTML = practicePhonemesHtml(d.phonemes, isKo);
    // Từ Hangul: ẩn ☆ (server /words chỉ nhận từ Latin — xem render.js bookmarkBtn).
    const bookmark = document.getElementById('practice-bookmark');
    if (bookmark) bookmark.classList.toggle('hidden', isKo);
    practiceRingUpdate(practicePct(d.phonemes));
    const status = document.getElementById('practice-status');
    status.className = 'practice-status';
    status.textContent = d.skip_reason
        ? 'Lần chấm trước chưa nghe rõ từ này — bấm mic để luyện và chấm lại.' : '';
    updatePracticeBookmarkStar();
}

// ── Định nghĩa + ví dụ (lazy, không chặn luyện tập khi lỗi) ─────────────
async function loadPracticeWordInfo(word) {
    const box = document.getElementById('practice-info');
    // Từ tiếng Hàn: /word-info hiện là từ điển tiếng ANH (validate + prompt EN)
    // → bỏ qua, không gọi cho đỡ 400 noise. Từ điển Hàn là backlog riêng (D7/M5).
    if (typeof hasHangul === 'function' && hasHangul(word)) { box.innerHTML = ''; return; }
    const key = word.toLowerCase();
    if (wordInfoCache.has(key)) { box.innerHTML = wordInfoCache.get(key); return; }
    box.innerHTML = '<span class="practice-info__loading">Đang tải định nghĩa…</span>';
    try {
        const res = await fetch(`${apiBase()}/word-info?word=${encodeURIComponent(key)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const info = await res.json();
        const html = `
            ${info.meaning ? `<div class="practice-info__meaning">🇻🇳 ${escapeHtml(info.meaning)}</div>` : ''}
            <div class="practice-info__label">Định nghĩa</div>
            <div>${escapeHtml(info.definition_en || '')}</div>
            <div class="practice-info__label">Ví dụ</div>
            <div>${escapeHtml(info.example_en || '')}</div>`;
        wordInfoCache.set(key, html);
        // Người dùng có thể đã mở từ khác trong lúc chờ — chỉ ghi khi còn đúng từ.
        if (practiceState.data && practiceState.data.word.toLowerCase() === key) box.innerHTML = html;
    } catch (err) {
        if (practiceState.data && practiceState.data.word.toLowerCase() === key) box.innerHTML = '';
    }
}

// ── Nghe lại đoạn user vừa ghi âm ───────────────────────────────────────
// Blob ghi âm chỉ nằm ở client — giữ object URL để user tự nghe lại mình nói,
// so với mẫu 🔊. Reset khi mở từ khác / đóng popup (recording cũ không còn nghĩa).
function setPracticeReplay(blob) {
    if (practiceState.playAudio) { try { practiceState.playAudio.pause(); } catch (e) { /* chưa play */ } }
    if (practiceState.playUrl) URL.revokeObjectURL(practiceState.playUrl);
    practiceState.playUrl = blob ? URL.createObjectURL(blob) : null;
    practiceState.playAudio = null;
    const btn = document.getElementById('practice-replay');
    if (btn) btn.classList.toggle('hidden', !practiceState.playUrl);
}

function playPracticeReplay() {
    if (!practiceState.playUrl) return;
    if (!practiceState.playAudio) practiceState.playAudio = new Audio(practiceState.playUrl);
    practiceState.playAudio.currentTime = 0;
    practiceState.playAudio.play().catch(() => { /* autoplay policy — user sẽ bấm lại */ });
}

// ── Mở / đóng ───────────────────────────────────────────────────────────
function openPracticePopup(data) {
    if (!data || !data.word) return;
    const overlay = ensurePracticeModal();
    practiceState.data = data;
    setPracticeReplay(null);
    renderPracticeBody();
    overlay.classList.remove('hidden');
    loadPracticeWordInfo(data.word);
}

function closePracticePopup() {
    const overlay = document.getElementById('practice-modal');
    if (overlay) overlay.classList.add('hidden');
    stopPracticeStream();
    setPracticeReplay(null);
    practiceState.data = null;
}

function stopPracticeStream() {
    if (practiceState.recorder && practiceState.recording) {
        try { practiceState.recorder.stop(); } catch (e) { /* đã stop */ }
    }
    if (practiceState.stream) {
        practiceState.stream.getTracks().forEach(t => t.stop());
        practiceState.stream = null;
    }
    practiceState.recording = false;
    const mic = document.getElementById('practice-mic');
    if (mic) mic.classList.remove('recording');
}

// ── Ghi âm + chấm lại (mirror pattern recording.js, scope cục bộ popup) ──
async function togglePracticeRecording() {
    const status = document.getElementById('practice-status');
    const mic = document.getElementById('practice-mic');
    if (practiceState.grading) return;
    if (practiceState.recording) { practiceState.recorder.stop(); return; }
    try {
        practiceState.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        status.className = 'practice-status err';
        status.textContent = 'Không truy cập được micro — kiểm tra quyền trình duyệt.';
        return;
    }
    practiceState.chunks = [];
    practiceState.recorder = new MediaRecorder(practiceState.stream);
    practiceState.recorder.addEventListener('dataavailable', e => {
        if (e.data && e.data.size) practiceState.chunks.push(e.data);
    });
    practiceState.recorder.addEventListener('stop', () => {
        const mime = practiceState.recorder.mimeType || 'audio/webm';
        const blob = new Blob(practiceState.chunks, { type: mime });
        stopPracticeStream();
        // Cho nghe lại NGAY cả khi chấm lỗi/chưa nghe rõ — user vẫn muốn tự nghe mình nói gì.
        setPracticeReplay(blob);
        gradePracticeAttempt(blob, mime);
    });
    practiceState.recorder.start();
    practiceState.recording = true;
    mic.classList.add('recording');
    status.className = 'practice-status';
    status.textContent = 'Đang ghi âm… bấm lần nữa để dừng.';
}

async function gradePracticeAttempt(blob, mime) {
    const d = practiceState.data;
    if (!d) return;
    const status = document.getElementById('practice-status');
    const mic = document.getElementById('practice-mic');
    practiceState.grading = true;
    mic.disabled = true;
    status.className = 'practice-status';
    status.textContent = 'Đang chấm…';
    try {
        const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm';
        const fd = new FormData();
        fd.append('audio', new File([blob], `practice-${d.word}.${ext}`, { type: mime }));
        fd.append('text', d.word);
        fd.append('mode', 'mock_test');   // ép bật phoneme analysis
        fd.append('no_ai', 'true');       // 1 từ không cần LLM chấm — chỉ cần phoneme
        fd.append('accent', currentAccent);
        // Từ tiếng Hàn: phải chấm bằng pipeline ko (G2P 표준발음법 + model acoustic
        // Hàn) — exam=topik là cách backend suy lang. Thiếu dòng này sẽ chấm bằng
        // pipeline EN → IPA reference rác.
        if (typeof hasHangul === 'function' && hasHangul(d.word)) fd.append('exam', 'topik');
        const res = await fetch(`${apiBase()}/grade`, { method: 'POST', body: fd });
        if (!res.ok) {
            const raw = await res.text();
            let detail = `HTTP ${res.status}`;
            try { detail = JSON.parse(raw).detail || detail; } catch (e) { /* giữ mã HTTP */ }
            throw new Error(detail);
        }
        const data = await res.json();
        const w = data?.phoneme?.score?.words?.[0];
        if (!w || !(w.phonemes || []).length) {
            throw new Error('Server không trả kết quả phoneme — thử lại.');
        }
        if (w.skip_reason || (w.phonemes || []).every(p => p.status === 'skipped')) {
            status.className = 'practice-status err';
            status.textContent = 'Chưa nghe rõ — hãy nói to, rõ và thử lại.';
            return;
        }
        const pct = practicePct(w.phonemes);
        practiceState.data = { ...d, phonemes: w.phonemes, skip_reason: null };
        renderPracticeBody();
        status.className = pct >= 80 ? 'practice-status good' : 'practice-status';
        status.textContent = pct >= 80 ? `Tuyệt vời — ${pct}%! 🎉` : `Được ${pct}% — nghe mẫu 🔊 rồi thử lại nhé.`;
        // Từ đã lưu → cập nhật điểm luyện gần nhất + snapshot phonemes mới trên
        // server (im lặng, lỗi bỏ qua) — snapshot mới giúp hồ sơ âm yếu
        // (/words/suggestions) phản ánh tiến bộ của user.
        if (window.SavedWords && SavedWords.has(d.word)) {
            SavedWords.add({ word: d.word, last_score: pct / 100, phonemes: w.phonemes }).catch(() => {});
        }
    } catch (err) {
        status.className = 'practice-status err';
        status.textContent = `Lỗi chấm: ${err.message || err}`;
    } finally {
        practiceState.grading = false;
        mic.disabled = false;
    }
}

// ── Bookmark trong popup ────────────────────────────────────────────────
function updatePracticeBookmarkStar() {
    const btn = document.getElementById('practice-bookmark');
    if (!btn || !practiceState.data) return;
    const saved = window.SavedWords && SavedWords.has(practiceState.data.word);
    btn.textContent = saved ? '★' : '☆';
    btn.classList.toggle('saved', !!saved);
    btn.title = saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập';
}

async function togglePracticeBookmark() {
    const d = practiceState.data;
    if (!d || !window.SavedWords) return;
    if (typeof hasHangul === 'function' && hasHangul(d.word)) return;  // nút đã ẩn — chặn nốt edge case
    try {
        if (SavedWords.has(d.word)) await SavedWords.remove(d.word);
        else await SavedWords.add({
            word: d.word, ipa: d.ipa, phonemes: d.phonemes, accuracy: d.accuracy,
        });
    } catch (err) {
        alert(`Lỗi lưu từ: ${err.message || err}`);
    }
    updatePracticeBookmarkStar();
    // Sao ☆ trên bảng lỗi + tab Từ đã lưu render lại lần tới; đồng bộ ngay các nút đang hiện.
    document.querySelectorAll(`.word-bookmark[data-word="${CSS.escape(d.word)}"]`).forEach(b => {
        const saved = SavedWords.has(d.word);
        b.textContent = saved ? '★' : '☆';
        b.classList.toggle('saved', saved);
    });
}

// ── Delegated listeners (gắn 1 lần, panel dựng lại mỗi render) ──────────
document.addEventListener('click', e => {
    if (!(e.target instanceof Element)) return;
    // ☆ trên bảng lỗi: toggle lưu, KHÔNG mở popup.
    const star = e.target.closest('.word-bookmark');
    if (star && star.dataset.practice) {
        e.preventDefault();
        e.stopPropagation();
        let data;
        try { data = JSON.parse(star.dataset.practice); } catch (err) { return; }
        if (!window.SavedWords) return;
        const done = SavedWords.has(data.word)
            ? SavedWords.remove(data.word)
            : SavedWords.add({ word: data.word, ipa: data.ipa, phonemes: data.phonemes, accuracy: data.accuracy });
        done.then(() => {
            const saved = SavedWords.has(data.word);
            document.querySelectorAll(`.word-bookmark[data-word="${CSS.escape(data.word)}"]`).forEach(b => {
                b.textContent = saved ? '★' : '☆';
                b.classList.toggle('saved', saved);
            });
            updatePracticeBookmarkStar();
        }).catch(err => alert(`Lỗi lưu từ: ${err.message || err}`));
        return;
    }
    const opener = e.target.closest('.practice-open');
    if (opener && opener.dataset.practice) {
        // Đừng nuốt click vào nút phát audio lồng trong ô từ.
        if (e.target.closest('.tts-play') || e.target.closest('.phoneme-play')) return;
        e.preventDefault();
        try { openPracticePopup(JSON.parse(opener.dataset.practice)); } catch (err) { /* attr hỏng */ }
    }
});
