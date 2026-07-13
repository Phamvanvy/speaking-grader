'use strict';

// Nút "Gợi ý bài mẫu": gọi /suggest sinh một bài nói mẫu band cao cho dạng câu
// mở hiện tại, hiển thị + cho nghe đọc to bằng Web Speech API của trình duyệt
// (xử lý được văn bản dài, không đụng /tts backend vốn giới hạn 100 ký tự).

// Tùy chọn band mục tiêu theo kỳ thi (option đầu = mặc định = cao nhất).
const TARGET_BAND_OPTIONS = {
    ielts: [
        { value: '9.0', label: 'Band 9.0 (cao nhất)' },
        { value: '8.0', label: 'Band 8.0' },
        { value: '7.0', label: 'Band 7.0' },
        { value: '6.5', label: 'Band 6.5' },
    ],
    toeic: [
        { value: 'TOEIC mức cao nhất (~200)', label: 'Mức cao nhất (~200)' },
        { value: 'TOEIC mức khá (~160)', label: 'Mức khá (~160)' },
    ],
};

// Đổ option band theo kỳ thi đang chọn (gọi từ form.js khi đổi exam/type).
function populateTargetBand() {
    const sel = document.getElementById('target-band');
    if (!sel) return;
    const opts = TARGET_BAND_OPTIONS[examSelect.value] || TARGET_BAND_OPTIONS.toeic;
    sel.innerHTML = opts
        .map(o => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`)
        .join('');
    sel.selectedIndex = 0;
}

// Hiện khối gợi ý cho dạng câu MỞ (cụ thể, không phải Read Aloud / auto-detect),
// đổ lại option band theo kỳ thi, và dọn kết quả cũ khi ẩn. Gọi từ form.js khi
// đổi exam/dạng câu, và một lần khi script này nạp (form.js đã chạy trước đó).
function updateSuggestUI() {
    populateTargetBand();
    const group = document.getElementById('suggest-group');
    if (!group) return;
    const qtVal = questionTypeSelect.value;
    const isOpenEnded = !!qtVal && qtVal !== 'read_aloud';
    group.classList.toggle('hidden', !isOpenEnded);
    if (!isOpenEnded) {
        const box = document.getElementById('suggest-result');
        if (box) { box.classList.add('hidden'); box.innerHTML = ''; }
        if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
}

// Bài mẫu hiện tại (để nút 🔊 đọc lại). Cập nhật mỗi lần render.
let _lastSampleText = '';

function suggestSample() {
    const url = apiBase();
    const questionType = questionTypeSelect.value;
    if (!questionType) {
        alert('Hãy chọn một dạng câu cụ thể trước khi gợi ý bài mẫu.');
        return;
    }

    const box = document.getElementById('suggest-result');
    const btn = document.getElementById('suggest-btn');

    const fd = new FormData();
    fd.append('exam', examSelect.value);
    fd.append('question_type', questionType);

    const promptText = document.getElementById('prompt-text').value;
    if (promptText) fd.append('prompt', promptText);

    const expectedDuration = document.getElementById('expected-duration').value;
    if (expectedDuration) fd.append('expected_duration_sec', expectedDuration);

    const targetBand = document.getElementById('target-band').value;
    if (targetBand) fd.append('target_band', targetBand);

    const feedbackLang = document.getElementById('feedback-lang').value;
    if (feedbackLang) fd.append('feedback_lang', feedbackLang);

    // Tả tranh: gửi ảnh đề bài (nếu ô ảnh đang hiện và có file).
    const imageGroup = document.getElementById('image-group');
    const imageFile = imageInput.files[0];
    if (imageGroup && !imageGroup.classList.contains('hidden') && imageFile) {
        fd.append('image', imageFile);
    }

    btn.disabled = true;
    btn.textContent = '⏳ Đang sinh bài mẫu...';
    box.classList.remove('hidden');
    box.innerHTML = '<div class="status-bar info"><div class="spinner"></div><span>Đang sinh bài mẫu…</span></div>';

    fetch(`${url}/suggest`, { method: 'POST', body: fd })
        .then(async res => {
            if (!res.ok) {
                let detail = `HTTP ${res.status}`;
                try { detail = (await res.json()).detail || detail; } catch (_) {}
                throw new Error(detail);
            }
            return res.json();
        })
        .then(renderSuggest)
        .catch(err => {
            box.innerHTML = `<div class="status-bar error"><span>❌ ${escapeHtml(err.message)}</span></div>`;
        })
        .finally(() => {
            btn.disabled = false;
            btn.textContent = '💡 Gợi ý bài mẫu';
        });
}

function renderSuggest(data) {
    const box = document.getElementById('suggest-result');
    _lastSampleText = data.answer || '';

    const highlights = Array.isArray(data.highlights) ? data.highlights : [];
    const outline = Array.isArray(data.outline) ? data.outline : [];

    const outlineHtml = outline.length
        ? `<div class="suggest-section"><h4>Dàn ý</h4><ul>${
            outline.map(o => `<li>${escapeHtml(o)}</li>`).join('')
        }</ul></div>`
        : '';

    const highlightsHtml = highlights.length
        ? `<div class="suggest-section"><h4>Điểm nhấn đáng học</h4><ul>${
            highlights.map(h => `<li>${escapeHtml(h)}</li>`).join('')
        }</ul></div>`
        : '';

    box.innerHTML = `
        <div class="suggest-card">
            <div class="suggest-head">
                <span class="suggest-band">🎯 ${escapeHtml(data.target_band || '')}</span>
                <button type="button" class="btn btn-secondary" id="suggest-speak-btn" onclick="toggleSpeakSample()" style="width:auto;padding:0.4rem 0.9rem;">🔊 Nghe</button>
            </div>
            <div class="suggest-answer">${escapeHtml(_lastSampleText)}</div>
            ${outlineHtml}
            ${highlightsHtml}
        </div>
    `;
}

// ── Đọc bài mẫu bằng Web Speech API ───────────────────────────────────
function _voiceLangForAccent() {
    if (typeof currentAccent !== 'undefined' && currentAccent === 'gb') return 'en-GB';
    return 'en-US';
}

function toggleSpeakSample() {
    const synth = window.speechSynthesis;
    if (!synth) {
        alert('Trình duyệt không hỗ trợ đọc văn bản (Web Speech API).');
        return;
    }
    const btn = document.getElementById('suggest-speak-btn');
    // Đang đọc → dừng (toggle).
    if (synth.speaking || synth.pending) {
        synth.cancel();
        if (btn) btn.textContent = '🔊 Nghe';
        return;
    }
    if (!_lastSampleText) return;

    const utter = new SpeechSynthesisUtterance(_lastSampleText);
    utter.lang = _voiceLangForAccent();
    utter.rate = 0.95;
    utter.onend = () => { if (btn) btn.textContent = '🔊 Nghe'; };
    utter.onerror = () => { if (btn) btn.textContent = '🔊 Nghe'; };
    if (btn) btn.textContent = '⏹ Dừng';
    synth.speak(utter);
}

// Init: form.js đã chạy syncConditionalFields() trước khi file này nạp, nên đặt
// trạng thái ban đầu cho khối gợi ý ở đây.
updateSuggestUI();
