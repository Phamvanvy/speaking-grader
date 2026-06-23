'use strict';

const API_URL_KEY = 'toeic-grader-api-url';

// ── Exam config ───────────────────────────────────────────────────────
// Mọi khác biệt theo kỳ thi gom về một chỗ (tránh if/else rải rác). Dùng SỐ
// (overallMax/criterionMax) thay vì chuỗi '/200' để dễ thêm TOEFL/VSTEP sau này.
const EXAM_CONFIG = {
    toeic: {
        label: 'TOEIC',
        scoreField: 'estimated_toeic_score',
        overallLabel: 'Estimated TOEIC Speaking Score',
        overallMax: 200,
        criterionMax: 3,
        // `uses` = ô nhập nào HIỆN cho dạng câu này (khớp display_inputs ở backend).
        // `required` = chỉ cần MỘT trong các input này là coi như "có đề" (khớp
        // required_inputs backend) — dùng cho popup cảnh báo trước khi chấm. Cả hai
        // chỉ là cosmetic/UX — backend mới là nơi quyết định chấm thật.
        questionTypes: [
            { value: '', label: 'Auto-detect', uses: ['reference', 'image', 'prompt'] },
            { value: 'read_aloud', label: 'Read Aloud', uses: ['reference'], required: ['reference'] },
            { value: 'describe_picture', label: 'Describe Picture', uses: ['image'], required: ['image'] },
            { value: 'respond_questions', label: 'Respond to Questions', uses: ['prompt'], required: ['prompt'] },
            { value: 'respond_with_info', label: 'Respond with Info', uses: ['prompt', 'image'], required: ['prompt'] },
            { value: 'express_opinion', label: 'Express Opinion', uses: ['prompt'], required: ['prompt'] },
        ],
    },
    ielts: {
        label: 'IELTS',
        scoreField: 'estimated_ielts_band',
        overallLabel: 'Estimated IELTS Band',
        overallMax: 9,
        criterionMax: 9,
        // Không có "Auto-detect": Part 1 vs Part 3 không phân biệt được → luôn gửi rõ.
        questionTypes: [
            { value: 'part1_interview', label: 'Part 1 — Interview', uses: ['prompt'], required: ['prompt'] },
            { value: 'part2_long_turn', label: 'Part 2 — Long turn (cue card)', uses: ['prompt'], required: ['prompt'] },
            { value: 'part3_discussion', label: 'Part 3 — Discussion', uses: ['prompt'], required: ['prompt'] },
        ],
    },
};

function examConfig(exam) {
    return EXAM_CONFIG[exam] || EXAM_CONFIG.toeic;
}

// Holds the most recent /grade-batch response so "Print / PDF" can rebuild a report.
let lastBatchData = null;

// Holds the most recent single /grade response (+ the file name it came from)
// so "Export CSV" / "Print" can rebuild a report.
let lastSingleData = null;
let lastSingleFilename = '';

// Load saved API URL, or pick a sensible default.
// - Saved value always wins.
// - Served from a real host (not file:// or localhost) → default the API to the
//   SAME origin the page is served from. So if FastAPI serves this page, it just
//   works on any domain with no editing and no CORS issue.
// - Otherwise keep http://localhost:8000 for local dev.
const savedUrl = localStorage.getItem(API_URL_KEY);
const apiUrlInput = document.getElementById('api-url');
if (savedUrl) {
    apiUrlInput.value = savedUrl;
} else if (
    location.protocol !== 'file:' &&
    location.hostname !== 'localhost' &&
    location.hostname !== '127.0.0.1'
) {
    apiUrlInput.value = location.origin;
}

// ── File upload handling ──────────────────────────────────────────────
const fileLabel = document.getElementById('single-file-label');
const fileInput = document.getElementById('audio-file');
const fileNameDisplay = document.getElementById('single-file-name');

// Object URLs created for inline <audio> playback — revoked before each
// re-render so blobs don't leak as the selection changes.
const audioObjectUrls = [];

function revokeAudioUrls() {
    audioObjectUrls.forEach(u => URL.revokeObjectURL(u));
    audioObjectUrls.length = 0;
}

// Lưu ý: KHÔNG cần handler click để gọi fileInput.click() ở đây.
// <input type="file"> đã nằm trong <label>, nên click vào label tự mở
// hộp thoại chọn file. Nếu thêm fileInput.click() thủ công thì dialog sẽ
// mở 2 lần (một lần do hành vi mặc định của label, một lần do JS).
fileInput.addEventListener('change', renderFileList);

// Add files to the audio input WITHOUT discarding what's already there.
// (File inputs are read-only, so we rebuild a DataTransfer list.) Used by the
// recorder to append a recording alongside any uploaded files.
function addAudioFiles(newFiles) {
    const dt = new DataTransfer();
    Array.from(fileInput.files).forEach(f => dt.items.add(f));
    newFiles.forEach(f => dt.items.add(f));
    fileInput.files = dt.files;
    renderFileList();
}

// ── Mic recording (MediaRecorder) ─────────────────────────────────────
let mediaRecorder = null;
let recordedChunks = [];
let recordTimerId = null;
let recordSeconds = 0;

function startRecordTimer() {
    // Clear any previous interval first so we never orphan one (which would
    // keep ticking after Stop). Guards against double-start re-entrancy.
    if (recordTimerId) clearInterval(recordTimerId);
    recordSeconds = 0;
    const t = document.getElementById('record-timer');
    t.textContent = '● 0:00';
    recordTimerId = setInterval(() => {
        recordSeconds++;
        const m = Math.floor(recordSeconds / 60);
        const s = recordSeconds % 60;
        t.textContent = `● ${m}:${String(s).padStart(2, '0')}`;
    }, 1000);
}

function stopRecordTimer() {
    if (recordTimerId) {
        clearInterval(recordTimerId);
        recordTimerId = null;
    }
    document.getElementById('record-timer').textContent = '';
}

// Pick a filename extension the API accepts, based on the recorder's mime type.
function recordingExtension(mimeType) {
    if (mimeType.includes('ogg')) return '.ogg';
    if (mimeType.includes('mp4') || mimeType.includes('mpeg')) return '.mp4';
    return '.webm';  // Chrome/Firefox default
}

let isStartingRecording = false;  // true while getUserMedia is pending

async function toggleRecording() {
    const btn = document.getElementById('record-btn');

    // Already recording → stop (the 'stop' handler does the rest).
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        return;
    }

    // Ignore extra clicks while the mic permission/stream is still resolving —
    // otherwise we'd spin up a second recorder + timer (orphaned timer bug).
    if (isStartingRecording) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Trình duyệt không hỗ trợ ghi âm (getUserMedia). Hãy dùng Chrome/Edge/Firefox bản mới và truy cập qua HTTPS hoặc localhost.');
        return;
    }

    let stream;
    isStartingRecording = true;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        alert(`Không truy cập được micro: ${err.message}`);
        return;
    } finally {
        isStartingRecording = false;
    }

    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.addEventListener('dataavailable', (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
    });

    mediaRecorder.addEventListener('stop', async () => {
        stream.getTracks().forEach(t => t.stop());  // release the mic
        stopRecordTimer();
        btn.textContent = '🎙️ Record audio';
        btn.classList.remove('recording');

        const type = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(recordedChunks, { type });
        const stamp = new Date().toISOString().slice(11, 19).replace(/:/g, '-');
        const name = `recording-${stamp}${recordingExtension(type)}`;
        const file = new File([blob], name, { type });
        addAudioFiles([file]);

        // Persist on-device so the recording survives a page reload.
        try {
            await saveRecording({ name, blob, type, size: blob.size, createdAt: Date.now() });
            renderSavedRecordings();
        } catch (err) {
            console.warn('Could not save recording locally:', err);
        }
    });

    mediaRecorder.start();
    btn.textContent = '⏹ Stop recording';
    btn.classList.add('recording');
    startRecordTimer();
}

// ── Saved recordings (IndexedDB) ──────────────────────────────────────
// Recordings are persisted ON THE DEVICE so they survive a page reload.
// localStorage can't hold Blobs reliably, so we use IndexedDB. Each row:
//   { id (auto), name, blob, type, size, createdAt }
const REC_DB_NAME = 'speaking-grader';
const REC_STORE = 'recordings';
let recDbPromise = null;

function recDb() {
    if (recDbPromise) return recDbPromise;
    recDbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(REC_DB_NAME, 1);
        req.onupgradeneeded = () => {
            req.result.createObjectStore(REC_STORE, { keyPath: 'id', autoIncrement: true });
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
    return recDbPromise;
}

function recStore(mode) {
    return recDb().then(db => db.transaction(REC_STORE, mode).objectStore(REC_STORE));
}

function reqDone(req) {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function saveRecording(rec) {
    const store = await recStore('readwrite');
    return reqDone(store.add(rec));
}

// Newest first.
async function listRecordings() {
    const store = await recStore('readonly');
    const all = await reqDone(store.getAll());
    return all.sort((a, b) => b.createdAt - a.createdAt);
}

async function getRecording(id) {
    const store = await recStore('readonly');
    return reqDone(store.get(id));
}

async function deleteRecordingDb(id) {
    const store = await recStore('readwrite');
    return reqDone(store.delete(id));
}

async function clearRecordingsDb() {
    const store = await recStore('readwrite');
    return reqDone(store.clear());
}

// ── Saved recordings UI ───────────────────────────────────────────────
const savedRecordingsCard = document.getElementById('saved-recordings-card');
const savedRecordingsList = document.getElementById('saved-recordings-list');

// Object URLs for the inline players — revoked before each re-render.
const savedRecordingUrls = [];
function revokeSavedRecordingUrls() {
    savedRecordingUrls.forEach(u => URL.revokeObjectURL(u));
    savedRecordingUrls.length = 0;
}

function formatBytes(n) {
    if (!n) return '';
    const kb = n / 1024;
    return kb < 1024 ? `${kb.toFixed(0)} KB` : `${(kb / 1024).toFixed(1)} MB`;
}

async function renderSavedRecordings() {
    let recs;
    try {
        recs = await listRecordings();
    } catch (err) {
        // IndexedDB unavailable (private mode, etc.) → just hide the panel.
        savedRecordingsCard.classList.add('hidden');
        return;
    }
    revokeSavedRecordingUrls();
    // Hide the whole card when nothing is stored, to keep the UI tidy.
    if (!recs.length) {
        savedRecordingsCard.classList.add('hidden');
        savedRecordingsList.innerHTML = '';
        return;
    }
    savedRecordingsCard.classList.remove('hidden');
    savedRecordingsList.innerHTML = recs.map(rec => {
        const url = URL.createObjectURL(rec.blob);
        savedRecordingUrls.push(url);
        const when = new Date(rec.createdAt).toLocaleString();
        const size = formatBytes(rec.size);
        return `
        <div class="file-item file-item-audio">
            <div class="saved-rec-head">
                <span class="name">📄 ${escapeHtml(rec.name)}</span>
                <span class="saved-rec-meta">${escapeHtml(when)}${size ? ' · ' + size : ''}</span>
            </div>
            <audio controls preload="metadata" src="${url}"></audio>
            <div class="saved-rec-actions">
                <button class="btn btn-secondary" onclick="useRecording(${rec.id})" style="width:auto;padding:0.35rem 0.8rem;">➕ Add to grading</button>
                <button class="btn btn-secondary remove-btn" onclick="deleteRecording(${rec.id})" style="width:auto;padding:0.35rem 0.8rem;">🗑 Delete</button>
            </div>
        </div>`;
    }).join('');
}

// Pull a saved recording back into the grading file list.
async function useRecording(id) {
    const rec = await getRecording(id);
    if (!rec) return;
    const file = new File([rec.blob], rec.name, { type: rec.type });
    addAudioFiles([file]);
    fileNameDisplay.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function deleteRecording(id) {
    if (!confirm('Delete this recording from your device?')) return;
    await deleteRecordingDb(id);
    renderSavedRecordings();
}

async function deleteAllRecordings() {
    const recs = await listRecordings().catch(() => []);
    if (!recs.length) return;
    if (!confirm(`Delete all ${recs.length} saved recording(s) from your device?`)) return;
    await clearRecordingsDb();
    renderSavedRecordings();
}

// ── Image upload (Describe Picture) ───────────────────────────────────
const imageInput = document.getElementById('image-file');
const imageLabel = document.getElementById('image-file-label');
const imagePreview = document.getElementById('image-preview');
let imageObjectUrl = null;

imageInput.addEventListener('change', renderImagePreview);

function renderImagePreview() {
    if (imageObjectUrl) {
        URL.revokeObjectURL(imageObjectUrl);
        imageObjectUrl = null;
    }
    const file = imageInput.files[0];
    if (!file) {
        imageLabel.classList.remove('has-file');
        imagePreview.innerHTML = '';
        return;
    }
    imageLabel.classList.add('has-file');
    imageObjectUrl = URL.createObjectURL(file);
    imagePreview.innerHTML = `
        <img src="${imageObjectUrl}" alt="${escapeHtml(file.name)}" class="preview-img">
        <button class="btn btn-secondary" onclick="clearImage(event)" style="width:auto;padding:0.4rem 0.9rem;">Clear image</button>
    `;
}

function clearImage(e) {
    e.stopPropagation();
    e.preventDefault();
    imageInput.value = '';
    renderImagePreview();
}

// ── Grading mode notes ────────────────────────────────────────────────
// Each mode uses a different ASR backend and toggles phoneme analysis on/off,
// so scores can legitimately differ across modes (by design). This note under
// the selector explains why.
const MODE_NOTES = {
    practice: 'Fast first pass (faster-whisper). Auto-upgrades to the Mock Test '
        + 'pipeline (better ASR + phoneme analysis) when confidence/coverage is low.',
    mock_test: 'Most accurate: best ASR (WhisperX) + phoneme analysis ON. '
        + 'Use this as the reference score.',
};

const modeSelect = document.getElementById('mode');
const modeNote = document.getElementById('mode-note');

function updateModeNote() {
    modeNote.textContent = MODE_NOTES[modeSelect.value] || '';
}

modeSelect.addEventListener('change', updateModeNote);
updateModeNote();

// ── Exam selector ─────────────────────────────────────────────────────
// Đổi exam → nạp lại danh sách Question Type và reset về option đầu (tránh giữ
// giá trị của exam cũ, vd 'read_aloud', gây 400 khi submit IELTS).
const examSelect = document.getElementById('exam');
const questionTypeSelect = document.getElementById('question-type');

function populateQuestionTypes() {
    const cfg = examConfig(examSelect.value);
    questionTypeSelect.innerHTML = cfg.questionTypes
        .map(qt => `<option value="${qt.value}">${escapeHtml(qt.label)}</option>`)
        .join('');
    questionTypeSelect.selectedIndex = 0;  // reset về option đầu của exam mới
}

// Hiện đúng ô nhập theo dạng câu đang chọn ("cái nào dùng cái đó"): Read Aloud
// chỉ Reference, Describe Picture chỉ Image, các dạng Q&A chỉ Prompt... Khi ẩn
// một group → XÓA luôn giá trị bên trong để không gửi nhầm dữ liệu cũ còn sót
// trong DOM khi user chuyển qua lại các dạng câu.
function setGroupVisible(groupId, visible) {
    const group = document.getElementById(groupId);
    if (!group) return;
    group.classList.toggle('hidden', !visible);
    if (!visible) {
        group.querySelectorAll('input, textarea').forEach(el => { el.value = ''; });
    }
}

function syncConditionalFields() {
    const cfg = examConfig(examSelect.value);
    const qt = cfg.questionTypes.find(q => q.value === questionTypeSelect.value);
    // Không tìm thấy metadata (phòng hờ) → hiện tất cả.
    const uses = (qt && qt.uses) || ['reference', 'image', 'prompt'];
    setGroupVisible('reference-group', uses.includes('reference'));
    setGroupVisible('prompt-group', uses.includes('prompt'));
    const imageVisible = uses.includes('image');
    setGroupVisible('image-group', imageVisible);
    // image-preview nằm ngoài <input> → đồng bộ lại preview sau khi reset value.
    if (!imageVisible) renderImagePreview();
}

examSelect.addEventListener('change', () => {
    populateQuestionTypes();
    syncConditionalFields();
});
questionTypeSelect.addEventListener('change', syncConditionalFields);
populateQuestionTypes();
syncConditionalFields();

// ── Dark mode ─────────────────────────────────────────────────────────
// Lựa chọn của user được lưu localStorage; lần đầu thì theo cài đặt hệ điều hành.
const THEME_KEY = 'toeic-grader-theme';
const themeToggle = document.getElementById('theme-toggle');

function applyTheme(theme) {
    const dark = theme === 'dark';
    document.body.classList.toggle('dark', dark);
    themeToggle.textContent = dark ? '☀️' : '🌙';
}

function currentTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'dark' || saved === 'light') return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function toggleTheme() {
    const next = document.body.classList.contains('dark') ? 'light' : 'dark';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
}

applyTheme(currentTheme());

function renderFileList() {
    revokeAudioUrls();
    const files = Array.from(fileInput.files);
    if (files.length === 0) {
        fileLabel.classList.remove('has-file');
        fileNameDisplay.innerHTML = '';
        return;
    }
    fileLabel.classList.add('has-file');
    const header = files.length > 1
        ? `<div class="file-item" style="background:#eef2ff;color:#3730a3;font-weight:600;">📦 ${files.length} files — will be graded as a batch</div>`
        : '';
    // Each file gets an inline <audio> player so the user can listen back
    // before grading (works for uploads and recordings alike).
    fileNameDisplay.innerHTML = header + files.map(f => {
        const url = URL.createObjectURL(f);
        audioObjectUrls.push(url);
        return `
        <div class="file-item file-item-audio">
            <span class="name">📄 ${escapeHtml(f.name)}</span>
            <audio controls preload="metadata" src="${url}"></audio>
        </div>
    `;
    }).join('') + `
        <button class="btn btn-secondary" onclick="clearFile(event)" style="margin-top:0.5rem;width:auto;padding:0.4rem 0.9rem;">Clear</button>
    `;
}

function clearFile(e) {
    e.stopPropagation();
    e.preventDefault();
    revokeAudioUrls();
    fileInput.value = '';
    fileLabel.classList.remove('has-file');
    fileNameDisplay.innerHTML = '';
}

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
    const url = document.getElementById('api-url').value.replace(/\/$/, '');
    const statusDiv = document.getElementById('health-status');

    statusDiv.innerHTML = '<div class="status-bar info"><div class="spinner"></div><span>Checking...</span></div>';

    try {
        const res = await fetch(`${url}/health`);
        const data = await res.json();

        if (res.ok) {
            localStorage.setItem(API_URL_KEY, url);
            statusDiv.innerHTML = `
                <div class="status-bar success">
                    <span>✅ Connected to ${escapeHtml(data.backend || 'TOEIC Speaking Grader')}</span>
                </div>
            `;
        } else {
            statusDiv.innerHTML = `
                <div class="status-bar error">
                    <span>❌ API returned error: ${res.status}</span>
                </div>
            `;
        }
    } catch (err) {
        statusDiv.innerHTML = `
            <div class="status-bar error">
                <span>❌ Cannot connect: ${escapeHtml(err.message)}</span>
            </div>
        `;
    }
}

// ── Grading ───────────────────────────────────────────────────────────
// Append the shared grading options (same form for single & batch).
function appendCommonFields(formData) {
    // Chỉ gửi input đang HIỆN cho dạng câu hiện tại (group không bị ẩn). Tránh
    // gửi dữ liệu cũ còn sót trong DOM khi chuyển dạng câu. (Group ẩn đã được
    // clear value ở syncConditionalFields nên đây là lớp bảo vệ thứ hai.)
    const isVisible = (groupId) =>
        !document.getElementById(groupId).classList.contains('hidden');

    const referenceText = document.getElementById('reference-text').value;
    if (isVisible('reference-group') && referenceText) formData.append('text', referenceText);

    const promptText = document.getElementById('prompt-text').value;
    if (isVisible('prompt-group') && promptText) formData.append('prompt', promptText);

    formData.append('exam', examSelect.value);

    const questionType = questionTypeSelect.value;
    if (questionType) formData.append('question_type', questionType);

    formData.append('mode', document.getElementById('mode').value);

    const feedbackLang = document.getElementById('feedback-lang').value;
    if (feedbackLang) formData.append('feedback_lang', feedbackLang);

    const expectedDuration = document.getElementById('expected-duration').value;
    if (expectedDuration) formData.append('expected_duration_sec', expectedDuration);

    // Ảnh đề bài (Describe Picture / Respond with Info) — chỉ gửi khi ô ảnh đang hiện.
    const imageFile = imageInput.files[0];
    if (isVisible('image-group') && imageFile) formData.append('image', imageFile);

    formData.append('no_ai', document.getElementById('no-ai').checked);
}

// True nếu dạng câu đang chọn đã có "đề" (mirror QuestionType.has_task_context
// ở backend — CHỈ để cảnh báo UX; backend vẫn tự enforce). Auto-detect / không có
// metadata `required` → bỏ pre-check, để backend quyết.
function hasTaskContext() {
    const cfg = examConfig(examSelect.value);
    const qt = cfg.questionTypes.find(q => q.value === questionTypeSelect.value);
    if (!qt || !qt.required) return true;
    const present = new Set();
    if (document.getElementById('prompt-text').value.trim()) present.add('prompt');
    if (document.getElementById('reference-text').value.trim()) present.add('reference');
    if (imageInput.files[0]) present.add('image');
    // provided_info: UI chưa có ô riêng → không có từ UI.
    return qt.required.some(r => present.has(r));
}

// Grade — routes to /grade (1 file) or /grade-batch (≥2 files).
async function grade() {
    const url = document.getElementById('api-url').value.replace(/\/$/, '');
    const files = Array.from(fileInput.files);

    if (files.length === 0) {
        alert('Please select at least one audio file');
        return;
    }

    // Thiếu đề bài → cảnh báo trước: vẫn chấm được nhưng CHỈ phát âm, không có
    // điểm tổng. Cho user cơ hội quay lại nhập đề (Cancel) thay vì chấm hụt.
    if (!hasTaskContext()) {
        const ok = confirm(
            '⚠️ Chưa nhập đề/câu hỏi cho dạng câu này nên không thể chấm điểm '
            + 'tổng — chỉ chấm phát âm.\n\n'
            + 'Nhấn OK để vẫn chấm phát âm, hoặc Cancel để quay lại nhập đề bài.'
        );
        if (!ok) return;
    }

    const isBatch = files.length > 1;
    const btn = document.getElementById('grade-btn');
    btn.disabled = true;
    btn.textContent = isBatch ? `Grading ${files.length} files...` : 'Grading...';

    const formData = new FormData();
    if (isBatch) {
        files.forEach(f => formData.append('audios', f));
    } else {
        formData.append('audio', files[0]);
        lastSingleFilename = files[0].name;
    }
    appendCommonFields(formData);

    const endpoint = isBatch ? '/grade-batch' : '/grade';
    try {
        const res = await fetch(`${url}${endpoint}`, { method: 'POST', body: formData });

        if (!res.ok) {
            let detail = `HTTP ${res.status}`;
            try { detail = (await res.json()).detail || detail; } catch (_) {}
            throw new Error(detail);
        }

        const data = await res.json();
        if (isBatch) {
            closeResult();
            showBatchResult(data);
        } else {
            closeBatchResult();
            showSingleResult(data);
        }
    } catch (err) {
        alert(`Error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Grade Now';
    }
}

// ── Rendering helpers (shared by single & batch) ──────────────────────
function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function pct(x) {
    return ((x || 0) * 100).toFixed(1) + '%';
}

// Build feature tiles from a /grade `features` object (real API shape).
function featureTiles(features) {
    const tiles = [
        { name: 'WPM', value: features.speech_rate_wpm != null ? Math.round(features.speech_rate_wpm) : '--' },
        { name: 'Words', value: features.word_count ?? '--' },
        { name: 'Duration', value: (features.audio_duration_sec || 0).toFixed(1) + 's' },
        { name: 'ASR Confidence', value: pct(features.avg_word_probability) },
        { name: 'Fillers', value: features.filler_count ?? '--' },
        { name: 'Pauses', value: features.pause_count ?? '--' },
    ];
    // Read Aloud only: accuracy_metrics present.
    const acc = features.accuracy_metrics;
    if (acc) {
        tiles.push({ name: 'Coverage', value: pct(acc.coverage) });
        tiles.push({ name: 'Word Accuracy', value: pct(1 - (acc.wer ?? 0)) });
    }
    return tiles;
}

function featureGridHtml(features) {
    return featureTiles(features).map(f => `
        <div class="feature-item">
            <div class="value">${escapeHtml(f.value)}</div>
            <div class="name">${f.name}</div>
        </div>
    `).join('');
}

// Vocabulary corrections table (said → suggested + reason + example).
// Renders nothing when there are no corrections (no empty table shell).
function correctionsHtml(corrections) {
    const items = Array.isArray(corrections) ? corrections : [];
    if (!items.length) return '';
    const rows = items.map(c => `
        <div style="border-top:1px solid #eee;padding:0.4rem 0;font-size:0.88rem;">
            <div><span style="color:#b91c1c;text-decoration:line-through;">${escapeHtml(c.said)}</span>
                 <span style="color:#888;">→</span>
                 <span style="color:#047857;font-weight:600;">${escapeHtml(c.suggested)}</span></div>
            ${c.reason ? `<div style="color:#666;">${escapeHtml(c.reason)}</div>` : ''}
            ${c.example ? `<div style="color:#555;font-style:italic;">“${escapeHtml(c.example)}”</div>` : ''}
        </div>`).join('');
    return `<div style="margin-top:0.5rem;">
        <div style="font-weight:600;color:#333;font-size:0.85rem;">Word corrections</div>${rows}
    </div>`;
}

// Severity helpers shared by phoneme renderers.
const sevColor = s => (s === 'high' ? '#b91c1c' : s === 'medium' ? '#b45309' : '#6b7280');
const sevLabel = s => (s === 'high' ? 'cao' : s === 'medium' ? 'trung bình' : s === 'low' ? 'thấp' : '');

// ELSA-style phoneme detail fed by data.phoneme.score.words: every word shows its
// full reference IPA with mispronounced sounds bolded/red in place, followed by a
// detail table (Từ / IPA đúng / Bạn đọc / Âm sai / Mức độ) for the words with errors.
// Falls back to the legacy errors-only table when `words` is absent (older payloads).
function phonemeErrorsHtml(phoneme, opts = {}) {
    const score = phoneme?.score;
    if (!score) return '';
    const words = Array.isArray(score.words) ? score.words : null;
    if (!words) return phonemeErrorsLegacyHtml(phoneme);   // older payloads
    if (!words.length) return '';

    const isBad = p => p.status === 'sub' || p.status === 'del';
    // Dấu nhấn âm (nhấn âm) — span riêng, render trước nguyên âm. Backend đã
    // suppress nhấn cho từ đơn âm tiết nên UI chỉ cần đọc p.stress (có thể vắng
    // ở payload cũ → bỏ qua).
    const stressMark = p =>
        p.stress === 'primary' ? '<span class="phoneme-stress">ˈ</span>'
      : p.stress === 'secondary' ? '<span class="phoneme-stress">ˌ</span>'
      : '';
    const symHtml = p => {
        const cls = p.status === 'del' ? 'phoneme-sym phoneme-sym--missing'
                  : p.status === 'sub' ? 'phoneme-sym phoneme-sym--bad'
                  : 'phoneme-sym';
        return `${stressMark(p)}<span class="${cls}">${escapeHtml(p.symbol)}</span>`;
    };
    // Full reference IPA, wrapped in /…/ here (backend stores symbols without slashes).
    const ipaHtml = w => `<span class="phoneme-ipa">/${(w.phonemes || []).map(symHtml).join('')}/</span>`;
    // Heard transcription: ok→symbol, sub→heard (bold+red), del→omitted.
    const heardHtml = w => {
        const parts = (w.phonemes || []).filter(p => p.status !== 'del').map(p =>
            p.status === 'sub'
                ? `<span class="phoneme-sym phoneme-sym--bad">${escapeHtml(p.heard ?? '')}</span>`
                : `<span class="phoneme-sym">${escapeHtml(p.symbol)}</span>`);
        return `<span class="phoneme-ipa">/${parts.join('')}/</span>`;
    };

    // ── Per-word cards (all words) ──
    const cardHtml = w => {
        const hasErr = (w.phonemes || []).some(isBad);
        return `<div class="phoneme-word${hasErr ? ' phoneme-word--err' : ''}">
            <span class="phoneme-word__text">${escapeHtml(w.word)}</span>
            ${ipaHtml(w)}
        </div>`;
    };
    const CAP = 12;
    const head = words.slice(0, CAP).map(cardHtml).join('');
    const rest = words.slice(CAP);
    const moreCards = rest.length
        ? `<details style="margin-top:0.3rem;"><summary style="cursor:pointer;color:#4338ca;font-size:0.85rem;">hiện ${rest.length} từ nữa</summary><div class="phoneme-words">${rest.map(cardHtml).join('')}</div></details>`
        : '';

    // ── Detail table (only words with at least one error) ──
    const sevRank = { high: 2, medium: 1, low: 0 };
    const errWords = words.filter(w => (w.phonemes || []).some(isBad));
    const tableRows = errWords.map(w => {
        const bad = (w.phonemes || []).filter(isBad);
        const pairs = bad.map(p => {
            const heard = p.status === 'del' ? '∅' : escapeHtml(p.heard ?? '');
            return `<span style="color:${sevColor(p.severity)};">${escapeHtml(p.symbol)} → ${heard}</span>`;
        }).join('<br>');
        const worst = bad.reduce((acc, p) =>
            (sevRank[p.severity] ?? 0) > (sevRank[acc] ?? -1) ? p.severity : acc, 'low');
        return `<tr>
            <td class="phoneme-table__word">${escapeHtml(w.word)}</td>
            <td>${ipaHtml(w)}</td>
            <td>${heardHtml(w)}</td>
            <td>${pairs}</td>
            <td style="color:${sevColor(worst)};white-space:nowrap;">${sevLabel(worst)}</td>
        </tr>`;
    }).join('');
    const table = errWords.length
        ? `<table class="phoneme-table">
            <thead><tr><th>Từ</th><th>IPA đúng</th><th>Bạn đọc</th><th>Âm sai</th><th>Mức độ</th></tr></thead>
            <tbody>${tableRows}</tbody>
        </table>`
        : '<div style="color:#16a34a;font-size:0.88rem;margin-top:0.4rem;">Tất cả các âm đều đúng 🎉</div>';

    const acc = score.overall_accuracy;
    const accLine = acc != null
        ? `<span style="color:#666;font-weight:400;font-size:0.85rem;"> · accuracy ${pct(acc)}</span>` : '';
    const truncLine = score.words_truncated
        ? `<div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">hiển thị ${words.length}/${score.words_total} từ</div>` : '';

    const titleText = `Pronunciation detail (phoneme)${accLine}`;
    const body = `
        <div class="phoneme-legend"><span class="phoneme-sym--bad">đỏ/đậm</span> = âm sai · <span class="phoneme-sym--missing">gạch</span> = thiếu âm · <span class="phoneme-stress">ˈ</span> = nhấn âm</div>
        <div class="phoneme-legend">Từ lặp lại là từ xuất hiện nhiều lần trong câu (câu có ý nghĩa) — không phải lỗi trùng lặp.</div>
        ${truncLine}
        <div class="phoneme-words">${head}</div>${moreCards}
        ${table}`;
    // Collapsible: lồng dưới tiêu chí Pronunciation — dùng <summary> làm tiêu đề
    // (giữ accuracy) thay cho .phoneme-detail__title để khỏi lặp tiêu đề.
    if (opts.collapsible) {
        return `<details class="phoneme-detail phoneme-detail-wrapper">
            <summary class="phoneme-detail__title">${titleText}</summary>
            ${body}
        </details>`;
    }
    return `<div class="phoneme-detail">
        <div class="phoneme-detail__title">${titleText}</div>
        ${body}
    </div>`;
}

// Legacy errors-only table — kept for payloads predating per-word `words` detail.
function phonemeErrorsLegacyHtml(phoneme) {
    const errors = phoneme?.score?.errors;
    if (!Array.isArray(errors) || !errors.length) return '';
    const shown = errors.filter(e => e.severity === 'high' || e.severity === 'medium');
    if (!shown.length) return '';
    const CAP = 8;
    const arrow = e => {
        const exp = e.expected != null ? `/${escapeHtml(e.expected)}/` : '∅';
        const pred = e.predicted != null ? `/${escapeHtml(e.predicted)}/` : '∅ (dropped)';
        return `${exp} <span style="color:#888;">→</span> ${pred}`;
    };
    const rowHtml = e => `
        <div style="display:flex;align-items:center;gap:0.6rem;border-top:1px solid #eee;padding:0.3rem 0;font-size:0.88rem;">
            <span style="min-width:5rem;font-weight:600;color:#333;">${e.word ? escapeHtml(e.word) : '—'}</span>
            <span style="flex:1;">${arrow(e)}</span>
            <span style="color:${sevColor(e.severity)};font-size:0.8rem;">${escapeHtml(e.severity)}</span>
        </div>`;
    const head = shown.slice(0, CAP).map(rowHtml).join('');
    const rest = shown.slice(CAP);
    const more = rest.length
        ? `<details style="margin-top:0.2rem;"><summary style="cursor:pointer;color:#4338ca;font-size:0.85rem;">show ${rest.length} more</summary>${rest.map(rowHtml).join('')}</details>`
        : '';
    const acc = phoneme.score.overall_accuracy;
    const accLine = acc != null
        ? `<span style="color:#666;font-weight:400;font-size:0.85rem;"> · accuracy ${pct(acc)}</span>` : '';
    return `<div style="margin-top:1rem;background:#fff7ed;border-radius:8px;padding:0.85rem;">
        <div style="font-weight:600;color:#333;margin-bottom:0.2rem;">Pronunciation detail (phoneme)${accLine}</div>
        <div style="color:#888;font-size:0.8rem;margin-bottom:0.3rem;">word · expected → heard · severity</div>
        ${head}${more}
    </div>`;
}

function scoresBreakdownHtml(scores, exam, phoneme, opts = {}) {
    if (!scores) {
        // pronunciation-only: thiếu đề bài → backend chủ động bỏ chấm điểm tổng,
        // chỉ trả phoneme. KHÔNG suy ra trạng thái này từ (scores == null) vì còn
        // nhiều lý do khác (no_ai, gating, lỗi/timeout LLM) → dựa vào cờ backend.
        if (opts.pronunciationOnly) {
            const msg = opts.notice
                || 'Chưa có đề bài — chỉ chấm phát âm. Nhập đề để chấm đầy đủ.';
            return `<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;padding:0.85rem;color:#854d0e;line-height:1.5;">
                    ⚠️ ${escapeHtml(msg)}
                </div>`
                + phonemeErrorsHtml(phoneme);
        }
        return '<p style="color:#666;">No AI scoring (ASR-only or skipped by gating).</p>'
             + phonemeErrorsHtml(phoneme);
    }
    const cfg = examConfig(exam);
    const overall = scores[cfg.scoreField];
    const row = (label, val) => `
        <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e5e7eb;">
            <span style="color:#555;">${label}</span>
            <span style="color:#333;font-weight:600;">${escapeHtml(val ?? '--')}</span>
        </div>`;
    let html = row('Task Completion', scores.task_completion)
             + row('Content Relevance', scores.content_relevance)
             + row(cfg.overallLabel,
                   overall != null ? overall + '/' + cfg.overallMax : '--');

    // Khối phoneme lồng dưới tiêu chí Pronunciation. Cờ chống render 2 lần khi
    // có nhiều tiêu chí khớp "pronun"; nếu không khớp tiêu chí nào → fallback cuối.
    let renderedPhoneme = false;
    const criteria = Array.isArray(scores.criteria) ? scores.criteria : [];
    if (criteria.length) {
        html += '<div style="margin-top:1rem;">' + criteria.map(c => {
            const suggestions = (c.suggestions || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');
            // Nhận diện tiêu chí phát âm: thử các field id/code khả dĩ trước, rồi
            // mới fallback heuristic chứa "pronun" (criterion có thể là label).
            const key = (c.code || c.id || c.key || c.criterion || '').toString().toLowerCase();
            const isPronunciation = key === 'pronunciation' || key.includes('pronun');
            let phonemeBlock = '';
            if (isPronunciation && !renderedPhoneme) {
                const detail = phonemeErrorsHtml(phoneme, { collapsible: true });
                if (detail) {
                    phonemeBlock = detail;
                    renderedPhoneme = true;
                }
            }
            return `
                <div style="background:#f9fafb;border-radius:8px;padding:0.85rem;margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                        <span style="font-weight:600;color:#333;">${escapeHtml(c.criterion)}</span>
                        <span style="background:#4f46e5;color:#fff;border-radius:6px;padding:0.1rem 0.55rem;font-weight:600;font-size:0.85rem;">${escapeHtml(c.score)}/${cfg.criterionMax}</span>
                    </div>
                    <div style="color:#555;line-height:1.5;font-size:0.92rem;">${escapeHtml(c.justification)}</div>
                    ${suggestions ? `<ul style="margin:0.5rem 0 0 1.1rem;color:#4338ca;font-size:0.9rem;">${suggestions}</ul>` : ''}
                    ${correctionsHtml(c.corrections)}
                    ${phonemeBlock}
                </div>`;
        }).join('') + '</div>';
    }
    if (scores.score_rationale) {
        html += `<div style="margin-top:0.75rem;">
            <div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Score Rationale</div>
            <p style="color:#555;line-height:1.6;white-space:pre-wrap;">${escapeHtml(scores.score_rationale)}</p>
        </div>`;
    }
    // Fallback: không có tiêu chí phát âm nào khớp (vd exam khác) → render rời ở
    // cuối như cũ, tránh mất dữ liệu. renderedPhoneme chặn render trùng.
    if (!renderedPhoneme) html += phonemeErrorsHtml(phoneme);
    return html;
}

// Milliseconds → human time. <1s stays in ms; otherwise seconds (or m:ss).
function fmtMs(ms) {
    const n = Number(ms) || 0;
    if (n < 1000) return `${n}ms`;
    const sec = n / 1000;
    if (sec < 60) return `${sec.toFixed(1)}s`;
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}m${String(s).padStart(2, '0')}s`;
}

// Wall-clock a single file took, pulled from its telemetry (camelCase wrapper
// key set by the API, falling back to the engine's snake_case total).
function itemProcessingMs(result) {
    const tel = (result && result.telemetry) || {};
    return tel.totalProcessingTimeMs ?? tel.pipeline_total_ms ?? null;
}

function telemetryHtml(telemetry) {
    const tel = telemetry || {};
    const steps = tel.step_timings_ms || {};
    const tiles = [
        { name: 'ASR Backend', value: tel.asr_backend_used || '--' },
        { name: 'Total Time', value: (tel.pipeline_total_ms || 0) + 'ms' },
        { name: 'Transcription', value: (tel.transcription_time_ms || 0) + 'ms' },
        { name: 'Scoring', value: (steps.scoring || 0) + 'ms' },
        { name: 'Features', value: (steps.features || 0) + 'ms' },
        { name: 'Phoneme', value: (steps.phoneme || 0) + 'ms' },
    ];
    return `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:0.5rem;">`
        + tiles.map(t => `<div class="feature-item"><div class="value" style="font-size:1rem;">${escapeHtml(t.value)}</div><div class="name">${t.name}</div></div>`).join('')
        + `</div>`;
}

// ── Single result ─────────────────────────────────────────────────────
function showSingleResult(data) {
    lastSingleData = data;
    const resultDiv = document.getElementById('result');
    const cfg = examConfig(data.exam);
    const pronunciationOnly = !!data.pronunciation_only;
    document.getElementById('score-label').textContent =
        pronunciationOnly ? 'Chỉ chấm phát âm (chưa có đề)' : cfg.overallLabel;
    document.getElementById('overall-score').textContent =
        pronunciationOnly ? '--' : (data.scores?.[cfg.scoreField] ?? '--');
    document.getElementById('transcript').textContent = data.transcript || 'No transcript available';
    document.getElementById('features-grid').innerHTML = featureGridHtml(data.features || {});
    document.getElementById('scores-breakdown').innerHTML = scoresBreakdownHtml(
        data.scores, data.exam, data.phoneme,
        { pronunciationOnly, notice: data.notice });
    document.getElementById('feedback').textContent =
        data.scores?.summary_feedback
        || (pronunciationOnly ? (data.notice || '') : 'No feedback available');
    document.getElementById('telemetry').innerHTML = telemetryHtml(data.telemetry);
    resultDiv.classList.add('visible');
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

// ── Batch result ──────────────────────────────────────────────────────
function showBatchResult(data) {
    lastBatchData = data;
    const cfg = examConfig(data.exam);
    const wrap = document.getElementById('batch-result');
    // Số bài chỉ chấm phát âm do thiếu đề (để báo gộp, khỏi mở từng item).
    const pronOnlyCount = (data.results || [])
        .filter(it => it.result && it.result.pronunciation_only).length;
    const pronOnlyNote = pronOnlyCount
        ? `<div class="status-bar info" style="justify-content:center;margin-top:0.5rem;">
               <span>⚠️ ${pronOnlyCount} bài chỉ chấm phát âm do thiếu đề bài.</span>
           </div>`
        : '';
    const batchTime = data.total_processing_time_ms != null
        ? ` · ⏱ ${fmtMs(data.total_processing_time_ms)}${data.concurrency > 1 ? ` (×${data.concurrency})` : ''}`
        : '';
    document.getElementById('batch-summary').innerHTML = `
        <div class="status-bar ${data.failed ? 'info' : 'success'}" style="justify-content:center;">
            <span>${data.succeeded}/${data.count} graded${data.failed ? ` · ${data.failed} failed` : ''} · exam: ${escapeHtml(cfg.label)} · type: ${escapeHtml(data.question_type)} · mode: ${escapeHtml(data.mode_requested)}${batchTime}</span>
        </div>${pronOnlyNote}`;

    const results = (data.results || []).slice().sort((a, b) => a.index - b.index);
    document.getElementById('batch-results-list').innerHTML = results.map(item => {
        if (item.error) {
            return `<div class="batch-result">
                <div class="filename">📄 ${escapeHtml(item.audio_filename)}</div>
                <div class="batch-error">❌ ${escapeHtml(item.error)}</div>
            </div>`;
        }
        const r = item.result || {};
        const pronOnly = !!r.pronunciation_only;
        const score = pronOnly ? '🔊' : (r.scores?.[cfg.scoreField] ?? '--');
        const feedback = r.scores?.summary_feedback || (pronOnly ? r.notice : '');
        const ms = itemProcessingMs(r);
        const timeTag = ms != null
            ? `<span style="color:#888;font-size:0.85rem;white-space:nowrap;">⏱ ${fmtMs(ms)}</span>`
            : '';
        return `<details class="batch-result">
            <summary style="cursor:pointer;display:flex;align-items:center;gap:0.75rem;list-style:none;">
                <span class="batch-score" style="margin:0;" title="${pronOnly ? 'Chỉ chấm phát âm' : ''}">${score}</span>
                <span class="filename" style="margin:0;flex:1;">📄 ${escapeHtml(item.audio_filename)}</span>
                ${timeTag}
                <span style="color:#888;font-size:0.85rem;">▼ details</span>
            </summary>
            <div style="margin-top:0.85rem;">
                <div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Transcript</div>
                <p style="color:#555;line-height:1.5;white-space:pre-wrap;">${escapeHtml(r.transcript || '(empty)')}</p>
                <div class="features-grid" style="margin-top:0.85rem;">${featureGridHtml(r.features || {})}</div>
                <div style="margin-top:0.85rem;">${scoresBreakdownHtml(r.scores, r.exam ?? data.exam, r.phoneme, { pronunciationOnly: pronOnly, notice: r.notice })}</div>
                ${feedback ? `<div style="font-weight:600;color:#333;margin:0.85rem 0 0.3rem;">Feedback</div><p style="color:#555;line-height:1.6;white-space:pre-wrap;">${escapeHtml(feedback)}</p>` : ''}
            </div>
        </details>`;
    }).join('');

    wrap.classList.add('visible');
    wrap.scrollIntoView({ behavior: 'smooth' });
}

function closeResult() {
    document.getElementById('result').classList.remove('visible');
}

function closeBatchResult() {
    document.getElementById('batch-result').classList.remove('visible');
}

// ── CSV export (single result) ────────────────────────────────────────
// One row per audio file. Suitable for opening in Excel.
const CSV_COLUMNS = [
    'index', 'filename', 'status', 'exam',
    'estimated_toeic_score', 'estimated_ielts_band',
    'task_completion', 'content_relevance', 'wpm', 'words',
    'duration_sec', 'asr_confidence', 'coverage', 'word_accuracy',
    'transcript', 'summary_feedback', 'error',
];

function csvCell(value) {
    const s = String(value ?? '');
    // Quote if it contains comma, quote, or newline; double-up inner quotes.
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

// Turn one batch item (or a single-result pseudo-item) into a CSV row object.
function resultRow(item, fallbackExam) {
    if (item.error) {
        return {
            index: item.index,
            filename: item.audio_filename,
            status: 'error',
            error: item.error,
        };
    }
    const r = item.result || {};
    const f = r.features || {};
    const s = r.scores || {};
    const acc = f.accuracy_metrics;
    return {
        index: item.index,
        filename: item.audio_filename,
        status: 'ok',
        exam: r.exam ?? fallbackExam ?? '',
        estimated_toeic_score: s.estimated_toeic_score ?? '',
        estimated_ielts_band: s.estimated_ielts_band ?? '',
        task_completion: s.task_completion ?? '',
        content_relevance: s.content_relevance ?? '',
        wpm: f.speech_rate_wpm != null ? Math.round(f.speech_rate_wpm) : '',
        words: f.word_count ?? '',
        duration_sec: f.audio_duration_sec != null ? f.audio_duration_sec.toFixed(1) : '',
        asr_confidence: f.avg_word_probability != null ? f.avg_word_probability.toFixed(4) : '',
        coverage: acc ? acc.coverage : '',
        word_accuracy: acc && acc.wer != null ? (1 - acc.wer).toFixed(4) : '',
        transcript: r.transcript ?? '',
        summary_feedback: s.summary_feedback ?? '',
        error: '',
    };
}

function buildCsv(rows) {
    const lines = [CSV_COLUMNS.join(',')];
    for (const row of rows) {
        lines.push(CSV_COLUMNS.map(c => csvCell(row[c])).join(','));
    }
    // Prefix BOM so Excel reads UTF-8 (Vietnamese feedback) correctly.
    return '﻿' + lines.join('\r\n');
}

// yyyy-mm-dd-hh-mm-ss, safe for filenames.
function fileStamp() {
    return new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
}

function downloadBlob(blob, filename) {
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(objectUrl);
}

function exportSingleCsv() {
    if (!lastSingleData) {
        alert('No result to export. Grade a file first.');
        return;
    }
    const cfg = examConfig(lastSingleData.exam);
    // Wrap the single result in the same shape resultRow() expects for a batch item.
    const item = {
        index: 1,
        audio_filename: lastSingleData.audio_filename || lastSingleFilename || 'recording',
        result: lastSingleData,
    };
    const row = resultRow(item, lastSingleData.exam);
    const blob = new Blob([buildCsv([row])], { type: 'text/csv;charset=utf-8;' });
    downloadBlob(blob, `${cfg.label.toLowerCase()}-result-${fileStamp()}.csv`);
}

// ── Printable report (single result → Print / Save as PDF) ────────────
function reportCriteriaHtml(scores, cfg) {
    const criteria = Array.isArray(scores.criteria) ? scores.criteria : [];
    if (!criteria.length) return '';
    const items = criteria.map(c => {
        const suggestions = (c.suggestions || []).map(x => `<li>${escapeHtml(x)}</li>`).join('');
        return `<div class="crit">
            <div class="crit-head"><span>${escapeHtml(c.criterion)}</span>
                <span class="badge">${escapeHtml(c.score)}/${cfg.criterionMax}</span></div>
            <div class="just">${escapeHtml(c.justification)}</div>
            ${suggestions ? `<ul>${suggestions}</ul>` : ''}
        </div>`;
    }).join('');
    return `<h2>Scores Breakdown</h2>${items}`;
}

// Shared CSS for the printable single / batch reports (kept identical so a
// class export looks like the individual ones).
function reportStyles() {
    return `
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: #1f2937; margin: 2rem; line-height: 1.5; }
  h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
  h2 { font-size: 1.1rem; margin: 1.5rem 0 0.6rem; border-bottom: 2px solid #4f46e5; padding-bottom: 0.25rem; }
  .meta { color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem; }
  .overall { display: flex; align-items: baseline; gap: 0.5rem; background: #eef2ff; border-radius: 10px; padding: 1rem 1.25rem; margin: 1rem 0; }
  .overall .big { font-size: 2.2rem; font-weight: 700; color: #4f46e5; }
  .overall .lbl { color: #4338ca; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; }
  td { padding: 0.4rem 0; border-bottom: 1px solid #e5e7eb; }
  td:last-child { text-align: right; font-weight: 600; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 0.5rem; }
  .tile { background: #f9fafb; border-radius: 8px; padding: 0.6rem; text-align: center; }
  .tval { font-size: 1.1rem; font-weight: 700; color: #111827; }
  .tname { font-size: 0.75rem; color: #6b7280; }
  .crit { background: #f9fafb; border-radius: 8px; padding: 0.85rem; margin-bottom: 0.6rem; }
  .crit-head { display: flex; justify-content: space-between; align-items: center; font-weight: 600; margin-bottom: 0.35rem; }
  .badge { background: #4f46e5; color: #fff; border-radius: 6px; padding: 0.1rem 0.55rem; font-size: 0.85rem; }
  .just { color: #4b5563; font-size: 0.92rem; }
  ul { margin: 0.5rem 0 0 1.1rem; color: #4338ca; font-size: 0.9rem; }
  p.body { white-space: pre-wrap; color: #374151; }
  /* ── Pronunciation detail (phoneme) — mirror of styles.css for the popup ── */
  .phoneme-detail { margin-top: 1.5rem; background: #fff7ed; border-radius: 8px; padding: 0.85rem; }
  .phoneme-detail__title { font-weight: 600; color: #333; margin-bottom: 0.3rem; }
  .phoneme-legend { color: #888; font-size: 0.8rem; margin-bottom: 0.5rem; }
  .phoneme-words { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; }
  .phoneme-word { background: #fff; border: 1px solid #fed7aa; border-radius: 8px; padding: 0.4rem 0.6rem; display: flex; flex-direction: column; gap: 0.15rem; }
  .phoneme-word--err { border-color: #fdba74; background: #fffbeb; }
  .phoneme-word__text { font-weight: 600; color: #333; font-size: 0.9rem; }
  .phoneme-ipa { color: #444; font-size: 0.95rem; }
  .phoneme-sym { letter-spacing: 0.03em; display: inline-block; }
  .phoneme-stress { color: #4338ca; font-weight: 700; font-family: Arial, sans-serif; font-size: 1.35em; line-height: 1; vertical-align: 0.05em; margin-right: 0.02em; }
  .phoneme-sym--bad { color: #b91c1c; font-weight: 700; }
  .phoneme-sym--missing { color: #b91c1c; font-weight: 700; text-decoration: line-through; }
  .phoneme-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.3rem; }
  .phoneme-table th, .phoneme-table td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid #fed7aa; vertical-align: top; }
  .phoneme-table th { color: #92400e; font-size: 0.8rem; font-weight: 600; }
  .phoneme-table__word { font-weight: 600; color: #333; }
  .phoneme-detail td:last-child { text-align: left; font-weight: 400; }
  /* ── Batch report extras ── */
  .overview thead th { text-align: left; font-size: 0.8rem; color: #6b7280; border-bottom: 2px solid #4f46e5; padding: 0.4rem 0.5rem; }
  .overview td { padding: 0.45rem 0.5rem; vertical-align: top; }
  .overview .col-idx { color: #9ca3af; width: 2.2rem; }
  .overview .col-score { text-align: right; font-weight: 700; color: #4f46e5; white-space: nowrap; }
  .overview .col-time { text-align: right; color: #6b7280; font-size: 0.85rem; white-space: nowrap; }
  .overview .col-fb { font-weight: 400; color: #4b5563; font-size: 0.85rem; }
  .overview .err { color: #b91c1c; font-weight: 600; }
  .file-head { background: #4f46e5; color: #fff; border-radius: 8px; padding: 0.6rem 1rem; margin: 0 0 0.5rem; font-weight: 700; font-size: 1.15rem; }
  @media print { body { margin: 1rem; } h2 { break-after: avoid; } .crit, .tile, .phoneme-word, .phoneme-table tr { break-inside: avoid; }
    section.file { break-before: page; } section.file:first-of-type { break-before: auto; } }`;
}

function printSingleReport() {
    if (!lastSingleData) {
        alert('No result to export. Grade a file first.');
        return;
    }
    const data = lastSingleData;
    const cfg = examConfig(data.exam);
    const s = data.scores || {};
    const f = data.features || {};
    const overall = s[cfg.scoreField];
    const filename = data.audio_filename || lastSingleFilename || 'recording';

    const featuresHtml = featureTiles(f).map(t =>
        `<div class="tile"><div class="tval">${escapeHtml(t.value)}</div><div class="tname">${escapeHtml(t.name)}</div></div>`
    ).join('');

    const summaryRows = [
        ['Task Completion', s.task_completion],
        ['Content Relevance', s.content_relevance],
    ].filter(([, v]) => v != null && v !== '')
     .map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(v)}</td></tr>`).join('');

    const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>${escapeHtml(cfg.label)} Speaking Report — ${escapeHtml(filename)}</title>
<style>${reportStyles()}</style></head>
<body>
  <h1>${escapeHtml(cfg.label)} Speaking Report</h1>
  <div class="meta">File: ${escapeHtml(filename)} · Generated ${escapeHtml(new Date().toLocaleString())}</div>

  ${data.pronunciation_only
    ? `<div class="overall"><span class="lbl">⚠️ ${escapeHtml(data.notice || 'Chỉ chấm phát âm (chưa có đề bài).')}</span></div>`
    : `<div class="overall">
    <span class="big">${escapeHtml(overall ?? '--')}</span>
    <span class="lbl">${escapeHtml(cfg.overallLabel)} (max ${cfg.overallMax})</span>
  </div>`}

  ${summaryRows ? `<table>${summaryRows}</table>` : ''}

  <h2>Transcript</h2>
  <p class="body">${escapeHtml(data.transcript || 'No transcript available')}</p>

  <h2>Features</h2>
  <div class="tiles">${featuresHtml}</div>

  ${reportCriteriaHtml(s, cfg)}

  ${phonemeErrorsHtml(data.phoneme) /* block carries its own title + accuracy; non-collapsible → expanded in print */}

  ${s.score_rationale ? `<h2>Score Rationale</h2><p class="body">${escapeHtml(s.score_rationale)}</p>` : ''}

  <h2>Feedback</h2>
  <p class="body">${escapeHtml(s.summary_feedback || 'No feedback available')}</p>

  <script>window.onload = function () { window.print(); };<\/script>
</body></html>`;

    const win = window.open('', '_blank');
    if (!win) {
        alert('Popup blocked. Allow popups for this site to print the report.');
        return;
    }
    win.document.write(html);
    win.document.close();
}

// ── Printable report (batch results → Print / Save as PDF) ────────────
// An overview table (one row per file) followed by a full per-file report,
// each file on its own page. Replaces the old CSV export.
function printBatchReport() {
    if (!lastBatchData || !Array.isArray(lastBatchData.results) || lastBatchData.results.length === 0) {
        alert('No batch results to export. Grade a batch first.');
        return;
    }
    const data = lastBatchData;
    const cfg = examConfig(data.exam);
    const results = data.results.slice().sort((a, b) => a.index - b.index);

    // Overview table — at-a-glance score + time + feedback per file.
    const overviewRows = results.map(item => {
        if (item.error) {
            return `<tr><td class="col-idx">${item.index}</td>
                <td>${escapeHtml(item.audio_filename)}</td>
                <td class="col-score err">error</td>
                <td class="col-time">—</td>
                <td class="col-fb err">${escapeHtml(item.error)}</td></tr>`;
        }
        const r = item.result || {};
        const pronOnly = !!r.pronunciation_only;
        const score = pronOnly ? '🔊 pron.' : escapeHtml(r.scores?.[cfg.scoreField] ?? '--');
        const fb = r.scores?.summary_feedback || (pronOnly ? r.notice : '') || '';
        const ms = itemProcessingMs(r);
        return `<tr><td class="col-idx">${item.index}</td>
            <td>${escapeHtml(item.audio_filename)}</td>
            <td class="col-score">${score}</td>
            <td class="col-time">${ms != null ? fmtMs(ms) : '—'}</td>
            <td class="col-fb">${escapeHtml(fb)}</td></tr>`;
    }).join('');

    // Per-file detail sections — same layout as the single report.
    const detailSections = results.map(item => {
        const head = `<div class="file-head">#${item.index} · ${escapeHtml(item.audio_filename)}</div>`;
        if (item.error) {
            return `<section class="file">${head}
                <p class="body err">❌ ${escapeHtml(item.error)}</p></section>`;
        }
        const r = item.result || {};
        const s = r.scores || {};
        const f = r.features || {};
        const pronOnly = !!r.pronunciation_only;
        const overall = s[cfg.scoreField];
        const featuresHtml = featureTiles(f).map(t =>
            `<div class="tile"><div class="tval">${escapeHtml(t.value)}</div><div class="tname">${escapeHtml(t.name)}</div></div>`
        ).join('');
        const summaryRows = [
            ['Task Completion', s.task_completion],
            ['Content Relevance', s.content_relevance],
        ].filter(([, v]) => v != null && v !== '')
         .map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(v)}</td></tr>`).join('');
        return `<section class="file">
            ${head}
            ${pronOnly
                ? `<div class="overall"><span class="lbl">⚠️ ${escapeHtml(r.notice || 'Chỉ chấm phát âm (chưa có đề bài).')}</span></div>`
                : `<div class="overall"><span class="big">${escapeHtml(overall ?? '--')}</span><span class="lbl">${escapeHtml(cfg.overallLabel)} (max ${cfg.overallMax})</span></div>`}
            ${summaryRows ? `<table>${summaryRows}</table>` : ''}
            <h2>Transcript</h2>
            <p class="body">${escapeHtml(r.transcript || 'No transcript available')}</p>
            <h2>Features</h2>
            <div class="tiles">${featuresHtml}</div>
            ${reportCriteriaHtml(s, cfg)}
            ${phonemeErrorsHtml(r.phoneme)}
            ${s.score_rationale ? `<h2>Score Rationale</h2><p class="body">${escapeHtml(s.score_rationale)}</p>` : ''}
            <h2>Feedback</h2>
            <p class="body">${escapeHtml(s.summary_feedback || (pronOnly ? r.notice : '') || 'No feedback available')}</p>
        </section>`;
    }).join('');

    const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>${escapeHtml(cfg.label)} Batch Speaking Report</title>
<style>${reportStyles()}</style></head>
<body>
  <h1>${escapeHtml(cfg.label)} Batch Speaking Report</h1>
  <div class="meta">${data.succeeded}/${data.count} graded${data.failed ? ` · ${data.failed} failed` : ''} · type: ${escapeHtml(data.question_type)} · mode: ${escapeHtml(data.mode_requested)}${data.total_processing_time_ms != null ? ` · ⏱ ${fmtMs(data.total_processing_time_ms)}${data.concurrency > 1 ? ` (×${data.concurrency})` : ''}` : ''} · Generated ${escapeHtml(new Date().toLocaleString())}</div>

  <h2>Overview</h2>
  <table class="overview">
    <thead><tr><th class="col-idx">#</th><th>File</th><th class="col-score">${escapeHtml(cfg.overallLabel)}</th><th class="col-time">Time</th><th class="col-fb">Feedback</th></tr></thead>
    <tbody>${overviewRows}</tbody>
  </table>

  ${detailSections}

  <script>window.onload = function () { window.print(); };<\/script>
</body></html>`;

    const win = window.open('', '_blank');
    if (!win) {
        alert('Popup blocked. Allow popups for this site to print the report.');
        return;
    }
    win.document.write(html);
    win.document.close();
}

// Show any recordings already saved on this device from a previous session.
renderSavedRecordings();
