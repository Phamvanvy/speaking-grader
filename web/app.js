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
        questionTypes: [
            { value: '', label: 'Auto-detect' },
            { value: 'read_aloud', label: 'Read Aloud' },
            { value: 'describe_picture', label: 'Describe Picture' },
            { value: 'respond_questions', label: 'Respond to Questions' },
            { value: 'respond_with_info', label: 'Respond with Info' },
            { value: 'express_opinion', label: 'Express Opinion' },
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
            { value: 'part1_interview', label: 'Part 1 — Interview' },
            { value: 'part2_long_turn', label: 'Part 2 — Long turn (cue card)' },
            { value: 'part3_discussion', label: 'Part 3 — Discussion' },
        ],
    },
};

function examConfig(exam) {
    return EXAM_CONFIG[exam] || EXAM_CONFIG.toeic;
}

// Holds the most recent /grade-batch response so "Export CSV" can rebuild it.
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

fileLabel.addEventListener('click', (e) => {
    if (e.target !== fileInput) {
        fileInput.click();
    }
});

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
    auto: 'Starts in Default; auto-escalates to Review (better ASR + phoneme '
        + 'analysis) when confidence/coverage is low. Recommended.',
    default: 'Balanced ASR (faster-whisper). Phoneme analysis follows server config.',
    fast: 'Fastest ASR, phoneme analysis OFF → pronunciation scored more leniently, '
        + 'so scores may be HIGHER than Review. Quick estimate only.',
    review: 'Most accurate: best ASR (WhisperX) + phoneme analysis ON. '
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

examSelect.addEventListener('change', populateQuestionTypes);
populateQuestionTypes();

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
    const referenceText = document.getElementById('reference-text').value;
    if (referenceText) formData.append('text', referenceText);

    const promptText = document.getElementById('prompt-text').value;
    if (promptText) formData.append('prompt', promptText);

    formData.append('exam', examSelect.value);

    const questionType = questionTypeSelect.value;
    if (questionType) formData.append('question_type', questionType);

    formData.append('mode', document.getElementById('mode').value);

    const feedbackLang = document.getElementById('feedback-lang').value;
    if (feedbackLang) formData.append('feedback_lang', feedbackLang);

    const expectedDuration = document.getElementById('expected-duration').value;
    if (expectedDuration) formData.append('expected_duration_sec', expectedDuration);

    // Ảnh đề bài (Describe Picture) — dùng chung cho cả single & batch.
    const imageFile = imageInput.files[0];
    if (imageFile) formData.append('image', imageFile);

    formData.append('no_ai', document.getElementById('no-ai').checked);
}

// Grade — routes to /grade (1 file) or /grade-batch (≥2 files).
async function grade() {
    const url = document.getElementById('api-url').value.replace(/\/$/, '');
    const files = Array.from(fileInput.files);

    if (files.length === 0) {
        alert('Please select at least one audio file');
        return;
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

function scoresBreakdownHtml(scores, exam) {
    if (!scores) {
        return '<p style="color:#666;">No AI scoring (ASR-only or skipped by gating).</p>';
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

    const criteria = Array.isArray(scores.criteria) ? scores.criteria : [];
    if (criteria.length) {
        html += '<div style="margin-top:1rem;">' + criteria.map(c => {
            const suggestions = (c.suggestions || []).map(s => `<li>${escapeHtml(s)}</li>`).join('');
            return `
                <div style="background:#f9fafb;border-radius:8px;padding:0.85rem;margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                        <span style="font-weight:600;color:#333;">${escapeHtml(c.criterion)}</span>
                        <span style="background:#4f46e5;color:#fff;border-radius:6px;padding:0.1rem 0.55rem;font-weight:600;font-size:0.85rem;">${escapeHtml(c.score)}/${cfg.criterionMax}</span>
                    </div>
                    <div style="color:#555;line-height:1.5;font-size:0.92rem;">${escapeHtml(c.justification)}</div>
                    ${suggestions ? `<ul style="margin:0.5rem 0 0 1.1rem;color:#4338ca;font-size:0.9rem;">${suggestions}</ul>` : ''}
                </div>`;
        }).join('') + '</div>';
    }
    if (scores.score_rationale) {
        html += `<div style="margin-top:0.75rem;">
            <div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Score Rationale</div>
            <p style="color:#555;line-height:1.6;white-space:pre-wrap;">${escapeHtml(scores.score_rationale)}</p>
        </div>`;
    }
    return html;
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
    document.getElementById('score-label').textContent = cfg.overallLabel;
    document.getElementById('overall-score').textContent = data.scores?.[cfg.scoreField] ?? '--';
    document.getElementById('transcript').textContent = data.transcript || 'No transcript available';
    document.getElementById('features-grid').innerHTML = featureGridHtml(data.features || {});
    document.getElementById('scores-breakdown').innerHTML = scoresBreakdownHtml(data.scores, data.exam);
    document.getElementById('feedback').textContent = data.scores?.summary_feedback || 'No feedback available';
    document.getElementById('telemetry').innerHTML = telemetryHtml(data.telemetry);
    resultDiv.classList.add('visible');
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

// ── Batch result ──────────────────────────────────────────────────────
function showBatchResult(data) {
    lastBatchData = data;
    const cfg = examConfig(data.exam);
    const wrap = document.getElementById('batch-result');
    document.getElementById('batch-summary').innerHTML = `
        <div class="status-bar ${data.failed ? 'info' : 'success'}" style="justify-content:center;">
            <span>${data.succeeded}/${data.count} graded${data.failed ? ` · ${data.failed} failed` : ''} · exam: ${escapeHtml(cfg.label)} · type: ${escapeHtml(data.question_type)} · mode: ${escapeHtml(data.mode_requested)}</span>
        </div>`;

    const results = (data.results || []).slice().sort((a, b) => a.index - b.index);
    document.getElementById('batch-results-list').innerHTML = results.map(item => {
        if (item.error) {
            return `<div class="batch-result">
                <div class="filename">📄 ${escapeHtml(item.audio_filename)}</div>
                <div class="batch-error">❌ ${escapeHtml(item.error)}</div>
            </div>`;
        }
        const r = item.result || {};
        const score = r.scores?.[cfg.scoreField] ?? '--';
        const feedback = r.scores?.summary_feedback;
        return `<details class="batch-result">
            <summary style="cursor:pointer;display:flex;align-items:center;gap:0.75rem;list-style:none;">
                <span class="batch-score" style="margin:0;">${score}</span>
                <span class="filename" style="margin:0;flex:1;">📄 ${escapeHtml(item.audio_filename)}</span>
                <span style="color:#888;font-size:0.85rem;">▼ details</span>
            </summary>
            <div style="margin-top:0.85rem;">
                <div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Transcript</div>
                <p style="color:#555;line-height:1.5;white-space:pre-wrap;">${escapeHtml(r.transcript || '(empty)')}</p>
                <div class="features-grid" style="margin-top:0.85rem;">${featureGridHtml(r.features || {})}</div>
                <div style="margin-top:0.85rem;">${scoresBreakdownHtml(r.scores, r.exam ?? data.exam)}</div>
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

// ── CSV export (shared by single & batch) ─────────────────────────────
// One row per audio file. Suitable for opening a whole class in Excel.
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

function exportBatchCsv() {
    if (!lastBatchData || !Array.isArray(lastBatchData.results) || lastBatchData.results.length === 0) {
        alert('No batch results to export. Grade a batch first.');
        return;
    }
    const cfg = examConfig(lastBatchData.exam);
    const rows = lastBatchData.results
        .slice()
        .sort((a, b) => a.index - b.index)
        .map(item => resultRow(item, lastBatchData.exam));
    const blob = new Blob([buildCsv(rows)], { type: 'text/csv;charset=utf-8;' });
    downloadBlob(blob, `${cfg.label.toLowerCase()}-batch-${rows.length}-${fileStamp()}.csv`);
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
<style>
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
  @media print { body { margin: 1rem; } h2 { break-after: avoid; } .crit, .tile { break-inside: avoid; } }
</style></head>
<body>
  <h1>${escapeHtml(cfg.label)} Speaking Report</h1>
  <div class="meta">File: ${escapeHtml(filename)} · Generated ${escapeHtml(new Date().toLocaleString())}</div>

  <div class="overall">
    <span class="big">${escapeHtml(overall ?? '--')}</span>
    <span class="lbl">${escapeHtml(cfg.overallLabel)} (max ${cfg.overallMax})</span>
  </div>

  ${summaryRows ? `<table>${summaryRows}</table>` : ''}

  <h2>Transcript</h2>
  <p class="body">${escapeHtml(data.transcript || 'No transcript available')}</p>

  <h2>Features</h2>
  <div class="tiles">${featuresHtml}</div>

  ${reportCriteriaHtml(s, cfg)}

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

// Show any recordings already saved on this device from a previous session.
renderSavedRecordings();
