'use strict';

// Upload file + ghi âm (MediaRecorder) + lưu/khôi phục bản ghi (IndexedDB) + UI.

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

// ── File list rendering (di chuyển từ app.js cũ — đi cùng upload handling) ──
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

// Show any recordings already saved on this device from a previous session.
renderSavedRecordings();
