'use strict';

// Image upload + grading mode note + exam/question-type selector + dark mode + health.

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

// Đồng bộ khối "Gợi ý bài mẫu" (định nghĩa ở suggest.js, nạp SAU form.js → gọi
// có kiểm tra tồn tại; lần init đầu do chính suggest.js tự chạy khi nạp).
function syncSuggestUI() {
    if (typeof updateSuggestUI === 'function') updateSuggestUI();
}

// Accent tham chiếu (default/gb/us) chỉ có nghĩa với tiếng ANH → ẩn khi chấm
// TOPIK (tiếng Hàn). Chỉ ẩn UI — giá trị currentAccent giữ nguyên cho lần quay
// lại TOEIC/IELTS; backend bỏ qua accent khi lang=ko nên gửi kèm cũng vô hại.
function syncAccentVisibility() {
    const group = document.getElementById('accent-group');
    if (group) group.classList.toggle('hidden', examConfig(examSelect.value).lang === 'ko');
}

examSelect.addEventListener('change', () => {
    populateQuestionTypes();
    syncConditionalFields();
    syncAccentVisibility();
    syncSuggestUI();
});
questionTypeSelect.addEventListener('change', () => {
    syncConditionalFields();
    syncSuggestUI();
});
populateQuestionTypes();
syncConditionalFields();
syncAccentVisibility();

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

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
    const url = apiBase();
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
