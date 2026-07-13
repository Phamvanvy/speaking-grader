'use strict';

// API URL key + shared exam config + accent state + cross-render state (object
// URL phát lại). Nạp ĐẦU TIÊN: chứa mọi const/state mà file sau dùng ở top-level.

const API_URL_KEY = 'toeic-grader-api-url';

// ── Lịch sử chấm bài (server-side) ────────────────────────────────────
// "User" = uuid ẩn danh sinh 1 lần cho trình duyệt này, gửi kèm mỗi request chấm
// để server tách lịch sử theo máy/trình duyệt. KHÔNG phải auth — chỉ cách ly mềm.
const USER_ID_KEY = 'speaking-grader-user-id';
function getUserId() {
    let id = localStorage.getItem(USER_ID_KEY);
    if (!id) {
        // Fallback uuid-shaped khi thiếu crypto.randomUUID (context không phải
        // secure) — server validate [A-Za-z0-9_-]{1,64} nên format phải chuẩn.
        id = (crypto.randomUUID) ? crypto.randomUUID()
            : ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
                (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16));
        localStorage.setItem(USER_ID_KEY, id);
    }
    return id;
}

// Opt-out privacy: tắt → không gửi user_id → server không lưu gì (audio là dữ
// liệu nhạy cảm). Default BẬT. Checkbox nằm ở đầu tab Lịch sử.
const HISTORY_OPT_KEY = 'speaking-grader-save-history';
function historySaveEnabled() {
    return localStorage.getItem(HISTORY_OPT_KEY) !== 'false';
}
function setHistorySaveEnabled(on) {
    localStorage.setItem(HISTORY_OPT_KEY, on ? 'true' : 'false');
}

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

// Accent tham chiếu phát âm: 'default' (mặc định) = tự chấp nhận cả Anh-Anh lẫn Anh-Mỹ
// (gửi lên backend lúc CHẤM → coda /r/ non-rhotic không bị trừ điểm); 'gb' = Anh-Anh,
// 'us' = Anh-Mỹ. 'gb'/'us' chỉ đổi HIỂN THỊ IPA. Lưu localStorage để nhớ giữa các lần mở
// trang. LƯU Ý: accent ảnh hưởng ĐIỂM ở thời điểm gửi /grade; đổi dropdown SAU khi có kết
// quả chỉ render lại hiển thị (không chấm lại).
const ACCENT_KEY = 'pron_accent';
const VALID_ACCENTS = ['default', 'gb', 'us'];
let currentAccent = VALID_ACCENTS.includes(localStorage.getItem(ACCENT_KEY))
    ? localStorage.getItem(ACCENT_KEY) : 'default';

// Đổi accent rồi render lại KẾT QUẢ HIỆN CÓ từ dữ liệu gốc (không chấm lại). Vì
// transform tạo bản clone, lastSingleData/lastBatchData luôn nguyên vẹn.
function setAccent(v) {
    currentAccent = VALID_ACCENTS.includes(v) ? v : 'default';
    localStorage.setItem(ACCENT_KEY, currentAccent);
    // Đồng bộ MỌI selector accent đang có (form chính + panel) để hai chỗ không lệch.
    document.querySelectorAll('.accent-select').forEach(s => { s.value = currentAccent; });
    if (lastSingleData) showSingleResult(lastSingleData);
    if (lastBatchData) showBatchResult(lastBatchData);
}

// Đồng bộ giá trị ban đầu cho selector trên form chính theo lựa chọn đã lưu.
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.accent-select').forEach(s => { s.value = currentAccent; });
});

// Delegated: bắt mọi <select class="accent-select"> (panel dựng lại mỗi lần render,
// có thể lồng trong <details>), gắn 1 lần ở document.
document.addEventListener('change', e => {
    const t = e.target;
    if (t instanceof HTMLSelectElement && t.classList.contains('accent-select')) {
        setAccent(t.value);
    }
});

// Holds the most recent /grade-batch response so "Print / PDF" can rebuild a report.
let lastBatchData = null;

// Files sent in the most recent batch, kept around (in-memory, index-aligned with
// `results[].index` from the API) purely so the result panel can offer a download
// link — no need to re-fetch from the server since the audio never left the browser.
let lastBatchFiles = [];

// Holds the most recent single /grade response (+ the file name it came from)
// so "Print / PDF" can rebuild a report.
let lastSingleData = null;
let lastSingleFilename = '';

// File audio của lần chấm single gần nhất + object URL phát lại (lazy). Giữ lại để
// nút "nghe lại" ở Pronunciation detail phát đúng đoạn audio của từng từ — Blob nằm
// sẵn ở client nên không cần server lưu audio. URL cũ được revoke khi chấm file mới.
let lastSingleFile = null;
let lastSinglePlaybackUrl = null;

// Tạo (1 lần) object URL cho file đang xét → dùng phát lại đoạn audio từng từ.
// Revoke URL của file trước để không rò Blob khi đổi file chấm.
function setPlaybackFile(file) {
    if (lastSinglePlaybackUrl) {
        URL.revokeObjectURL(lastSinglePlaybackUrl);
        lastSinglePlaybackUrl = null;
    }
    lastSingleFile = file || null;
}

function playbackUrl() {
    if (!lastSingleFile) return null;
    if (!lastSinglePlaybackUrl) lastSinglePlaybackUrl = URL.createObjectURL(lastSingleFile);
    return lastSinglePlaybackUrl;
}

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
