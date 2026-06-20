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

fileLabel.addEventListener('click', (e) => {
    if (e.target !== fileInput) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', renderFileList);

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
    fileNameDisplay.innerHTML = header + files.map(f => `
        <div class="file-item">
            <span class="name">📄 ${escapeHtml(f.name)}</span>
        </div>
    `).join('') + `
        <button class="btn btn-secondary" onclick="clearFile(event)" style="margin-top:0.5rem;width:auto;padding:0.4rem 0.9rem;">Clear</button>
    `;
}

function clearFile(e) {
    e.stopPropagation();
    e.preventDefault();
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

// ── CSV export of batch scores ────────────────────────────────────────
// One row per audio file. Suitable for opening a whole class in Excel.
function csvCell(value) {
    const s = String(value ?? '');
    // Quote if it contains comma, quote, or newline; double-up inner quotes.
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function exportBatchCsv() {
    if (!lastBatchData || !Array.isArray(lastBatchData.results) || lastBatchData.results.length === 0) {
        alert('No batch results to export. Grade a batch first.');
        return;
    }

    const cfg = examConfig(lastBatchData.exam);
    const columns = [
        'index', 'filename', 'status', 'exam',
        'estimated_toeic_score', 'estimated_ielts_band',
        'task_completion', 'content_relevance', 'wpm', 'words',
        'duration_sec', 'asr_confidence', 'coverage', 'word_accuracy',
        'transcript', 'summary_feedback', 'error',
    ];

    const rows = lastBatchData.results
        .slice()
        .sort((a, b) => a.index - b.index)
        .map(item => {
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
                exam: r.exam ?? lastBatchData.exam ?? '',
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
        });

    const lines = [columns.join(',')];
    for (const row of rows) {
        lines.push(columns.map(c => csvCell(row[c])).join(','));
    }
    // Prefix BOM so Excel reads UTF-8 (Vietnamese feedback) correctly.
    const csv = '﻿' + lines.join('\r\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    link.href = objectUrl;
    link.download = `${cfg.label.toLowerCase()}-batch-${rows.length}-${stamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(objectUrl);
}
