'use strict';

// CSV export (single) + báo cáo in (single & batch → Print / Save as PDF).

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
  .accent-row { display: none; }
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
