// @ts-nocheck
// Print/PDF cho kết quả cả đề — port printExamReport() (exam.js). Dùng chung
// reportStyles/reportCriteriaHtml + renderer phoneme của legacy.

import { escapeHtml } from '../../lib/format';
import { examConfig } from '../../lib/config';
import { featureTiles, phonemeErrorsHtml } from '../../legacy/render';
import { reportStyles, reportCriteriaHtml, openPrintWindow } from '../../legacy/report';

export function printExamReport(result, typeLabel) {
  if (!result || !result.questions.length) {
    alert('Chưa có kết quả để export.');
    return;
  }
  const cfg = examConfig(result.exam);
  const data = result;

  const overviewRows = data.questions
    .map((item) => {
      const label = `Câu ${item.sequence} · ${typeLabel(item.type)}`;
      if (item.error) {
        return `<tr><td class="col-idx">${item.sequence}</td>
                <td>${escapeHtml(label)}</td>
                <td class="col-score err">error</td>
                <td class="col-fb err">${escapeHtml(item.error)}</td></tr>`;
      }
      const r = item.result || {};
      const pronOnly = !!r.pronunciation_only;
      const score = pronOnly ? '🔊 pron.' : escapeHtml(r.scores?.[cfg.scoreField] ?? '--');
      const fb = r.scores?.summary_feedback || (pronOnly ? r.notice : '') || '';
      return `<tr><td class="col-idx">${item.sequence}</td>
            <td>${escapeHtml(label)}</td>
            <td class="col-score">${score}</td>
            <td class="col-fb">${escapeHtml(fb)}</td></tr>`;
    })
    .join('');

  const detailSections = data.questions
    .map((item) => {
      const head = `<div class="file-head">Câu ${item.sequence} · ${escapeHtml(typeLabel(item.type))}</div>`;
      if (item.error) {
        return `<section class="file">${head}<p class="body err">❌ ${escapeHtml(item.error)}</p></section>`;
      }
      const r = item.result || {};
      const s = r.scores || {};
      const f = r.features || {};
      const pronOnly = !!r.pronunciation_only;
      const overall = s[cfg.scoreField];
      const featuresHtml = featureTiles(f)
        .map((t) => `<div class="tile"><div class="tval">${escapeHtml(t.value)}</div><div class="tname">${escapeHtml(t.name)}</div></div>`)
        .join('');
      const summaryRows = [
        ['Task Completion', s.task_completion],
        ['Content Relevance', s.content_relevance],
      ]
        .filter(([, v]) => v != null && v !== '')
        .map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(v)}</td></tr>`)
        .join('');
      return `<section class="file">
                ${head}
                ${
                  pronOnly
                    ? `<div class="overall"><span class="lbl">⚠️ ${escapeHtml(r.notice || 'Chỉ chấm phát âm (chưa có đề bài).')}</span></div>`
                    : `<div class="overall"><span class="big">${escapeHtml(overall ?? '--')}</span><span class="lbl">${escapeHtml(cfg.overallLabel)} (max ${cfg.overallMax})</span></div>`
                }
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
    })
    .join('');

  const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>${escapeHtml(cfg.label)} Exam Report${data.title ? ' — ' + escapeHtml(data.title) : ''}</title>
<style>${reportStyles()}</style></head>
<body>
  <h1>${escapeHtml(cfg.label)} Exam Report</h1>
  <div class="meta">${data.title ? escapeHtml(data.title) + ' · ' : ''}${data.graded || data.questions.length}/${data.count} câu đã chấm · Generated ${escapeHtml(new Date().toLocaleString())}</div>

  <div class="overall">
    <span class="big">${escapeHtml(data.overall ?? '--')}</span>
    <span class="lbl">${escapeHtml(cfg.overallLabel)} (max ${data.overall_max})</span>
  </div>

  <h2>Overview</h2>
  <table class="overview">
    <thead><tr><th class="col-idx">#</th><th>Câu</th><th class="col-score">${escapeHtml(cfg.overallLabel)}</th><th class="col-fb">Feedback</th></tr></thead>
    <tbody>${overviewRows}</tbody>
  </table>

  ${detailSections}

  <script>window.onload = function () { window.print(); };<\/script>
</body></html>`;

  openPrintWindow(html);
}
