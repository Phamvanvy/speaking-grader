'use strict';

// ── Tab "Lịch sử" — bài đã chấm, lưu server-side theo user_id ẩn danh ──────
// List/chi tiết/xoá qua GET /history/list, GET|DELETE /history/{id},
// GET /history/{id}/audio. Chi tiết tái dùng builder cấp thấp của render.js
// (featureGridHtml, scoresBreakdownHtml, telemetryHtml) giống exam.js
// _renderQuestionResult — KHÔNG dùng showSingleResult (nó ghi vào #result của
// mode-classic và mutate lastSingleData). Nút ▶ nghe lại từng từ hoạt động vì
// playback.js nhận data-src URL (ở đây là URL audio server-side).

const HISTORY_PAGE_SIZE = 20;
let historyOffset = 0;

const HISTORY_KIND_LABEL = {
    single: { label: 'Chấm lẻ', cls: 'single' },
    batch: { label: 'Cả lớp', cls: 'batch' },
    exam: { label: 'Thi cả đề', cls: 'exam' },
};

function historyScoreText(rec) {
    if (rec.overall_score != null) return `${rec.overall_score}/${rec.overall_max}`;
    if (rec.pronunciation_only) return '🔊 pron.';
    return '--';
}

function historyDateText(iso) {
    const d = new Date(iso);
    return isNaN(d) ? (iso || '') : d.toLocaleString();
}

async function loadHistoryList(offset = 0) {
    historyOffset = offset;
    const listEl = document.getElementById('history-list');
    const pagerEl = document.getElementById('history-pager');
    const toggle = document.getElementById('history-save-toggle');
    if (toggle) toggle.checked = historySaveEnabled();
    listEl.innerHTML = '<p class="history-empty">⏳ Đang tải…</p>';
    pagerEl.innerHTML = '';
    try {
        const url = `${apiBase()}/history/list?user_id=${encodeURIComponent(getUserId())}`
            + `&limit=${HISTORY_PAGE_SIZE}&offset=${offset}`;
        const res = await fetch(url);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        // Trang hiện tại rỗng nhưng vẫn còn bản ghi (vd vừa xoá bản ghi cuối của
        // trang cuối) → lùi một trang thay vì hiện "chưa có bài chấm nào".
        if (!data.records.length && offset > 0) {
            return loadHistoryList(Math.max(0, offset - HISTORY_PAGE_SIZE));
        }
        renderHistoryList(data);
    } catch (e) {
        listEl.innerHTML = `<p class="history-empty">⚠️ Không tải được lịch sử: ${escapeHtml(e.message)}</p>`;
    }
}

function renderHistoryList(data) {
    const listEl = document.getElementById('history-list');
    const pagerEl = document.getElementById('history-pager');
    if (!data.records.length) {
        // Chỉ tới đây khi total = 0 (trang rỗng giữa chừng đã được lùi ở loadHistoryList).
        listEl.innerHTML = '<p class="history-empty">Chưa có bài chấm nào được lưu.'
            + ' Chấm một bài ở tab "Chấm bài lẻ" hoặc "Thi cả đề" rồi quay lại đây.</p>';
        return;
    }
    listEl.innerHTML = data.records.map(rec => {
        const kind = HISTORY_KIND_LABEL[rec.kind] || HISTORY_KIND_LABEL.single;
        const examLabel = rec.exam ? examConfig(rec.exam).label : '';
        const sub = [historyDateText(rec.created_at), examLabel,
            rec.item_count > 1 ? `${rec.item_count} bài` : '']
            .filter(Boolean).join(' · ');
        return `<div class="history-row" data-id="${escapeHtml(rec.id)}">
            <span class="history-badge ${kind.cls}">${kind.label}</span>
            <div class="history-info">
                <div class="history-title">${escapeHtml(rec.title || '(không tên)')}</div>
                <div class="history-sub">${escapeHtml(sub)}</div>
            </div>
            <div class="history-score">${escapeHtml(historyScoreText(rec))}</div>
            <div class="history-actions">
                <button class="btn btn-secondary btn-inline" onclick="openHistoryDetail('${escapeHtml(rec.id)}')">Xem</button>
                <button class="btn btn-secondary btn-inline history-del" title="Xoá bản ghi này (kèm audio)"
                    onclick="deleteHistoryRecord('${escapeHtml(rec.id)}')">🗑</button>
            </div>
        </div>`;
    }).join('');

    // Pager: ‹ trước / trang X/Y / sau ›
    const pages = Math.max(1, Math.ceil(data.total / data.limit));
    const page = Math.floor(data.offset / data.limit) + 1;
    if (pages > 1) {
        const prev = data.offset - data.limit;
        const next = data.offset + data.limit;
        pagerEl.innerHTML =
            `<button class="btn btn-secondary btn-inline" ${page <= 1 ? 'disabled' : ''}
                onclick="loadHistoryList(${Math.max(0, prev)})">‹ Trước</button>
            <span class="history-page">Trang ${page}/${pages} · ${data.total} bản ghi</span>
            <button class="btn btn-secondary btn-inline" ${page >= pages ? 'disabled' : ''}
                onclick="loadHistoryList(${next})">Sau ›</button>`;
    } else {
        pagerEl.innerHTML = `<span class="history-page">${data.total} bản ghi</span>`;
    }
}

async function deleteHistoryRecord(id) {
    if (!confirm('Xoá bản ghi này khỏi lịch sử (kèm audio đã lưu)?')) return;
    try {
        const url = `${apiBase()}/history/${encodeURIComponent(id)}?user_id=${encodeURIComponent(getUserId())}`;
        const res = await fetch(url, { method: 'DELETE' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    } catch (e) {
        alert(`Không xoá được: ${e.message}`);
        return;
    }
    if (document.getElementById('history-detail-wrap').dataset.recordId === id) closeHistoryDetail();
    loadHistoryList(historyOffset);
}

function closeHistoryDetail() {
    const wrap = document.getElementById('history-detail-wrap');
    wrap.classList.remove('visible');
    delete wrap.dataset.recordId;
    document.getElementById('history-detail').innerHTML = '';
}

async function openHistoryDetail(id) {
    const wrap = document.getElementById('history-detail-wrap');
    const el = document.getElementById('history-detail');
    wrap.dataset.recordId = id;
    wrap.classList.add('visible');
    el.innerHTML = '<p class="history-empty">⏳ Đang tải…</p>';
    try {
        const url = `${apiBase()}/history/${encodeURIComponent(id)}?user_id=${encodeURIComponent(getUserId())}`;
        const res = await fetch(url);
        const rec = await res.json();
        if (!res.ok) throw new Error(rec.detail || `HTTP ${res.status}`);
        renderHistoryDetail(rec);
        wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
        el.innerHTML = `<p class="history-empty">⚠️ ${escapeHtml(e.message)}</p>`;
    }
}

function historyAudioUrl(recordId, itemId) {
    let url = `${apiBase()}/history/${encodeURIComponent(recordId)}/audio`
        + `?user_id=${encodeURIComponent(getUserId())}`;
    if (itemId) url += `&item_id=${encodeURIComponent(itemId)}`;
    // <audio src> không qua fetch → không có header Authorization; đính token qua
    // query để audio của tài khoản (user_id "khoá") vẫn được cấp phép.
    const token = (typeof authToken === 'function') ? authToken() : null;
    if (token) url += `&token=${encodeURIComponent(token)}`;
    return url;
}

// Khối kết quả 1 bài (dùng chung cho single + từng item exam/batch). `src` =
// URL audio server-side → <audio controls> + nút ▶ nghe lại từng từ (playbackSrc).
function historyResultHtml(result, src) {
    const r = result || {};
    const audio = src
        ? `<audio controls preload="none" src="${escapeHtml(src)}" style="width:100%;margin-bottom:0.6rem;"></audio>`
        : '';
    return audio
        + `<div class="result-section"><h4>📝 Transcript</h4><p>${escapeHtml(r.transcript || '')}</p></div>`
        + `<div class="result-section"><h4>📈 Features</h4>${featureGridHtml(r.features || {})}</div>`
        + `<div class="result-section"><h4>📋 Điểm</h4>${scoresBreakdownHtml(r.scores, r.exam, r.phoneme, {
            pronunciationOnly: !!r.pronunciation_only, notice: r.notice,
            playback: !!src, playbackSrc: src,
        })}</div>`
        + (r.telemetry ? `<div class="result-section"><h4>⚙️ Telemetry</h4>${telemetryHtml(r.telemetry)}</div>` : '');
}

function renderHistoryDetail(rec) {
    const el = document.getElementById('history-detail');
    const kind = HISTORY_KIND_LABEL[rec.kind] || HISTORY_KIND_LABEL.single;
    const title = document.getElementById('history-detail-title');
    if (title) title.textContent = `📊 ${kind.label} — ${rec.title || ''}`;

    const header = `<div class="history-detail-meta">
        ${escapeHtml(historyDateText(rec.created_at))}
        ${rec.exam ? ' · ' + escapeHtml(examConfig(rec.exam).label) : ''}
        ${rec.mode ? ' · ' + escapeHtml(rec.mode) : ''}
        · Điểm: <strong>${escapeHtml(historyScoreText(rec))}</strong>
    </div>`;

    if (rec.kind === 'single') {
        const src = rec.has_audio ? historyAudioUrl(rec.id) : null;
        el.innerHTML = header + historyResultHtml(rec.result, src);
        return;
    }

    // exam & batch: mỗi item một <details> (giống màn kết quả thi cả đề).
    const items = (rec.items || []).map(it => {
        const src = it.has_audio ? historyAudioUrl(rec.id, it.id) : null;
        const score = it.error ? '⚠️' : (it.score != null ? it.score : '--');
        const body = it.error
            ? (src ? `<audio controls preload="none" src="${escapeHtml(src)}" style="width:100%;margin-bottom:0.6rem;"></audio>` : '')
                + `<p class="exam-error">${escapeHtml(it.error)}</p>`
            : historyResultHtml(it.result, src);
        return `<details class="exam-result-q">
            <summary>
                <span>${escapeHtml(it.label || '')}</span>
                <span class="exam-q-summary-right"><span class="exam-q-score">${escapeHtml(String(score))}</span></span>
            </summary>
            <div class="exam-q-body">${body}</div>
        </details>`;
    }).join('');
    el.innerHTML = header + (items || '<p class="history-empty">Không có bài nào trong bản ghi này.</p>');
}

// Đồng bộ checkbox opt-out với localStorage ngay khi trang tải.
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('history-save-toggle');
    if (toggle) toggle.checked = historySaveEnabled();
});
