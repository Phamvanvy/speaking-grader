import { useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiGet, apiFetch, authedResourceUrl } from '../../lib/api';
import { getUserId, historySaveEnabled, setHistorySaveEnabled } from '../../lib/identity';
import { examConfig } from '../../lib/config';
import { escapeHtml } from '../../lib/format';
import { featureGridHtml, scoresBreakdownHtml, telemetryHtml, setRenderAccent } from '../../legacy/render';
import { useUiStore } from '../../store/ui';

const PAGE_SIZE = 20;
const KIND_LABEL: Record<string, { label: string; cls: string }> = {
  single: { label: 'Chấm lẻ', cls: 'single' },
  batch: { label: 'Cả lớp', cls: 'batch' },
  exam: { label: 'Thi cả đề', cls: 'exam' },
};

function scoreText(rec: any): string {
  if (rec.overall_score != null) return `${rec.overall_score}/${rec.overall_max}`;
  if (rec.pronunciation_only) return '🔊 pron.';
  return '--';
}
function dateText(iso: string): string {
  const d = new Date(iso);
  return isNaN(+d) ? iso || '' : d.toLocaleString();
}

// Khối kết quả 1 bài (port historyResultHtml) — src = URL audio server-side → <audio>
// + nút ▶ (playbackSrc). Dùng renderer interop.
function resultHtml(result: any, src: string | null): string {
  const r = result || {};
  const audio = src
    ? `<audio controls preload="none" src="${escapeHtml(src)}" style="width:100%;margin-bottom:0.6rem;"></audio>`
    : '';
  return (
    audio +
    `<div class="result-section"><h4>📝 Transcript</h4><p>${escapeHtml(r.transcript || '')}</p></div>` +
    `<div class="result-section"><h4>📈 Features</h4>${featureGridHtml(r.features || {})}</div>` +
    `<div class="result-section"><h4>📋 Điểm</h4>${scoresBreakdownHtml(r.scores, r.exam, r.phoneme, {
      pronunciationOnly: !!r.pronunciation_only,
      notice: r.notice,
      playback: !!src,
      playbackSrc: src,
    })}</div>` +
    (r.telemetry ? `<div class="result-section"><h4>⚙️ Telemetry</h4>${telemetryHtml(r.telemetry)}</div>` : '')
  );
}

export default function HistoryTab() {
  const accent = useUiStore((s) => s.accent);
  setRenderAccent(accent);
  const userId = getUserId();
  const qc = useQueryClient();
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [saveEnabled, setSaveEnabledState] = useState(historySaveEnabled());

  const listQuery = useQuery({
    queryKey: ['history', 'list', userId, offset],
    queryFn: () => apiGet<any>(`/history/list?user_id=${encodeURIComponent(userId)}&limit=${PAGE_SIZE}&offset=${offset}`),
  });

  const detailQuery = useQuery({
    queryKey: ['history', 'detail', userId, selectedId],
    queryFn: () => apiGet<any>(`/history/${encodeURIComponent(selectedId!)}?user_id=${encodeURIComponent(userId)}`),
    enabled: !!selectedId,
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) =>
      apiFetch(`/history/${encodeURIComponent(id)}?user_id=${encodeURIComponent(userId)}`, { method: 'DELETE', noRetry: true }),
    onSuccess: (_res, id) => {
      if (selectedId === id) setSelectedId(null);
      qc.invalidateQueries({ queryKey: ['history', 'list'] });
    },
  });

  function audioUrl(recordId: string, itemId?: string): string {
    let path = `/history/${encodeURIComponent(recordId)}/audio?user_id=${encodeURIComponent(userId)}`;
    if (itemId) path += `&item_id=${encodeURIComponent(itemId)}`;
    return authedResourceUrl(path);
  }
  function downloadZip(recordId: string) {
    const url = authedResourceUrl(`/history/${encodeURIComponent(recordId)}/audio.zip?user_id=${encodeURIComponent(userId)}`);
    const a = document.createElement('a');
    a.href = url;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  const data = listQuery.data;
  const records: any[] = data?.records || [];
  const pages = data ? Math.max(1, Math.ceil(data.total / data.limit)) : 1;
  const page = data ? Math.floor(data.offset / data.limit) + 1 : 1;

  // Detail HTML — re-render khi đổi accent (đọc từ detailQuery.data).
  const detail = detailQuery.data;
  const detailBody = useMemo(() => {
    if (!detail) return null;
    if (detail.kind === 'single') {
      const src = detail.has_audio ? audioUrl(detail.id) : null;
      return resultHtml(detail.result, src);
    }
    // exam & batch: mỗi item một <details>
    return (detail.items || [])
      .map((it: any) => {
        const src = it.has_audio ? audioUrl(detail.id, it.id) : null;
        const score = it.error ? '⚠️' : it.score != null ? it.score : '--';
        const body = it.error
          ? (src ? `<audio controls preload="none" src="${escapeHtml(src)}" style="width:100%;margin-bottom:0.6rem;"></audio>` : '') +
            `<p class="exam-error">${escapeHtml(it.error)}</p>`
          : resultHtml(it.result, src);
        return `<details class="exam-result-q">
            <summary><span>${escapeHtml(it.label || '')}</span>
            <span class="exam-q-summary-right"><span class="exam-q-score">${escapeHtml(String(score))}</span></span></summary>
            <div class="exam-q-body">${body}</div></details>`;
      })
      .join('') || '<p class="history-empty">Không có bài nào trong bản ghi này.</p>';
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detail, accent]);

  const detailKind = detail ? KIND_LABEL[detail.kind] || KIND_LABEL.single : null;

  return (
    <div id="mode-history">
      <div className="card">
        <div className="result-header">
          <h2>🕘 Lịch sử chấm bài</h2>
          <button className="btn btn-secondary btn-inline" onClick={() => listQuery.refetch()}>
            ↻ Tải lại
          </button>
        </div>
        <label className="history-opt">
          <input
            type="checkbox"
            checked={saveEnabled}
            onChange={(e) => {
              setHistorySaveEnabled(e.target.checked);
              setSaveEnabledState(e.target.checked);
            }}
          />{' '}
          💾 Lưu bài chấm vào lịch sử (kèm ghi âm — lưu trên máy chủ, tách theo trình duyệt này)
        </label>

        <div className="history-list" id="history-list">
          {listQuery.isLoading && <p className="history-empty">⏳ Đang tải…</p>}
          {listQuery.isError && <p className="history-empty">⚠️ Không tải được lịch sử.</p>}
          {listQuery.isSuccess && records.length === 0 && (
            <p className="history-empty">
              Chưa có bài chấm nào được lưu. Chấm một bài ở tab "Chấm bài lẻ" hoặc "Thi cả đề" rồi quay lại đây.
            </p>
          )}
          {records.map((rec) => {
            const kind = KIND_LABEL[rec.kind] || KIND_LABEL.single;
            const examLabel = rec.exam ? examConfig(rec.exam).label : '';
            const sub = [dateText(rec.created_at), examLabel, rec.item_count > 1 ? `${rec.item_count} bài` : '']
              .filter(Boolean)
              .join(' · ');
            return (
              <div className="history-row" key={rec.id} data-id={rec.id}>
                <span className={'history-badge ' + kind.cls}>{kind.label}</span>
                <div className="history-info">
                  <div className="history-title">{rec.title || '(không tên)'}</div>
                  <div className="history-sub">{sub}</div>
                </div>
                <div className="history-score">{scoreText(rec)}</div>
                <div className="history-actions">
                  <button className="btn btn-secondary btn-inline" onClick={() => setSelectedId(rec.id)}>
                    Xem
                  </button>
                  {rec.has_audio && (
                    <button className="btn btn-secondary btn-inline" title="Tải tất cả audio (zip)" onClick={() => downloadZip(rec.id)}>
                      ⬇
                    </button>
                  )}
                  <button
                    className="btn btn-secondary btn-inline history-del"
                    title="Xoá bản ghi này (kèm audio)"
                    onClick={() => {
                      if (confirm('Xoá bản ghi này khỏi lịch sử (kèm audio đã lưu)?')) deleteMut.mutate(rec.id);
                    }}
                  >
                    🗑
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        <div className="history-pager" id="history-pager">
          {data && pages > 1 ? (
            <>
              <button className="btn btn-secondary btn-inline" disabled={page <= 1} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
                ‹ Trước
              </button>
              <span className="history-page">
                Trang {page}/{pages} · {data.total} bản ghi
              </span>
              <button className="btn btn-secondary btn-inline" disabled={page >= pages} onClick={() => setOffset(offset + PAGE_SIZE)}>
                Sau ›
              </button>
            </>
          ) : data ? (
            <span className="history-page">{data.total} bản ghi</span>
          ) : null}
        </div>
      </div>

      {selectedId && (
        <div className="result visible" id="history-detail-wrap">
          <div className="card">
            <div className="result-header">
              <h2 id="history-detail-title">
                📊 {detailKind ? detailKind.label : 'Chi tiết'}
                {detail?.title ? ' — ' + detail.title : ''}
              </h2>
              <button className="btn btn-secondary btn-inline" onClick={() => setSelectedId(null)}>
                Đóng
              </button>
            </div>
            <div id="history-detail">
              {detailQuery.isLoading && <p className="history-empty">⏳ Đang tải…</p>}
              {detailQuery.isError && <p className="history-empty">⚠️ Không tải được chi tiết.</p>}
              {detail && (
                <>
                  <div
                    className="history-detail-meta"
                    dangerouslySetInnerHTML={{
                      __html: `${escapeHtml(dateText(detail.created_at))}${detail.exam ? ' · ' + escapeHtml(examConfig(detail.exam).label) : ''}${
                        detail.mode ? ' · ' + escapeHtml(detail.mode) : ''
                      } · Điểm: <strong>${escapeHtml(scoreText(detail))}</strong>`,
                    }}
                  />
                  {detailBody && <div dangerouslySetInnerHTML={{ __html: detailBody }} />}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
