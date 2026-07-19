import { useEffect, useRef, useSyncExternalStore } from 'react';
import { ExamController } from './examController';
import { printExamReport } from './examPrint';
import { featureGridHtml, scoresBreakdownHtml, escapeHtml, setRenderAccent } from '../../legacy/render';
import { useUiStore } from '../../store/ui';

// Kết quả 1 câu (port _renderQuestionResult) — audio + tải để ở JSX; transcript/
// features/scores dựng bằng renderer legacy (playback:false ở M1: nút ▶/🔊 từng từ
// cần playback.js/tts — nối ở M2; nghe cả câu dùng <audio controls> JSX bên dưới).
function questionResultInnerHtml(item: any): string {
  const r = item.result || {};
  return (
    `<div class="result-section"><h4>📝 Transcript</h4><p>${escapeHtml(r.transcript || '')}</p></div>` +
    `<div class="result-section"><h4>📈 Features</h4>${featureGridHtml(r.features || {})}</div>` +
    `<div class="result-section"><h4>📋 Điểm</h4>${scoresBreakdownHtml(r.scores, r.exam, r.phoneme, {
      pronunciationOnly: !!r.pronunciation_only,
      notice: r.notice,
      playback: false,
    })}</div>`
  );
}

export default function ExamTab() {
  const ctrlRef = useRef<ExamController>();
  if (!ctrlRef.current) ctrlRef.current = new ExamController();
  const ctrl = ctrlRef.current;
  const s = useSyncExternalStore(ctrl.subscribe, ctrl.getSnapshot);
  const accent = useUiStore((st) => st.accent);
  // Renderer legacy đọc accent qua module-level → đồng bộ trước mỗi lần render kết quả.
  setRenderAccent(accent);
  (ctrl as any).accent = accent;

  // Nạp bộ đề mẫu khi vào bước setup (thay x-init trên card setup).
  useEffect(() => {
    if (s.step === 'setup') ctrl.loadBuiltinSets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.step]);

  const typeOptions = ctrl.typeOptions();

  return (
    <div id="mode-exam">
      {/* BƯỚC 1: nạp đề */}
      {s.step === 'setup' && (
        <div className="card">
          <h2>📄 Thi cả đề — import đề</h2>
          <p className="exam-hint">
            Import tài liệu đề thi (PDF / ảnh / Word). Hệ thống tự bóc tách thành từng câu để bạn review, chỉnh
            sửa rồi làm bài tuần tự như thi thật. Điểm cuối là <strong>ước tính nội bộ</strong>, không thay thế
            kết quả thi chính thức.
          </p>
          <div className="row">
            <div className="form-group">
              <label>Kỳ thi</label>
              <select value={s.exam} onChange={(e) => ctrl.setExam(e.target.value)}>
                <option value="toeic">TOEIC</option>
                <option value="ielts">IELTS</option>
                <option value="topik">TOPIK 말하기 (tiếng Hàn)</option>
              </select>
            </div>
            {s.builtinSets.length > 0 && (
              <div className="form-group">
                <label>Bộ đề mẫu</label>
                <select value={s.builtinSetId} onChange={(e) => ctrl.setBuiltinSetId(e.target.value)}>
                  {s.builtinSets.map((set: any) => (
                    <option key={set.id} value={set.id}>
                      {set.title}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
          <label className="file-upload">
            <input
              type="file"
              accept=".pdf,.docx,image/*"
              onChange={(e) => ctrl.importFile(e.target.files?.[0])}
            />
            <div className="file-upload-text">
              <span className="icon">📤</span>
              <span>{s.importing ? 'Đang bóc tách đề…' : 'Chọn tài liệu đề (PDF / ảnh / Word)'}</span>
            </div>
          </label>
          <div style={{ marginTop: '0.75rem' }}>
            <button
              className="btn btn-secondary"
              style={{ width: 'auto', padding: '0.5rem 1rem' }}
              onClick={() => ctrl.loadBuiltin()}
              disabled={s.importing}
            >
              Dùng đề mẫu (không cần upload)
            </button>
          </div>
          {s.error && <div className="exam-error">{s.error}</div>}
        </div>
      )}

      {/* BƯỚC 2: review/sửa */}
      {s.step === 'review' && (
        <div className="card">
          <div className="result-header">
            <h2>🔎 Review đề trước khi thi</h2>
            <button
              className="btn btn-secondary"
              style={{ width: 'auto', padding: '0.5rem 1rem' }}
              onClick={() => ctrl.reset()}
            >
              ↺ Import đề khác
            </button>
          </div>
          <div className="form-group">
            <label>Tiêu đề đề thi</label>
            <input type="text" value={s.title} onChange={(e) => ctrl.setTitle(e.target.value)} />
          </div>
          {s.warnings.length > 0 && (
            <div className="exam-warnings">
              ⚠️ Bóc tách tự động — vui lòng kiểm tra:
              <ul>
                {s.warnings.map((w: string) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          {s.questions.map((q: any, i: number) => (
            <div className="exam-q" key={q.id}>
              <div className="exam-q-head">
                <span className="exam-q-seq">Câu {q.sequence}</span>
                <div className="exam-q-actions">
                  <button onClick={() => ctrl.move(i, -1)} title="Lên">
                    ▲
                  </button>
                  <button onClick={() => ctrl.move(i, 1)} title="Xuống">
                    ▼
                  </button>
                  <button onClick={() => ctrl.removeQuestion(i)} title="Xóa">
                    ✕
                  </button>
                </div>
              </div>
              <div className="form-group">
                <label>Dạng câu</label>
                <select value={q.type} onChange={(e) => ctrl.updateQuestion(i, { type: e.target.value })}>
                  {typeOptions.map((o: any) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Đề bài / câu hỏi (prompt)</label>
                <textarea rows={2} value={q.prompt} onChange={(e) => ctrl.updateQuestion(i, { prompt: e.target.value })} />
              </div>
              {ctrl.needsScript(q.type) && (
                <div className="form-group">
                  <label>Reference Script (đoạn cần đọc to)</label>
                  <textarea
                    rows={3}
                    value={q.reference_script}
                    onChange={(e) => ctrl.updateQuestion(i, { reference_script: e.target.value })}
                  />
                </div>
              )}
              {ctrl.needsProvided(q.type) && (
                <div className="form-group">
                  <label>Provided info / cue card</label>
                  <textarea
                    rows={3}
                    value={q.provided_info}
                    onChange={(e) => ctrl.updateQuestion(i, { provided_info: e.target.value })}
                  />
                </div>
              )}
              {ctrl.isPicture(q.type) && (
                <div className="form-group">
                  <label>Ảnh đề bài</label>
                  {q.image_b64 ? (
                    <img className="exam-q-img" src={ctrl.imgSrc(q)} alt="picture" />
                  ) : (
                    <p className="mode-note">Chưa có ảnh cho câu tả tranh này.</p>
                  )}
                </div>
              )}
              <div className="form-group">
                <label>Thời lượng trả lời (giây)</label>
                <input
                  type="number"
                  min={1}
                  value={q.expected_duration_sec ?? ''}
                  onChange={(e) =>
                    ctrl.updateQuestion(i, { expected_duration_sec: e.target.value === '' ? null : Number(e.target.value) })
                  }
                />
              </div>
            </div>
          ))}

          <button
            className="btn btn-secondary"
            style={{ width: 'auto', padding: '0.5rem 1rem', marginBottom: '1rem' }}
            onClick={() => ctrl.addQuestion()}
          >
            ＋ Thêm câu
          </button>

          <div className="checkbox-group" style={{ marginBottom: '1rem' }}>
            <input type="checkbox" id="exam-timed" checked={s.timed} onChange={(e) => ctrl.setTimed(e.target.checked)} />
            <label htmlFor="exam-timed">Bấm giờ như thi thật (tự ghi âm theo thời lượng từng câu)</label>
          </div>
          <button className="btn btn-primary" onClick={() => ctrl.startTest()} disabled={!s.questions.length}>
            ▶ Bắt đầu thi
          </button>
        </div>
      )}

      {/* BƯỚC 3: làm bài */}
      {s.step === 'running' && (
        <div className="card exam-running">
          {s.timed && s.phase !== 'done' && (
            <button className="exam-emergency" onClick={() => ctrl.emergencyStop()} title="Dừng khẩn cấp">
              ✕ Dừng thi
            </button>
          )}

          {/* A. Part intro */}
          {s.timed && s.phase === 'intro' && (
            <div className="exam-intro">
              <div className="exam-part-name">{s.partName}</div>
              <p className="exam-directions">{s.directionsText}</p>
              {s.countdownNum === 0 && <div className="exam-status-pill listening">🔊 Đang đọc hướng dẫn…</div>}
              {s.countdownNum > 0 && <div className="exam-countdown-num">{s.countdownNum}</div>}
            </div>
          )}

          {/* B. Câu hỏi tự động (PREP / RECORDING) */}
          {s.timed && (s.phase === 'prep' || s.phase === 'recording') && ctrl.current && (
            <div>
              <div className="exam-runner-meta">
                <h2>
                  Câu {s.idx + 1} / {s.order.length}
                </h2>
                <span className={'exam-status-pill ' + s.statusKey}>{ctrl.statusText()}</span>
              </div>
              <div className="exam-prompt-box">
                <div className="label">{ctrl.typeLabel(ctrl.current.type)}</div>
                {ctrl.current.prompt && <div style={{ marginBottom: '0.5rem' }}>{ctrl.current.prompt}</div>}
                {ctrl.current.reference_script && <div className="script">{ctrl.current.reference_script}</div>}
                {ctrl.current.provided_info && (
                  <div className="script" style={{ color: '#555' }}>
                    {ctrl.current.provided_info}
                  </div>
                )}
                {ctrl.current.image_b64 && <img className="exam-runner-img" src={ctrl.imgSrc(ctrl.current)} alt="picture" />}
              </div>
              <div className="exam-timer-wrap">
                <div className={'exam-timer ' + s.statusKey}>{s.secondsLeft}s</div>
                {s.phase === 'recording' && (
                  <div className="exam-wave-wrap">
                    <span className="exam-mic">🎙</span>
                    <canvas id="exam-wave" width={600} height={80} className="exam-wave"></canvas>
                  </div>
                )}
              </div>
              <div className="exam-nav">
                {s.phase === 'prep' && (
                  <button className="btn btn-secondary" onClick={() => ctrl.skip()}>
                    Bỏ qua chuẩn bị →
                  </button>
                )}
                {s.phase === 'recording' && (
                  <button className="btn btn-secondary" onClick={() => ctrl.skip()}>
                    Kết thúc trả lời →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* C. Chế độ THỦ CÔNG (không bấm giờ) */}
          {!s.timed && ctrl.current && (
            <div>
              <div className="exam-runner-meta">
                <h2>
                  Câu {s.idx + 1} / {s.order.length}
                </h2>
              </div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  flexWrap: 'wrap',
                  marginBottom: '1rem',
                  padding: '0.75rem',
                  background: '#f8fafc',
                  borderRadius: 8,
                }}
              >
                <label
                  className="btn btn-secondary"
                  style={{ width: 'auto', padding: '0.5rem 1rem', margin: 0, cursor: 'pointer' }}
                >
                  📁📁 Import nhiều audio cùng lúc
                  <input
                    type="file"
                    multiple
                    accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm,.weba,.mp4"
                    onChange={(e) => {
                      ctrl.uploadBatch(e.target.files);
                      e.target.value = '';
                    }}
                    style={{ display: 'none' }}
                  />
                </label>
                <span className="exam-hint" style={{ margin: 0 }}>
                  Gán lần lượt theo tên file, bắt đầu từ câu hiện tại. Đã có audio:{' '}
                  <strong>
                    {ctrl.recordedCount()}/{s.order.length}
                  </strong>
                </span>
              </div>
              <div className="exam-prompt-box">
                <div className="label">{ctrl.typeLabel(ctrl.current.type)}</div>
                {ctrl.current.prompt && <div style={{ marginBottom: '0.5rem' }}>{ctrl.current.prompt}</div>}
                {ctrl.current.reference_script && <div className="script">{ctrl.current.reference_script}</div>}
                {ctrl.current.provided_info && (
                  <div className="script" style={{ color: '#555' }}>
                    {ctrl.current.provided_info}
                  </div>
                )}
                {ctrl.current.image_b64 && <img className="exam-runner-img" src={ctrl.imgSrc(ctrl.current)} alt="picture" />}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
                <button
                  className="btn btn-primary"
                  style={{ width: 'auto', padding: '0.6rem 1.2rem' }}
                  onClick={() => ctrl.toggleRec()}
                >
                  {s.recording ? '⏹ Dừng ghi' : '🎙 Ghi âm câu này'}
                </button>
                <label
                  className={'btn btn-secondary' + (s.recording ? ' disabled' : '')}
                  style={{ width: 'auto', padding: '0.6rem 1.2rem', margin: 0, cursor: 'pointer' }}
                >
                  📁 Import audio
                  <input
                    type="file"
                    accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm,.weba,.mp4"
                    disabled={s.recording}
                    onChange={(e) => {
                      ctrl.uploadRec(e.target.files?.[0]);
                      e.target.value = '';
                    }}
                    style={{ display: 'none' }}
                  />
                </label>
                {s.recording && <span className="exam-rec-dot">● đang ghi…</span>}
                {ctrl.current._recBlob && !s.recording && <span style={{ color: '#16a34a' }}>✓ đã có audio</span>}
              </div>
              {ctrl.current._recUrl && (
                <audio src={ctrl.current._recUrl} controls style={{ marginTop: '0.75rem', width: '100%' }} />
              )}
              <div className="exam-nav">
                <button className="btn btn-secondary" onClick={() => ctrl.prevManual()} disabled={s.idx === 0}>
                  ← Câu trước
                </button>
                {s.idx < s.order.length - 1 && (
                  <button className="btn btn-secondary" onClick={() => ctrl.nextManual()}>
                    Câu tiếp →
                  </button>
                )}
                <button className="btn btn-primary" onClick={() => ctrl.submitExam()} disabled={s.grading}>
                  {s.grading ? 'Đang chấm…' : `✔ Nộp & chấm (${ctrl.recordedCount()}/${s.order.length})`}
                </button>
              </div>
            </div>
          )}

          {/* D. Hoàn thành (tự động) */}
          {s.timed && s.phase === 'done' && (
            <div>
              <h2>✅ Đã hoàn thành tất cả câu</h2>
              <p className="exam-hint">Nghe lại nếu cần rồi nộp để chấm gộp.</p>
              {s.order
                .filter((q: any) => q._recBlob)
                .map((q: any) => (
                  <div className="exam-q" key={q.id}>
                    <div className="exam-q-head">
                      <span className="exam-q-seq">Câu {q.sequence}</span>
                      <span style={{ color: '#888', fontSize: '0.85rem' }}>{ctrl.typeLabel(q.type)}</span>
                    </div>
                    <audio src={q._recUrl} controls style={{ width: '100%' }} />
                  </div>
                ))}
              <div className="exam-nav">
                <button className="btn btn-secondary" onClick={() => ctrl.reset()}>
                  ↺ Làm lại đề
                </button>
                <button className="btn btn-primary" onClick={() => ctrl.submitExam()} disabled={s.grading}>
                  {s.grading ? 'Đang chấm…' : `✔ Nộp & chấm (${ctrl.recordedCount()}/${s.order.length})`}
                </button>
              </div>
            </div>
          )}

          {s.error && <div className="exam-error">{s.error}</div>}
        </div>
      )}

      {/* BƯỚC 4: kết quả gộp */}
      {s.step === 'result' && (
        <div className="card">
          <div className="result-header">
            <h2>📊 Kết quả cả đề</h2>
            <div className="actions">
              <button
                className="btn btn-secondary"
                style={{ width: 'auto', padding: '0.5rem 1rem' }}
                onClick={() => printExamReport(s.result, (t: string) => ctrl.typeLabel(t))}
              >
                🖨 Print / PDF
              </button>
              <button
                className="btn btn-secondary"
                style={{ width: 'auto', padding: '0.5rem 1rem' }}
                onClick={() => ctrl.downloadAllRecordings()}
              >
                ⬇ Tải tất cả audio
              </button>
              <button
                className="btn btn-secondary"
                style={{ width: 'auto', padding: '0.5rem 1rem' }}
                onClick={() => ctrl.reset()}
              >
                Thi đề khác
              </button>
            </div>
          </div>
          <div className="exam-overall">
            <div className="label">{ctrl.overallLabelText()}</div>
            <div className="score">{ctrl.overallText()}</div>
            <div className="note">Ước tính nội bộ (trung bình các câu) — không thay thế kết quả thi chính thức.</div>
          </div>
          {(s.result ? s.result.questions : []).map((item: any) => {
            const q = (s.order || []).find((o: any) => o.id === item.question_id);
            return (
              <details className="exam-result-q" key={item.question_id}>
                <summary>
                  <span>
                    Câu {item.sequence || '?'} · {ctrl.typeLabel(item.type)}
                  </span>
                  <span className="exam-q-summary-right">
                    <span className="exam-q-score">{ctrl.questionScore(item)}</span>
                    {ctrl.questionDone(item) && (
                      <span className="exam-q-caret" aria-hidden="true">
                        ▸
                      </span>
                    )}
                  </span>
                </summary>
                <div className="exam-q-body">
                  {q && q._recUrl && (
                    <a
                      className="btn btn-secondary"
                      href={q._recUrl}
                      download={ctrl._recordingFilename(q)}
                      style={{
                        width: 'auto',
                        display: 'inline-block',
                        textDecoration: 'none',
                        padding: '0.35rem 0.9rem',
                        marginBottom: '0.6rem',
                      }}
                    >
                      ⬇ Tải audio câu này
                    </a>
                  )}
                  {q && q._recUrl && (
                    <audio src={q._recUrl} controls style={{ width: '100%', marginBottom: '0.6rem' }} />
                  )}
                  {item.error ? (
                    <p className="exam-error">{item.error}</p>
                  ) : (
                    <div dangerouslySetInnerHTML={{ __html: questionResultInnerHtml(item) }} />
                  )}
                </div>
              </details>
            );
          })}
        </div>
      )}
    </div>
  );
}
