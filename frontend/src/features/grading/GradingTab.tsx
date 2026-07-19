import { useEffect, useMemo, useRef, useState } from 'react';
import { examConfig } from '../../lib/config';
import { apiFetch } from '../../lib/api';
import { getUserId, historySaveEnabled } from '../../lib/identity';
import { useUiStore } from '../../store/ui';
import {
  featureGridHtml,
  scoresBreakdownHtml,
  telemetryHtml,
  setRenderAccent,
  setPlaybackUrlFn,
} from '../../legacy/render';
import { setPlaybackUrl, setPlaybackAccent } from './playback';
import { printSingleReport, printBatchReport, downloadBlob, downloadBlobsSequentially } from '../../legacy/report';
import {
  saveRecording,
  listRecordings,
  getRecording,
  deleteRecordingDb,
  clearRecordingsDb,
  recordingExtension,
  formatBytes,
} from './recordingsDb';
import { SuggestPanel } from './SuggestPanel';

const MODE_NOTES: Record<string, string> = {
  practice:
    'Fast first pass (faster-whisper). Auto-upgrades to the Mock Test pipeline (better ASR + phoneme analysis) when confidence/coverage is low.',
  mock_test: 'Most accurate: best ASR (WhisperX) + phoneme analysis ON. Use this as the reference score.',
};

export default function GradingTab() {
  const accent = useUiStore((st) => st.accent);
  const setAccent = useUiStore((st) => st.setAccent);

  // ── form state ──
  const [exam, setExam] = useState('toeic');
  const [questionType, setQuestionType] = useState('');
  const [referenceText, setReferenceText] = useState('');
  const [promptText, setPromptText] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState('practice');
  const [expectedDuration, setExpectedDuration] = useState('');
  const [feedbackLang, setFeedbackLang] = useState('');
  const [noAi, setNoAi] = useState(false);
  const [grading, setGrading] = useState(false);

  // ── results ──
  const [singleData, setSingleData] = useState<any>(null);
  const [singleFilename, setSingleFilename] = useState('');
  const [batchData, setBatchData] = useState<any>(null);
  const batchFilesRef = useRef<File[]>([]);
  const singleUrlRef = useRef<string | null>(null);

  // ── saved recordings + recorder ──
  const [savedRecs, setSavedRecs] = useState<any[]>([]);
  const [recording, setRecording] = useState(false);
  const [recSeconds, setRecSeconds] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recTimerRef = useRef<any>(null);
  const startingRef = useRef(false);

  const cfg = examConfig(exam);
  const questionTypes = cfg.questionTypes;
  const qt = useMemo(() => questionTypes.find((q) => q.value === questionType), [questionTypes, questionType]);
  const uses = (qt && qt.uses) || ['reference', 'image', 'prompt'];
  const showReference = uses.includes('reference');
  const showImage = uses.includes('image');
  const showPrompt = uses.includes('prompt');
  const showAccent = cfg.lang !== 'ko';

  // Đổi exam → reset questionType về option đầu của exam mới (tránh giữ giá trị exam cũ).
  useEffect(() => {
    setQuestionType(questionTypes[0]?.value ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exam]);

  // Nạp bản ghi đã lưu (IndexedDB) khi mở tab.
  useEffect(() => {
    refreshSaved();
  }, []);

  // Renderer + playback đọc accent qua module-level → đồng bộ trước mỗi render kết quả.
  setRenderAccent(accent);
  setPlaybackAccent(accent);

  async function refreshSaved() {
    try {
      setSavedRecs(await listRecordings());
    } catch {
      setSavedRecs([]);
    }
  }

  // ── file helpers ──
  function addFiles(newFiles: File[]) {
    setFiles((prev) => [...prev, ...newFiles]);
  }
  function clearFiles() {
    setFiles([]);
  }

  // ── recorder ──
  function startRecTimer() {
    if (recTimerRef.current) clearInterval(recTimerRef.current);
    setRecSeconds(0);
    recTimerRef.current = setInterval(() => setRecSeconds((s) => s + 1), 1000);
  }
  function stopRecTimer() {
    if (recTimerRef.current) {
      clearInterval(recTimerRef.current);
      recTimerRef.current = null;
    }
  }
  async function toggleRecording() {
    if (recorderRef.current && recorderRef.current.state === 'recording') {
      recorderRef.current.stop();
      return;
    }
    if (startingRef.current) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Trình duyệt không hỗ trợ ghi âm (getUserMedia). Hãy dùng Chrome/Edge/Firefox bản mới và truy cập qua HTTPS hoặc localhost.');
      return;
    }
    let stream: MediaStream;
    startingRef.current = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err: any) {
      alert(`Không truy cập được micro: ${err.message}`);
      startingRef.current = false;
      return;
    }
    startingRef.current = false;
    chunksRef.current = [];
    const rec = new MediaRecorder(stream);
    recorderRef.current = rec;
    rec.addEventListener('dataavailable', (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    });
    rec.addEventListener('stop', async () => {
      stream.getTracks().forEach((t) => t.stop());
      stopRecTimer();
      setRecording(false);
      const type = rec.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type });
      const stamp = new Date().toISOString().slice(11, 19).replace(/:/g, '-');
      const name = `recording-${stamp}${recordingExtension(type)}`;
      const file = new File([blob], name, { type });
      addFiles([file]);
      try {
        await saveRecording({ name, blob, type, size: blob.size, createdAt: Date.now() });
        refreshSaved();
      } catch (err) {
        console.warn('Could not save recording locally:', err);
      }
    });
    rec.start();
    setRecording(true);
    startRecTimer();
  }

  async function useRecording(id: number) {
    const rec: any = await getRecording(id);
    if (!rec) return;
    addFiles([new File([rec.blob], rec.name, { type: rec.type })]);
  }
  async function deleteRecording(id: number) {
    if (!confirm('Delete this recording from your device?')) return;
    await deleteRecordingDb(id);
    refreshSaved();
  }
  async function deleteAllRecordings() {
    if (!savedRecs.length) return;
    if (!confirm(`Delete all ${savedRecs.length} saved recording(s) from your device?`)) return;
    await clearRecordingsDb();
    refreshSaved();
  }

  // ── grade ──
  function hasTaskContext() {
    if (!qt || !qt.required) return true;
    const present = new Set<string>();
    if (promptText.trim()) present.add('prompt');
    if (referenceText.trim()) present.add('reference');
    if (imageFile) present.add('image');
    return qt.required.some((r) => present.has(r));
  }
  function appendCommonFields(fd: FormData) {
    if (showReference && referenceText) fd.append('text', referenceText);
    if (showPrompt && promptText) fd.append('prompt', promptText);
    fd.append('exam', exam);
    if (questionType) fd.append('question_type', questionType);
    fd.append('mode', mode);
    fd.append('accent', accent);
    if (feedbackLang) fd.append('feedback_lang', feedbackLang);
    if (expectedDuration) fd.append('expected_duration_sec', expectedDuration);
    if (showImage && imageFile) fd.append('image', imageFile);
    fd.append('no_ai', String(noAi));
    if (historySaveEnabled()) fd.append('user_id', getUserId());
  }
  async function grade() {
    if (files.length === 0) {
      alert('Please select at least one audio file');
      return;
    }
    if (!hasTaskContext()) {
      const ok = confirm(
        '⚠️ Chưa nhập đề/câu hỏi cho dạng câu này nên không thể chấm điểm tổng — chỉ chấm phát âm.\n\n' +
          'Nhấn OK để vẫn chấm phát âm, hoặc Cancel để quay lại nhập đề bài.',
      );
      if (!ok) return;
    }
    const isBatch = files.length > 1;
    setGrading(true);
    const fd = new FormData();
    if (isBatch) {
      files.forEach((f) => fd.append('audios', f));
      batchFilesRef.current = files;
    } else {
      fd.append('audio', files[0]);
      setSingleFilename(files[0].name);
      if (singleUrlRef.current) URL.revokeObjectURL(singleUrlRef.current);
      singleUrlRef.current = URL.createObjectURL(files[0]);
      setPlaybackUrl(singleUrlRef.current);
      setPlaybackUrlFn(() => singleUrlRef.current);
    }
    appendCommonFields(fd);
    const endpoint = isBatch ? '/grade-batch' : '/grade';
    try {
      const res = await apiFetch(endpoint, { method: 'POST', body: fd });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          detail = (await res.json()).detail || detail;
        } catch {
          /* */
        }
        throw new Error(detail);
      }
      const data = await res.json();
      if (isBatch) {
        setSingleData(null);
        setBatchData(data);
      } else {
        setBatchData(null);
        setSingleData(data);
      }
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    } finally {
      setGrading(false);
    }
  }

  // ── single result HTML (features/scores/telemetry qua renderer legacy) ──
  const singleHtml = useMemo(() => {
    if (!singleData) return null;
    const d = singleData;
    const c = examConfig(d.exam);
    const pronOnly = !!d.pronunciation_only;
    return {
      scoreLabel: pronOnly ? 'Chỉ chấm phát âm (chưa có đề)' : c.overallLabel,
      scoreVal: pronOnly ? '--' : d.scores?.[c.scoreField] ?? '--',
      transcript: d.transcript || 'No transcript available',
      features: featureGridHtml(d.features || {}),
      scores: scoresBreakdownHtml(d.scores, d.exam, d.phoneme, {
        pronunciationOnly: pronOnly,
        notice: d.notice,
        playback: true,
      }),
      feedback: d.scores?.summary_feedback || (pronOnly ? d.notice || '' : 'No feedback available'),
      telemetry: telemetryHtml(d.telemetry),
    };
    // accent trong deps → đổi giọng re-render kết quả từ data gốc (không chấm lại).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [singleData, accent]);

  const batchView = useMemo(() => {
    if (!batchData) return null;
    const c = examConfig(batchData.exam);
    const results = (batchData.results || []).slice().sort((a: any, b: any) => a.index - b.index);
    return { c, results };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [batchData, accent]);

  return (
    <div id="mode-classic">
      {/* Grading form */}
      <div className="card">
        <h2>
          📝 Grading <span style={{ fontWeight: 400, fontSize: '0.85rem', color: '#888' }}>— 1 file = single · 2+ files = batch</span>
        </h2>

        <div className="row">
          <div className="form-group">
            <label htmlFor="exam">Exam</label>
            <select id="exam" value={exam} onChange={(e) => setExam(e.target.value)}>
              <option value="toeic">TOEIC</option>
              <option value="ielts">IELTS</option>
              <option value="topik">TOPIK 말하기 (tiếng Hàn)</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="question-type">Question Type</label>
            <select id="question-type" value={questionType} onChange={(e) => setQuestionType(e.target.value)}>
              {questionTypes.map((q) => (
                <option key={q.value} value={q.value}>
                  {q.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-group">
          <label className={'file-upload' + (files.length ? ' has-file' : '')}>
            <input
              type="file"
              multiple
              accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm,.weba,.mp4,.mov,.mkv,.avi"
              onChange={(e) => {
                addFiles(Array.from(e.target.files || []));
                e.target.value = '';
              }}
            />
            <div className="file-upload-text">
              <span className="icon">📁</span>
              <span>Click to select audio file(s) — pick several to grade a whole batch</span>
            </div>
          </label>
          <div className="recorder">
            <button type="button" className={'btn btn-secondary' + (recording ? ' recording' : '')} onClick={toggleRecording}>
              {recording ? '⏹ Stop recording' : '🎙️ Record audio'}
            </button>
            <span className="record-timer">
              {recording ? `● ${Math.floor(recSeconds / 60)}:${String(recSeconds % 60).padStart(2, '0')}` : ''}
            </span>
          </div>
          <div className="file-list">
            {files.length > 1 && (
              <div className="file-item" style={{ background: '#eef2ff', color: '#3730a3', fontWeight: 600 }}>
                📦 {files.length} files — will be graded as a batch
              </div>
            )}
            {files.map((f, i) => (
              <div className="file-item file-item-audio" key={i}>
                <span className="name">📄 {f.name}</span>
                <audio controls preload="metadata" src={URL.createObjectURL(f)} />
              </div>
            ))}
            {files.length > 0 && (
              <button className="btn btn-secondary" onClick={clearFiles} style={{ marginTop: '0.5rem', width: 'auto', padding: '0.4rem 0.9rem' }}>
                Clear
              </button>
            )}
          </div>
        </div>

        {showReference && (
          <div className="form-group" id="reference-group">
            <label htmlFor="reference-text">Reference Script (optional - for Read Aloud)</label>
            <textarea id="reference-text" value={referenceText} onChange={(e) => setReferenceText(e.target.value)} placeholder="Enter the reference transcript here..." />
          </div>
        )}

        {showImage && (
          <div className="form-group" id="image-group">
            <label className={'file-upload' + (imageFile ? ' has-file' : '')}>
              <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] || null)} />
              <div className="file-upload-text">
                <span className="icon">🖼️</span>
                <span>Click to upload a picture (optional — for Describe Picture)</span>
              </div>
            </label>
            {imageFile && (
              <div className="image-preview">
                <img src={URL.createObjectURL(imageFile)} alt={imageFile.name} className="preview-img" />
                <button className="btn btn-secondary" onClick={() => setImageFile(null)} style={{ width: 'auto', padding: '0.4rem 0.9rem' }}>
                  Clear image
                </button>
              </div>
            )}
          </div>
        )}

        {showPrompt && (
          <div className="form-group" id="prompt-group">
            <label htmlFor="prompt-text">Prompt (optional)</label>
            <textarea id="prompt-text" value={promptText} onChange={(e) => setPromptText(e.target.value)} placeholder="Enter the prompt shown to the test-taker..." />
          </div>
        )}

        {/* Gợi ý bài nói mẫu cho dạng câu mở (không phải read_aloud/auto) */}
        {questionType && questionType !== 'read_aloud' && (
          <SuggestPanel
            exam={exam}
            questionType={questionType}
            promptText={promptText}
            expectedDuration={expectedDuration}
            feedbackLang={feedbackLang}
            imageFile={showImage ? imageFile : null}
            accent={accent}
            examLang={cfg.lang}
          />
        )}

        <div className="row">
          <div className="form-group">
            <label htmlFor="mode">Grading Mode</label>
            <select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="practice">Practice</option>
              <option value="mock_test">Mock Test</option>
            </select>
            <p className="mode-note">{MODE_NOTES[mode]}</p>
          </div>
          <div className="form-group">
            <label htmlFor="expected-duration">Expected Duration (sec)</label>
            <input type="number" id="expected-duration" value={expectedDuration} onChange={(e) => setExpectedDuration(e.target.value)} placeholder="e.g. 60" min={1} />
          </div>
        </div>

        <div className="row">
          <div className="form-group">
            <label htmlFor="feedback-lang">Feedback Language</label>
            <select id="feedback-lang" value={feedbackLang} onChange={(e) => setFeedbackLang(e.target.value)}>
              <option value="">Default</option>
              <option value="vi">Vietnamese</option>
              <option value="en">English</option>
              <option value="ko">Korean (한국어)</option>
            </select>
          </div>
          {showAccent && (
            <div className="form-group" id="accent-group">
              <label htmlFor="accent">Pronunciation Reference</label>
              <select id="accent" className="accent-select" value={accent} onChange={(e) => setAccent(e.target.value as any)}>
                <option value="default">Tự động (default)</option>
                <option value="gb">Anh-Anh (British)</option>
                <option value="us">Anh-Mỹ (American)</option>
              </select>
            </div>
          )}
        </div>

        <div className="checkbox-group" style={{ marginBottom: '1.5rem' }}>
          <input type="checkbox" id="no-ai" checked={noAi} onChange={(e) => setNoAi(e.target.checked)} />
          <label htmlFor="no-ai">ASR only (skip AI scoring)</label>
        </div>

        <button className="btn btn-primary" id="grade-btn" onClick={grade} disabled={grading}>
          {grading ? (files.length > 1 ? `Grading ${files.length} files...` : 'Grading...') : 'Grade Now'}
        </button>
      </div>

      {/* Saved recordings (IndexedDB) */}
      {savedRecs.length > 0 && (
        <div className="card" id="saved-recordings-card">
          <div className="result-header">
            <h2>
              💾 Saved Recordings <span style={{ fontWeight: 400, fontSize: '0.85rem', color: '#888' }}>— stored on this device</span>
            </h2>
            <button className="btn btn-secondary" onClick={deleteAllRecordings} style={{ width: 'auto', padding: '0.5rem 1rem' }}>
              Delete all
            </button>
          </div>
          <div className="file-list">
            {savedRecs.map((rec) => (
              <div className="file-item file-item-audio" key={rec.id}>
                <div className="saved-rec-head">
                  <span className="name">📄 {rec.name}</span>
                  <span className="saved-rec-meta">
                    {new Date(rec.createdAt).toLocaleString()}
                    {rec.size ? ' · ' + formatBytes(rec.size) : ''}
                  </span>
                </div>
                <audio controls preload="metadata" src={URL.createObjectURL(rec.blob)} />
                <div className="saved-rec-actions">
                  <button className="btn btn-secondary" onClick={() => useRecording(rec.id)} style={{ width: 'auto', padding: '0.35rem 0.8rem' }}>
                    ➕ Add to grading
                  </button>
                  <button className="btn btn-secondary remove-btn" onClick={() => deleteRecording(rec.id)} style={{ width: 'auto', padding: '0.35rem 0.8rem' }}>
                    🗑 Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Single result */}
      {singleData && singleHtml && (
        <div className="result visible" id="result">
          <div className="card">
            <div className="result-header">
              <h2>📊 Results</h2>
              <div className="actions">
                <button
                  className="btn btn-secondary"
                  onClick={() => singleUrlRef.current && downloadBlob(files[0] || new Blob(), singleData.audio_filename || singleFilename || 'recording')}
                  style={{ width: 'auto', padding: '0.5rem 1rem' }}
                >
                  ⬇ Tải audio
                </button>
                <button className="btn btn-secondary" onClick={() => printSingleReport(singleData, singleFilename)} style={{ width: 'auto', padding: '0.5rem 1rem' }}>
                  🖨 Print / PDF
                </button>
                <button className="btn btn-secondary" onClick={() => setSingleData(null)} style={{ width: 'auto', padding: '0.5rem 1rem' }}>
                  Close
                </button>
              </div>
            </div>
            <div className="score-display">
              <div className="label">{singleHtml.scoreLabel}</div>
              <div className="score">{singleHtml.scoreVal}</div>
            </div>
            <div className="result-section">
              <h3>📝 Transcript</h3>
              <p>{singleHtml.transcript}</p>
            </div>
            <div className="result-section">
              <h3>📈 Features</h3>
              <div className="features-grid" dangerouslySetInnerHTML={{ __html: singleHtml.features }} />
            </div>
            <div className="result-section">
              <h3>📋 Scores Breakdown</h3>
              <div dangerouslySetInnerHTML={{ __html: singleHtml.scores }} />
            </div>
            <div className="result-section">
              <h3>💬 Feedback</h3>
              <p>{singleHtml.feedback}</p>
            </div>
            <div className="result-section">
              <h3>🔍 Telemetry</h3>
              <div dangerouslySetInnerHTML={{ __html: singleHtml.telemetry }} />
            </div>
          </div>
        </div>
      )}

      {/* Batch result */}
      {batchData && batchView && (
        <div className="result visible" id="batch-result">
          <div className="card">
            <div className="result-header">
              <h2>📊 Batch Results</h2>
              <div className="actions">
                <button
                  className="btn btn-secondary"
                  onClick={() =>
                    downloadBlobsSequentially(
                      batchFilesRef.current.map((f, i) => ({ blob: f, filename: `${String(i + 1).padStart(2, '0')}-${f.name || 'recording'}` })),
                    )
                  }
                  style={{ width: 'auto', padding: '0.5rem 1rem' }}
                >
                  ⬇ Tải tất cả audio
                </button>
                <button className="btn btn-primary" onClick={() => printBatchReport(batchData)} style={{ width: 'auto', padding: '0.5rem 1rem' }}>
                  🖨 Print / PDF
                </button>
                <button className="btn btn-secondary" onClick={() => setBatchData(null)} style={{ width: 'auto', padding: '0.5rem 1rem' }}>
                  Close
                </button>
              </div>
            </div>
            <div id="batch-summary">
              <div className={'status-bar ' + (batchData.failed ? 'info' : 'success')} style={{ justifyContent: 'center' }}>
                <span>
                  {batchData.succeeded}/{batchData.count} graded
                  {batchData.failed ? ` · ${batchData.failed} failed` : ''} · exam: {batchView.c.label} · type: {batchData.question_type} · mode:{' '}
                  {batchData.mode_requested}
                </span>
              </div>
            </div>
            <div id="batch-results-list">
              {batchView.results.map((item: any) => (
                <BatchItem key={item.index} item={item} exam={batchData.exam} file={batchFilesRef.current[item.index]} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Một dòng kết quả batch (details) — scoresBreakdown playback:false (batch không phát
// lại từng từ để khỏi phát nhầm file khác, giống legacy).
function BatchItem({ item, exam, file }: { item: any; exam: string; file?: File }) {
  const c = examConfig(exam);
  if (item.error) {
    return (
      <div className="batch-result">
        <div className="filename">
          📄 {item.audio_filename}{' '}
          {file && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => downloadBlob(file, `${String(item.index + 1).padStart(2, '0')}-${file.name || 'recording'}`)}
              style={{ width: 'auto', padding: '0.2rem 0.6rem', fontSize: '0.85rem' }}
            >
              ⬇
            </button>
          )}
        </div>
        <div className="batch-error">❌ {item.error}</div>
      </div>
    );
  }
  const r = item.result || {};
  const pronOnly = !!r.pronunciation_only;
  const score = pronOnly ? '🔊' : r.scores?.[c.scoreField] ?? '--';
  const feedback = r.scores?.summary_feedback || (pronOnly ? r.notice : '');
  const inner =
    `<div style="font-weight:600;color:#333;margin-bottom:0.3rem;">Transcript</div>` +
    `<p style="color:#555;line-height:1.5;white-space:pre-wrap;">${escapeForBatch(r.transcript || '(empty)')}</p>` +
    `<div class="features-grid" style="margin-top:0.85rem;">${featureGridHtml(r.features || {})}</div>` +
    `<div style="margin-top:0.85rem;">${scoresBreakdownHtml(r.scores, r.exam ?? exam, r.phoneme, {
      pronunciationOnly: pronOnly,
      notice: r.notice,
    })}</div>` +
    (feedback
      ? `<div style="font-weight:600;color:#333;margin:0.85rem 0 0.3rem;">Feedback</div><p style="color:#555;line-height:1.6;white-space:pre-wrap;">${escapeForBatch(feedback)}</p>`
      : '');
  return (
    <details className="batch-result">
      <summary style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.75rem', listStyle: 'none' }}>
        <span className="batch-score" style={{ margin: 0 }}>
          {score}
        </span>
        <span className="filename" style={{ margin: 0, flex: 1 }}>
          📄 {item.audio_filename}
        </span>
        {file && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={(e) => {
              e.preventDefault();
              downloadBlob(file, `${String(item.index + 1).padStart(2, '0')}-${file.name || 'recording'}`);
            }}
            style={{ width: 'auto', padding: '0.2rem 0.6rem', fontSize: '0.85rem' }}
          >
            ⬇
          </button>
        )}
        <span style={{ color: '#888', fontSize: '0.85rem' }}>▼ details</span>
      </summary>
      <div style={{ marginTop: '0.85rem' }} dangerouslySetInnerHTML={{ __html: inner }} />
    </details>
  );
}

function escapeForBatch(s: unknown): string {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] as string));
}
