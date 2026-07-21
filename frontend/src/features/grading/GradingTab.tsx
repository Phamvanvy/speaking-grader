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
import { printSingleReport, printBatchReport, downloadBlob } from '../../legacy/report';
import { downloadZipFromBlobs } from '../../lib/zip';
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

const AUDIO_EXT_RE = /\.(wav|mp3|m4a|ogg|oga|flac|webm|weba|mp4|mov|mkv|avi|aac|opus)$/i;

// Kéo-thả từ Explorer: MIME hay rỗng/sai với .m4a/.weba → fallback theo đuôi file.
function isAudioFile(f: File): boolean {
  const t = (f.type || '').toLowerCase();
  if (t.startsWith('audio/') || t.startsWith('video/')) return true;
  return AUDIO_EXT_RE.test(f.name || '');
}

// Ảnh dán từ clipboard thường không có tên → đặt tên theo MIME cho server.
function pickPastedImage(dt: DataTransfer | null): File | null {
  if (!dt) return null;
  const files: File[] = Array.from(dt.files || []);
  if (!files.length && dt.items) {
    for (const it of Array.from(dt.items)) {
      if (it.kind !== 'file') continue;
      const f = it.getAsFile();
      if (f) files.push(f);
    }
  }
  for (const f of files) {
    const t = (f.type || '').toLowerCase();
    if (!t.startsWith('image/')) continue;
    if (f.name) return f;
    const ext = t.split('/')[1] || 'png';
    return new File([f], `paste.${ext}`, { type: t });
  }
  return null;
}

function filesFromDrop(e: React.DragEvent): File[] {
  return Array.from(e.dataTransfer?.files || []);
}

const MODE_NOTES: Record<string, string> = {
  practice:
    'Chấm nhanh (faster-whisper). Tự động chuyển sang pipeline Mock Test khi độ tin cậy / độ khớp thấp.',
  mock_test: 'Chính xác nhất: ASR tốt nhất (WhisperX) + phân tích âm vị. Chậm hơn — dùng khi cần điểm chuẩn.',
};

// Tên tiếng Việt của từng loại "đề bài" — dùng chung cho nhãn ô nhập và câu gợi ý
// "Dạng câu này cần …" nên chỉ khai báo một chỗ.
const INPUT_LABELS: Record<string, string> = {
  reference: 'đoạn văn mẫu',
  image: 'ảnh đề bài',
  prompt: 'câu hỏi / đề bài',
};

function requirementHint(qt?: { required?: string[] }): string {
  const req = qt?.required;
  if (!req || !req.length) return 'Có thể chấm ngay chỉ với file audio. Nhập thêm đề bài để chấm điểm tổng.';
  const names = req.map((r) => INPUT_LABELS[r] || r);
  const joined = names.length > 1 ? names.join(' hoặc ') : names[0];
  return `Dạng câu này cần ${joined} thì mới chấm được điểm tổng.`;
}

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
  const [dragOver, setDragOver] = useState<'audio' | 'image' | null>(null);

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
  // "Tự nhận dạng" mở cả 3 ô đề bài → với người mới trông như một tờ khai dài.
  // Gộp lại thành khối gấp gọn; dạng câu cụ thể thì hiện thẳng ô cần nhập.
  const genericTask = !qt || !qt.required || !qt.required.length;

  // Object URL cho <audio>/<img> preview: tạo trong useMemo (không phải mỗi lần
  // render như trước — mỗi render cũ rò một URL) và thu hồi khi danh sách đổi.
  const fileUrls = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => fileUrls.forEach(URL.revokeObjectURL), [fileUrls]);
  const imageUrl = useMemo(() => (imageFile ? URL.createObjectURL(imageFile) : null), [imageFile]);
  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl); }, [imageUrl]);
  const savedUrls = useMemo(() => savedRecs.map((r) => URL.createObjectURL(r.blob)), [savedRecs]);
  useEffect(() => () => savedUrls.forEach(URL.revokeObjectURL), [savedUrls]);

  // Đổi exam → reset questionType về option đầu của exam mới (tránh giữ giá trị exam cũ).
  useEffect(() => {
    setQuestionType(questionTypes[0]?.value ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exam]);

  // Nạp bản ghi đã lưu (IndexedDB) khi mở tab.
  useEffect(() => {
    refreshSaved();
  }, []);

  // Ctrl+V: dán ảnh đề (Describe Picture) hoặc file audio copy từ Explorer.
  // Chỉ chặn paste khi thật sự có file → dán text vào textarea vẫn như thường.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const dt = e.clipboardData;
      if (showImage) {
        const img = pickPastedImage(dt);
        if (img) {
          e.preventDefault();
          setImageFile(img);
          return;
        }
      }
      const audio = Array.from(dt?.files || []).filter(isAudioFile);
      if (audio.length) {
        e.preventDefault();
        addFiles(audio);
      }
    };
    document.addEventListener('paste', onPaste);
    return () => document.removeEventListener('paste', onPaste);
  }, [showImage]);

  // Thả file trượt ra ngoài vùng drop → mặc định trình duyệt mở file đó, mất trang.
  // Chặn ở window (tab này chỉ mount ở /grade nên không ảnh hưởng tab khác).
  useEffect(() => {
    const swallow = (e: DragEvent) => e.preventDefault();
    const onWindowDrop = (e: DragEvent) => {
      e.preventDefault();
      setDragOver(null);
    };
    window.addEventListener('dragover', swallow);
    window.addEventListener('drop', onWindowDrop);
    return () => {
      window.removeEventListener('dragover', swallow);
      window.removeEventListener('drop', onWindowDrop);
    };
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
  function removeFile(i: number) {
    setFiles((prev) => prev.filter((_, idx) => idx !== i));
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
    if (!confirm('Xoá bản ghi này khỏi máy?')) return;
    await deleteRecordingDb(id);
    refreshSaved();
  }
  async function deleteAllRecordings() {
    if (!savedRecs.length) return;
    if (!confirm(`Xoá tất cả ${savedRecs.length} bản ghi đã lưu khỏi máy?`)) return;
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
    // Thiếu audio: nút đã disabled; thiếu đề bài: đã cảnh báo tại chỗ ngay trên nút
    // (bỏ confirm() — người dùng đã đọc cảnh báo trước khi bấm).
    if (files.length === 0) return;
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
      transcript: d.transcript || 'Không nhận dạng được lời nói',
      features: featureGridHtml(d.features || {}),
      scores: scoresBreakdownHtml(d.scores, d.exam, d.phoneme, {
        pronunciationOnly: pronOnly,
        notice: d.notice,
        playback: true,
      }),
      feedback: d.scores?.summary_feedback || (pronOnly ? d.notice || '' : 'Chưa có nhận xét'),
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

  // Ô nhập đề bài — dùng lại cho cả 2 chỗ đặt (hiện thẳng / trong khối gấp gọn).
  const taskFields = (
    <>
      {showReference && (
        <div className="form-group" id="reference-group">
          <label htmlFor="reference-text">
            Đoạn văn mẫu (bài đọc)
            <span className={'gbadge ' + (qt?.required?.includes('reference') ? 'gbadge--req' : 'gbadge--opt')}>
              {qt?.required?.includes('reference') ? 'bắt buộc' : 'không bắt buộc'}
            </span>
          </label>
          <textarea
            id="reference-text"
            value={referenceText}
            onChange={(e) => setReferenceText(e.target.value)}
            placeholder="Dán đoạn văn học viên phải đọc to…"
          />
        </div>
      )}

      {showImage && (
        <div className="form-group" id="image-group">
          <label
            className={'file-upload' + (imageFile ? ' has-file' : '') + (dragOver === 'image' ? ' dragover' : '')}
            onDragOver={(e) => {
              e.preventDefault();
              e.dataTransfer.dropEffect = 'copy';
              setDragOver('image');
            }}
            onDragLeave={(e) => {
              // dragleave cũng bắn khi con trỏ đi vào phần tử con → chỉ tắt khi rời hẳn label.
              if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(null);
            }}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(null);
              const img = filesFromDrop(e).find((f) => (f.type || '').toLowerCase().startsWith('image/'));
              if (img) setImageFile(img);
            }}
          >
            <input type="file" accept="image/*" onChange={(e) => setImageFile(e.target.files?.[0] || null)} />
            <div className="file-upload-text">
              <span className="icon">🖼️</span>
              <span>Ảnh đề bài — kéo-thả, dán Ctrl+V, hoặc bấm để chọn</span>
              <span className="file-upload-hint">Dùng cho dạng câu tả tranh / nhìn tranh</span>
            </div>
          </label>
          {imageFile && imageUrl && (
            <div className="image-preview">
              <img src={imageUrl} alt={imageFile.name} className="preview-img" />
              <button className="btn btn-secondary" onClick={() => setImageFile(null)} style={{ width: 'auto', padding: '0.4rem 0.9rem' }}>
                Xoá ảnh
              </button>
            </div>
          )}
        </div>
      )}

      {showPrompt && (
        <div className="form-group" id="prompt-group">
          <label htmlFor="prompt-text">
            Câu hỏi / đề bài
            <span className={'gbadge ' + (qt?.required?.includes('prompt') ? 'gbadge--req' : 'gbadge--opt')}>
              {qt?.required?.includes('prompt') ? 'bắt buộc' : 'không bắt buộc'}
            </span>
          </label>
          <textarea
            id="prompt-text"
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            placeholder="Nhập câu hỏi/đề bài mà học viên nhìn thấy…"
          />
        </div>
      )}
    </>
  );

  return (
    <div id="mode-classic">
      {/* Grading form — 3 bước, tuỳ chọn ít dùng gấp vào cuối */}
      <div className="card">
        <h2>📝 Chấm bài</h2>
        <p className="gform-sub">
          Tải lên hoặc ghi âm bài nói để nhận điểm và phân tích phát âm. Chọn <strong>1 file</strong> để chấm một bài, hoặc{' '}
          <strong>nhiều file</strong> để chấm cả lớp một lượt.
        </p>

        {/* ── Bước 1 — kỳ thi & dạng câu ───────────────────────────────── */}
        <section className="gstep is-done">
          <div className="gstep__head">
            <span className="gstep__num">✓</span>
            <div>
              <h3 className="gstep__title">Chọn kỳ thi và dạng câu</h3>
              <p className="gstep__hint">{requirementHint(qt)}</p>
            </div>
          </div>
          <div className="gstep__body">
            <div className="row">
              <div className="form-group">
                <label htmlFor="exam">Kỳ thi</label>
                <select id="exam" value={exam} onChange={(e) => setExam(e.target.value)}>
                  <option value="toeic">TOEIC</option>
                  <option value="ielts">IELTS</option>
                  <option value="topik">TOPIK 말하기 (tiếng Hàn)</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="question-type">Dạng câu</label>
                <select id="question-type" value={questionType} onChange={(e) => setQuestionType(e.target.value)}>
                  {questionTypes.map((q) => (
                    <option key={q.value} value={q.value}>
                      {q.value === '' ? 'Tự nhận dạng' : q.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </section>

        {/* ── Bước 2 — audio ───────────────────────────────────────────── */}
        <section className={'gstep' + (files.length ? ' is-done' : '')}>
          <div className="gstep__head">
            <span className="gstep__num">{files.length ? '✓' : '2'}</span>
            <div>
              <h3 className="gstep__title">Thêm bài nói</h3>
              <p className="gstep__hint">Ghi âm trực tiếp, hoặc chọn file có sẵn (wav, mp3, m4a, webm…).</p>
            </div>
          </div>
          <div className="gstep__body">
            <div className="form-group">
              <label
                className={'file-upload' + (files.length ? ' has-file' : '') + (dragOver === 'audio' ? ' dragover' : '')}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'copy';
                  setDragOver('audio');
                }}
                onDragLeave={(e) => {
                  if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(null);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(null);
                  const dropped = filesFromDrop(e).filter(isAudioFile);
                  if (dropped.length) addFiles(dropped);
                }}
              >
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
                  <span>Kéo-thả hoặc bấm để chọn file audio</span>
                  <span className="file-upload-hint">Chọn nhiều file để chấm cả lớp cùng lúc</span>
                </div>
              </label>
              <div className="recorder">
                <button type="button" className={'btn btn-secondary' + (recording ? ' recording' : '')} onClick={toggleRecording}>
                  {recording ? '⏹ Dừng ghi âm' : '🎙️ Ghi âm'}
                </button>
                <span className="record-timer">
                  {recording ? `● ${Math.floor(recSeconds / 60)}:${String(recSeconds % 60).padStart(2, '0')}` : ''}
                </span>
              </div>
              <div className="file-list">
                {files.length > 1 && (
                  <div className="file-item" style={{ background: '#eef2ff', color: '#3730a3', fontWeight: 600 }}>
                    📦 {files.length} file — sẽ chấm cả loạt
                  </div>
                )}
                {files.map((f, i) => (
                  <div className="file-item file-item-audio" key={i}>
                    <div className="gfile-head">
                      <span className="name">📄 {f.name}</span>
                      <button type="button" className="gfile-remove" title="Bỏ file này" aria-label="Bỏ file này" onClick={() => removeFile(i)}>
                        ✕
                      </button>
                    </div>
                    <audio controls preload="metadata" src={fileUrls[i]} />
                  </div>
                ))}
                {files.length > 1 && (
                  <button className="btn btn-secondary" onClick={clearFiles} style={{ marginTop: '0.5rem', width: 'auto', padding: '0.4rem 0.9rem' }}>
                    Bỏ hết
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ── Bước 3 — đề bài ──────────────────────────────────────────── */}
        {(showReference || showImage || showPrompt) && (
          <section className={'gstep' + (hasTaskContext() && (referenceText.trim() || promptText.trim() || imageFile) ? ' is-done' : '')}>
            <div className="gstep__head">
              <span className="gstep__num">{hasTaskContext() && (referenceText.trim() || promptText.trim() || imageFile) ? '✓' : '3'}</span>
              <div>
                <h3 className="gstep__title">Nhập đề bài</h3>
                <p className="gstep__hint">
                  {genericTask
                    ? 'Không bắt buộc — bỏ trống thì hệ thống chỉ chấm phát âm.'
                    : 'Cần có đề bài thì AI mới chấm được nội dung câu trả lời.'}
                </p>
              </div>
            </div>
            <div className="gstep__body">
              {genericTask ? (
                // "Tự nhận dạng": mở sẵn nếu đã nhập gì đó, còn lại gấp gọn cho đỡ rối.
                <details className="gcollapse" open={!!(referenceText || promptText || imageFile)}>
                  <summary>Thêm đề bài (không bắt buộc)</summary>
                  {taskFields}
                </details>
              ) : (
                taskFields
              )}
            </div>
          </section>
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

        {/* ── Tuỳ chọn nâng cao (mặc định đã hợp lý) ───────────────────── */}
        <details className="gcollapse gadvanced">
          <summary>Tuỳ chọn nâng cao</summary>
          <div className="row">
            <div className="form-group">
              <label htmlFor="mode">Chế độ chấm</label>
              <select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="practice">Luyện tập (nhanh)</option>
                <option value="mock_test">Thi thử (chính xác nhất)</option>
              </select>
              <p className="mode-note">{MODE_NOTES[mode]}</p>
            </div>
            <div className="form-group">
              <label htmlFor="expected-duration">Thời lượng mong đợi (giây)</label>
              <input type="number" id="expected-duration" value={expectedDuration} onChange={(e) => setExpectedDuration(e.target.value)} placeholder="vd. 60" min={1} />
            </div>
          </div>

          <div className="row">
            <div className="form-group">
              <label htmlFor="feedback-lang">Ngôn ngữ nhận xét</label>
              <select id="feedback-lang" value={feedbackLang} onChange={(e) => setFeedbackLang(e.target.value)}>
                <option value="">Mặc định</option>
                <option value="vi">Tiếng Việt</option>
                <option value="en">English</option>
                <option value="ko">한국어</option>
              </select>
            </div>
            {showAccent && (
              <div className="form-group" id="accent-group">
                <label htmlFor="accent">Giọng chuẩn đối chiếu</label>
                <select id="accent" className="accent-select" value={accent} onChange={(e) => setAccent(e.target.value as any)}>
                  <option value="default">Tự động (mặc định)</option>
                  <option value="gb">Anh-Anh (British)</option>
                  <option value="us">Anh-Mỹ (American)</option>
                </select>
              </div>
            )}
          </div>

          <div className="checkbox-group">
            <input type="checkbox" id="no-ai" checked={noAi} onChange={(e) => setNoAi(e.target.checked)} />
            <label htmlFor="no-ai" style={{ marginBottom: 0 }}>Chỉ nhận dạng lời nói (bỏ qua chấm điểm AI)</label>
          </div>
        </details>

        {/* Cảnh báo TRƯỚC khi bấm, thay vì confirm() sau khi bấm. */}
        {files.length > 0 && !hasTaskContext() && (
          <p className="gnotice">⚠️ Chưa có {(qt?.required || []).map((r) => INPUT_LABELS[r] || r).join(' hoặc ')} nên sẽ chỉ chấm phát âm, không có điểm tổng.</p>
        )}

        <div className="gsubmit">
          <button className="btn btn-primary btn-cta" id="grade-btn" onClick={grade} disabled={grading || files.length === 0}>
            {grading ? (files.length > 1 ? `Đang chấm ${files.length} file…` : 'Đang chấm…') : files.length > 1 ? `Chấm ${files.length} bài` : 'Chấm bài'}
          </button>
          {files.length === 0 && <p className="gsubmit__hint">Hãy ghi âm hoặc chọn file audio ở bước 2 trước.</p>}
        </div>
      </div>

      {/* Saved recordings (IndexedDB) */}
      {savedRecs.length > 0 && (
        <div className="card" id="saved-recordings-card">
          <div className="result-header">
            <h2>
              💾 Bản ghi đã lưu <span style={{ fontWeight: 400, fontSize: '0.85rem', color: '#888' }}>— lưu trên máy này</span>
            </h2>
            <button className="btn btn-secondary" onClick={deleteAllRecordings} style={{ width: 'auto', padding: '0.5rem 1rem' }}>
              Xoá hết
            </button>
          </div>
          <div className="file-list">
            {savedRecs.map((rec, ri) => (
              <div className="file-item file-item-audio" key={rec.id}>
                <div className="saved-rec-head">
                  <span className="name">📄 {rec.name}</span>
                  <span className="saved-rec-meta">
                    {new Date(rec.createdAt).toLocaleString()}
                    {rec.size ? ' · ' + formatBytes(rec.size) : ''}
                  </span>
                </div>
                <audio controls preload="metadata" src={savedUrls[ri]} />
                <div className="saved-rec-actions">
                  <button className="btn btn-secondary" onClick={() => useRecording(rec.id)} style={{ width: 'auto', padding: '0.35rem 0.8rem' }}>
                    ➕ Đưa vào chấm
                  </button>
                  <button className="btn btn-secondary remove-btn" onClick={() => deleteRecording(rec.id)} style={{ width: 'auto', padding: '0.35rem 0.8rem' }}>
                    🗑 Xoá
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
              <h2>📊 Kết quả</h2>
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
                  Đóng
                </button>
              </div>
            </div>
            <div className="score-display">
              <div className="label">{singleHtml.scoreLabel}</div>
              <div className="score">{singleHtml.scoreVal}</div>
            </div>
            <div className="result-section">
              <h3>📝 Lời nói nhận dạng được</h3>
              <p>{singleHtml.transcript}</p>
            </div>
            <div className="result-section">
              <h3>📈 Chỉ số</h3>
              <div className="features-grid" dangerouslySetInnerHTML={{ __html: singleHtml.features }} />
            </div>
            <div className="result-section">
              <h3>📋 Chi tiết điểm</h3>
              <div dangerouslySetInnerHTML={{ __html: singleHtml.scores }} />
            </div>
            <div className="result-section">
              <h3>💬 Nhận xét</h3>
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
              <h2>📊 Kết quả cả loạt</h2>
              <div className="actions">
                <button
                  className="btn btn-secondary"
                  onClick={() =>
                    downloadZipFromBlobs(
                      batchFilesRef.current.map((f, i) => ({ blob: f, filename: `${String(i + 1).padStart(2, '0')}-${f.name || 'recording'}` })),
                      'audio-batch',
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
                  Đóng
                </button>
              </div>
            </div>
            <div id="batch-summary">
              <div className={'status-bar ' + (batchData.failed ? 'info' : 'success')} style={{ justifyContent: 'center' }}>
                <span>
                  {batchData.succeeded}/{batchData.count} bài đã chấm
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
