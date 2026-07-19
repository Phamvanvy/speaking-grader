import { useEffect, useRef, useState } from 'react';
import { apiFetch } from '../../lib/api';

// Port web/js/suggest.js — gợi ý bài nói mẫu band cao cho dạng câu MỞ + đọc bằng
// Web Speech API (xử lý được văn bản dài, không đụng /tts backend giới hạn 100 ký tự).

const TARGET_BAND_OPTIONS: Record<string, { value: string; label: string }[]> = {
  ielts: [
    { value: '9.0', label: 'Band 9.0 (cao nhất)' },
    { value: '8.0', label: 'Band 8.0' },
    { value: '7.0', label: 'Band 7.0' },
    { value: '6.5', label: 'Band 6.5' },
  ],
  toeic: [
    { value: 'TOEIC mức cao nhất (~200)', label: 'Mức cao nhất (~200)' },
    { value: 'TOEIC mức khá (~160)', label: 'Mức khá (~160)' },
  ],
  topik: [
    { value: 'TOPIK 말하기 mức cao nhất (6급, ~190/200)', label: '6급 (cao nhất)' },
    { value: 'TOPIK 말하기 mức khá (4-5급, ~130-170/200)', label: '4–5급 (khá)' },
  ],
};

interface Props {
  exam: string;
  questionType: string;
  promptText: string;
  expectedDuration: string;
  feedbackLang: string;
  imageFile: File | null;
  accent: string;
  examLang: string;
}

export function SuggestPanel(props: Props) {
  const bandOpts = TARGET_BAND_OPTIONS[props.exam] || TARGET_BAND_OPTIONS.toeic;
  const [targetBand, setTargetBand] = useState(bandOpts[0].value);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<any>(null);
  const [speaking, setSpeaking] = useState(false);
  const sampleRef = useRef('');

  // Đổi exam → reset band về option đầu (cao nhất).
  useEffect(() => {
    setTargetBand(bandOpts[0].value);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.exam]);

  // Dọn khi rời dạng câu mở (component unmount) — dừng đọc.
  useEffect(() => {
    return () => {
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    };
  }, []);

  async function suggest() {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const fd = new FormData();
      fd.append('exam', props.exam);
      fd.append('question_type', props.questionType);
      if (props.promptText) fd.append('prompt', props.promptText);
      if (props.expectedDuration) fd.append('expected_duration_sec', props.expectedDuration);
      if (targetBand) fd.append('target_band', targetBand);
      if (props.feedbackLang) fd.append('feedback_lang', props.feedbackLang);
      if (props.imageFile) fd.append('image', props.imageFile);
      const res = await apiFetch('/suggest', { method: 'POST', body: fd, noRetry: true });
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
      sampleRef.current = data.answer || '';
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function voiceLang() {
    if (props.examLang === 'ko') return 'ko-KR';
    if (props.accent === 'gb') return 'en-GB';
    return 'en-US';
  }
  function toggleSpeak() {
    const synth = window.speechSynthesis;
    if (!synth) {
      alert('Trình duyệt không hỗ trợ đọc văn bản (Web Speech API).');
      return;
    }
    if (synth.speaking || synth.pending) {
      synth.cancel();
      setSpeaking(false);
      return;
    }
    if (!sampleRef.current) return;
    const utter = new SpeechSynthesisUtterance(sampleRef.current);
    utter.lang = voiceLang();
    utter.rate = 0.95;
    utter.onend = () => setSpeaking(false);
    utter.onerror = () => setSpeaking(false);
    setSpeaking(true);
    synth.speak(utter);
  }

  const outline = Array.isArray(result?.outline) ? result.outline : [];
  const highlights = Array.isArray(result?.highlights) ? result.highlights : [];

  return (
    <div className="form-group" id="suggest-group">
      <label htmlFor="target-band">Bài nói mẫu (tham khảo)</label>
      <div className="suggest-controls">
        <select id="target-band" value={targetBand} onChange={(e) => setTargetBand(e.target.value)}>
          {bandOpts.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <button type="button" className="btn btn-secondary" onClick={suggest} disabled={loading}>
          {loading ? '⏳ Đang sinh bài mẫu...' : '💡 Gợi ý bài mẫu'}
        </button>
      </div>
      <p className="mode-note">
        Sinh một bài nói mẫu chất lượng cao cho đề + dạng câu hiện tại (không tính điểm). Có thể bấm 🔊 để nghe đọc.
      </p>
      {(loading || error || result) && (
        <div id="suggest-result">
          {loading && (
            <div className="status-bar info">
              <div className="spinner"></div>
              <span>Đang sinh bài mẫu…</span>
            </div>
          )}
          {error && (
            <div className="status-bar error">
              <span>❌ {error}</span>
            </div>
          )}
          {result && !loading && (
            <div className="suggest-card">
              <div className="suggest-head">
                <span className="suggest-band">🎯 {result.target_band || ''}</span>
                <button type="button" className="btn btn-secondary" onClick={toggleSpeak} style={{ width: 'auto', padding: '0.4rem 0.9rem' }}>
                  {speaking ? '⏹ Dừng' : '🔊 Nghe'}
                </button>
              </div>
              <div className="suggest-answer">{sampleRef.current}</div>
              {outline.length > 0 && (
                <div className="suggest-section">
                  <h4>Dàn ý</h4>
                  <ul>
                    {outline.map((o: string, i: number) => (
                      <li key={i}>{o}</li>
                    ))}
                  </ul>
                </div>
              )}
              {highlights.length > 0 && (
                <div className="suggest-section">
                  <h4>Điểm nhấn đáng học</h4>
                  <ul>
                    {highlights.map((h: string, i: number) => (
                      <li key={i}>{h}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
