// 📖 StorySession — Quest truyện đọc-to tuyến tính (Phase 3C). Truyện chia đoạn; học
// viên đọc to LẦN LƯỢT từng đoạn (KHÔNG nhánh/lựa chọn), chấm qua ĐÚNG gradePronunciation
// dùng chung (text = đoạn). Thanh tiến độ + combo chỉ là hiệu ứng (KHÔNG cấp XP/đoạn).
// Xong hết đoạn → onCompleted(avgScore); XP/xu/huy hiệu do StoryView xử lý server-side
// MỘT LẦN (bonus-only). Reuse khuôn RolePlaySession (session ⇢ ghi âm + hàm chấm chung).

import { useEffect, useRef, useState } from 'react';
import { Volume2 } from 'lucide-react';
import { playWordTts } from '@/features/grading/playback';
import { playSfx } from '@/lib/sfx';
import { gradePronunciation, type GradeContext } from '../gradePron';
import type { StoryQuest } from '../courseApi';

type Stage = 'reading' | 'recording' | 'grading' | 'revealed';

export default function StorySession({
  story,
  accent,
  onCompleted,
}: {
  story: StoryQuest;
  accent: string;
  onCompleted: (avgScore: number) => void;
}) {
  const [segIdx, setSegIdx] = useState(0);
  const [stage, setStage] = useState<Stage>('reading');
  const [status, setStatus] = useState('');
  const [lastPct, setLastPct] = useState<number | null>(null);
  const [combo, setCombo] = useState(0);
  const scoresRef = useRef<number[]>([]);
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const segments = story.segments;
  const seg = segments[segIdx];
  const ctx: GradeContext = { id: story.quest_id, exam: story.exam };
  const targetPct = Math.round(story.threshold * 100);
  const pctDone = Math.round((segIdx / segments.length) * 100);

  // Tự phát đoạn mẫu khi vào đoạn mới (autoplay bị chặn thì im lặng — còn nút 🔊).
  useEffect(() => {
    if (!seg) return;
    const t = window.setTimeout(() => playWordTts(seg.text), 350);
    return () => window.clearTimeout(t);
  }, [segIdx]); // eslint-disable-line react-hooks/exhaustive-deps

  function stopStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function toggle() {
    if (stage === 'grading' || stage === 'revealed') return;
    if (stage === 'recording') {
      recRef.current?.stop();
      return;
    }
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setStatus('Không truy cập được micro — kiểm tra quyền trình duyệt.');
      return;
    }
    chunksRef.current = [];
    const rec = new MediaRecorder(streamRef.current);
    recRef.current = rec;
    rec.addEventListener('dataavailable', (e) => e.data?.size && chunksRef.current.push(e.data));
    rec.addEventListener('stop', () => {
      const mime = rec.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      stopStream();
      grade(blob, mime);
    });
    rec.start();
    setStage('recording');
    setStatus('Đang ghi âm… bấm lần nữa để dừng.');
  }

  async function grade(blob: Blob, mime: string) {
    if (!blob || blob.size < 1024) {
      setStage('reading');
      setStatus('Ghi âm quá ngắn — bấm 🎙️, đọc to cả đoạn rồi bấm dừng.');
      return;
    }
    setStage('grading');
    setStatus('Đang chấm…');
    try {
      const pct = await gradePronunciation(ctx, seg.text, blob, mime, accent);
      const val = pct == null ? 0 : pct;
      scoresRef.current.push(val / 100);
      setLastPct(pct);
      const passed = pct != null && pct / 100 >= story.threshold;
      if (passed) {
        setCombo((c) => c + 1);
        playSfx('correct');
        setStatus(`Tốt — ${pct}%! 🎉`);
      } else {
        setCombo(0);
        setStatus(pct == null ? 'Chưa nghe rõ — thử lại hoặc đọc tiếp.' : `Được ${pct}% — cứ tiếp tục.`);
      }
      setStage('revealed');
    } catch (e: any) {
      scoresRef.current.push(0);
      setLastPct(null);
      setCombo(0);
      setStatus(`Lỗi chấm: ${e.message || e}`);
      setStage('revealed');
    }
  }

  function next() {
    if (segIdx + 1 < segments.length) {
      setSegIdx((i) => i + 1);
      setStage('reading');
      setStatus('');
      setLastPct(null);
      return;
    }
    const scores = scoresRef.current;
    const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    onCompleted(avg);
  }

  function retry() {
    scoresRef.current.pop();
    setStage('reading');
    setStatus('');
    setLastPct(null);
  }

  if (!seg) {
    return <div className="course-game">Chưa dựng được truyện.</div>;
  }

  const recording = stage === 'recording';
  const revealed = stage === 'revealed';
  const isLast = segIdx + 1 >= segments.length;

  return (
    <div className="course-game flex flex-col gap-4">
      <div className="course-section-label">📖 {story.title}</div>

      {/* Thanh tiến độ đọc truyện */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-gradient-to-r from-sky-400 to-indigo-500 transition-all"
            style={{ width: `${pctDone}%` }}
          />
        </div>
        <span className="tabular-nums">
          Đoạn {segIdx + 1}/{segments.length}
          {combo >= 2 && <span className="ml-2 font-bold text-amber-500">🔥 x{combo}</span>}
        </span>
      </div>

      {/* Đoạn truyện + nút nghe */}
      <div className="flex items-start gap-2 rounded-xl border bg-card p-4">
        <p className="min-w-0 flex-1 text-base leading-relaxed">{seg.text}</p>
        <button
          type="button"
          onClick={() => playWordTts(seg.text)}
          className="mt-0.5 shrink-0 rounded-full p-1.5 text-primary hover:bg-accent"
          title="Nghe đoạn mẫu"
        >
          <Volume2 className="h-5 w-5" />
        </button>
      </div>

      {/* Mic + trạng thái */}
      <div className="course-complete flex flex-col items-center gap-2">
        {!revealed && (
          <div className="course-complete__hint text-center">
            Bấm 🎙️ và đọc to đoạn trên. Đạt {targetPct}% âm đúng để qua đoạn.
          </div>
        )}
        {!revealed ? (
          <button
            type="button"
            className={'practice-mic' + (recording ? ' recording' : '')}
            onClick={toggle}
            disabled={stage === 'grading'}
            title="Ghi âm đọc đoạn"
          >
            🎙️
          </button>
        ) : (
          <div className="flex items-center gap-3">
            {lastPct == null || lastPct / 100 < story.threshold ? (
              <button type="button" className="btn btn-secondary" onClick={retry}>
                Đọc lại đoạn này
              </button>
            ) : null}
            <button type="button" className="btn btn-primary" onClick={next}>
              {isLast ? '🏁 Kết thúc truyện' : 'Đoạn tiếp →'}
            </button>
          </div>
        )}
        {status && <div className="course-complete__status text-center">{status}</div>}
      </div>
    </div>
  );
}
