// 👾 BossSession — thử thách nói TỔNG HỢP cuối chặng (Phase 3A). Đọc to
// `reference_text` (gộp từ/đoạn của cả Unit) → chấm qua ĐÚNG gradePronunciation dùng
// chung (không engine chấm mới). Đạt ngưỡng → hạ Boss (onDefeated). Bonus-only: XP/xu/
// huy hiệu do BossView xử lý server-side, KHÔNG đụng mastery. UI thanh máu Boss quanh
// đúng lời gọi chấm đó — không thêm đường chấm thứ hai.

import { useRef, useState } from 'react';
import { useUiStore } from '@/store/ui';
import { gradePronunciation } from '../gradePron';
import type { BossContent } from '../courseApi';

type Stage = 'idle' | 'recording' | 'grading';

export default function BossSession({
  boss,
  onDefeated,
}: {
  boss: BossContent;
  onDefeated: (score: number) => void;
}) {
  const accent = useUiStore((s) => s.accent);
  const [stage, setStage] = useState<Stage>('idle');
  const [status, setStatus] = useState('');
  const [lastPct, setLastPct] = useState<number | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const ctx = { id: boss.boss_id, exam: boss.exam };
  const targetPct = Math.round(boss.threshold * 100);
  // Máu Boss còn lại = 100 − % đọc đúng lần gần nhất (chỉ minh họa; hạ khi đạt ngưỡng).
  const bossHp = lastPct == null ? 100 : Math.max(0, 100 - lastPct);

  function stopStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function toggle() {
    if (stage === 'grading') return;
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
      setStage('idle');
      setStatus('Ghi âm quá ngắn — bấm 🎙️, đọc to cả đoạn rồi bấm dừng.');
      return;
    }
    setStage('grading');
    setStatus('Đang chấm…');
    try {
      const pct = await gradePronunciation(ctx, boss.reference_text, blob, mime, accent);
      setStage('idle');
      if (pct == null) {
        setStatus('Chưa nghe rõ — hãy đọc to, rõ rồi thử lại.');
        return;
      }
      setLastPct(pct);
      if (pct / 100 >= boss.threshold) {
        setStatus(`Hạ Boss! ${pct}% 🎉`);
        onDefeated(pct / 100);
      } else {
        setStatus(`Được ${pct}% — cần ${targetPct}% để hạ Boss. Thử lại nhé!`);
      }
    } catch (e: any) {
      setStage('idle');
      setStatus(`Lỗi chấm: ${e.message || e}`);
    }
  }

  const recording = stage === 'recording';
  return (
    <div className="course-game">
      <div className="course-section-label">👾 {boss.title} — đọc to để hạ Boss</div>

      {/* Thanh máu Boss */}
      <div className="course-boss-hp" aria-label={`Máu Boss ${bossHp}%`}>
        <div className="course-boss-hp__fill" style={{ width: `${bossHp}%` }} />
      </div>

      <div className="course-boss-words">{boss.reference_text || 'Chưa dựng được nội dung Boss.'}</div>

      <div className="course-complete">
        <div className="course-complete__hint">
          Đọc to, rõ tất cả. Đạt {targetPct}% âm đúng để hạ Boss.
          {boss.best_score != null && ` · Kỷ lục: ${Math.round(boss.best_score * 100)}%`}
        </div>
        <button
          type="button"
          className={'practice-mic' + (recording ? ' recording' : '')}
          onClick={toggle}
          disabled={stage === 'grading' || !boss.reference_text}
          title="Ghi âm đọc đoạn Boss"
        >
          🎙️
        </button>
        {status && <div className="course-complete__status">{status}</div>}
      </div>
    </div>
  );
}
