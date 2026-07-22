// 🎭 RolePlaySession — Quest hội thoại nhập vai (Phase 3B). NPC nói (TTS) → học viên
// đáp: CHỈ thấy `hint` trong lúc trả lời (ẩn câu mẫu `expected_user`), ghi âm rồi chấm
// qua ĐÚNG gradePronunciation dùng chung (text = expected_user) → chấm xong mới LỘ câu
// mẫu làm feedback. Combo chỉ là hiệu ứng thị giác (KHÔNG cấp XP/lượt). Xong hết lượt →
// onCompleted(avgScore); XP/xu/huy hiệu do QuestView xử lý server-side MỘT LẦN (bonus-only).

import { useEffect, useRef, useState } from 'react';
import { Volume2 } from 'lucide-react';
import { playWordTts } from '@/features/grading/playback';
import { playSfx } from '@/lib/sfx';
import { gradePronunciation, type GradeContext } from '../gradePron';
import type { RolePlayScript } from '../courseApi';

type Stage = 'answering' | 'recording' | 'grading' | 'revealed';

export default function RolePlaySession({
  script,
  accent,
  onCompleted,
}: {
  script: RolePlayScript;
  accent: string;
  onCompleted: (avgScore: number) => void;
}) {
  const [turnIdx, setTurnIdx] = useState(0);
  const [stage, setStage] = useState<Stage>('answering');
  const [status, setStatus] = useState('');
  const [lastPct, setLastPct] = useState<number | null>(null);
  const [combo, setCombo] = useState(0);
  const scoresRef = useRef<number[]>([]);
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const turns = script.turns;
  const turn = turns[turnIdx];
  const ctx: GradeContext = { id: script.quest_id, exam: script.exam };
  const targetPct = Math.round(script.threshold * 100);

  // Tự phát lời NPC khi vào lượt mới (autoplay bị chặn thì im lặng — còn nút 🔊).
  useEffect(() => {
    if (!turn) return;
    const t = window.setTimeout(() => playWordTts(turn.npc), 350);
    return () => window.clearTimeout(t);
  }, [turnIdx]); // eslint-disable-line react-hooks/exhaustive-deps

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
      setStage('answering');
      setStatus('Ghi âm quá ngắn — bấm 🎙️, nói câu trả lời rồi bấm dừng.');
      return;
    }
    setStage('grading');
    setStatus('Đang chấm…');
    try {
      const pct = await gradePronunciation(ctx, turn.expected_user, blob, mime, accent);
      // Điểm 0 khi không nghe rõ (null) — vẫn tính vào trung bình để không farm skip.
      const val = pct == null ? 0 : pct;
      scoresRef.current.push(val / 100);
      setLastPct(pct);
      const passed = pct != null && pct / 100 >= script.threshold;
      if (passed) {
        setCombo((c) => c + 1);
        playSfx('correct');
        setStatus(pct == null ? '' : `Hay lắm — ${pct}%! 🎉`);
      } else {
        setCombo(0);
        setStatus(pct == null ? 'Chưa nghe rõ — xem câu mẫu bên dưới nhé.' : `Được ${pct}% — tham khảo câu mẫu.`);
      }
      setStage('revealed'); // lộ expected_user làm feedback
    } catch (e: any) {
      // Lỗi mạng: không chặn phiên — cho tiếp, tính 0 điểm lượt này.
      scoresRef.current.push(0);
      setLastPct(null);
      setCombo(0);
      setStatus(`Lỗi chấm: ${e.message || e}`);
      setStage('revealed');
    }
  }

  function next() {
    if (turnIdx + 1 < turns.length) {
      setTurnIdx((i) => i + 1);
      setStage('answering');
      setStatus('');
      setLastPct(null);
      return;
    }
    // Hết lượt → trung bình điểm các lượt (0-1) → parent ghi kết quả + thưởng.
    const scores = scoresRef.current;
    const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    onCompleted(avg);
  }

  if (!turn) {
    return <div className="course-game">Chưa dựng được kịch bản nhập vai.</div>;
  }

  const recording = stage === 'recording';
  const revealed = stage === 'revealed';
  const isLast = turnIdx + 1 >= turns.length;

  return (
    <div className="course-game flex flex-col gap-4">
      <div className="course-section-label">
        🎭 {script.scenario}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          Bạn: <b>{script.role_user}</b> · Đối thoại: <b>{script.role_npc}</b>
        </span>
        <span className="tabular-nums">
          Lượt {turnIdx + 1}/{turns.length}
          {combo >= 2 && <span className="ml-2 font-bold text-amber-500">🔥 Combo x{combo}</span>}
        </span>
      </div>

      {/* Lời NPC */}
      <div className="flex items-start gap-2">
        <div className="max-w-[85%] rounded-2xl rounded-tl-sm bg-muted px-4 py-2 text-sm">
          {turn.npc}
        </div>
        <button
          type="button"
          onClick={() => playWordTts(turn.npc)}
          className="mt-1 shrink-0 rounded-full p-1.5 text-primary hover:bg-accent"
          title="Nghe lại lời thoại"
        >
          <Volume2 className="h-4 w-4" />
        </button>
      </div>

      {/* Gợi ý (khi đang trả lời) hoặc câu mẫu (sau khi chấm) */}
      {!revealed ? (
        turn.hint && (
          <div className="self-end rounded-lg border border-dashed border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:bg-amber-950/30 dark:text-amber-200">
            💡 {turn.hint}
          </div>
        )
      ) : (
        <div className="self-end max-w-[85%] rounded-2xl rounded-tr-sm bg-primary/10 px-4 py-2 text-right text-sm">
          <div className="mb-0.5 text-[0.65rem] font-semibold uppercase text-muted-foreground">Câu mẫu</div>
          {turn.expected_user}
          <button
            type="button"
            onClick={() => playWordTts(turn.expected_user)}
            className="ml-1 align-middle text-primary hover:text-primary/80"
            title="Nghe câu mẫu"
          >
            <Volume2 className="inline h-4 w-4" />
          </button>
        </div>
      )}

      {/* Mic + trạng thái */}
      <div className="course-complete flex flex-col items-center gap-2">
        {!revealed && (
          <div className="course-complete__hint text-center">
            Bấm 🎙️ và trả lời NPC bằng lời nói. Đạt {targetPct}% âm đúng để qua lượt.
          </div>
        )}
        {!revealed ? (
          <button
            type="button"
            className={'practice-mic' + (recording ? ' recording' : '')}
            onClick={toggle}
            disabled={stage === 'grading'}
            title="Ghi âm câu trả lời"
          >
            🎙️
          </button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={next}>
            {isLast ? '🏁 Hoàn thành hội thoại' : 'Lượt tiếp →'}
          </button>
        )}
        {status && <div className="course-complete__status text-center">{status}</div>}
        {revealed && lastPct == null && (
          <button
            type="button"
            className="text-xs text-primary underline"
            onClick={() => {
              // Thử lại lượt này: bỏ điểm vừa ghi, quay lại trả lời.
              scoresRef.current.pop();
              setStage('answering');
              setStatus('');
              setLastPct(null);
            }}
          >
            Thử lại lượt này
          </button>
        )}
      </div>
    </div>
  );
}
