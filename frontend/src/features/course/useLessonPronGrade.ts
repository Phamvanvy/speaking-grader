// Hook chấm phát âm dùng CHUNG cho lesson pronunciation: cả màn cũ (PronBody,
// fallback) lẫn bước Boss của vòng chơi đều gọi ĐÚNG hook này → chỉ tồn tại MỘT
// đường chấm (cùng /grade + cùng tham số + cùng practicePct). Đảm bảo "chấm trong
// lesson cũ == chấm trong Boss" bit-for-bit; UI Boss chỉ bọc thêm hiệu ứng.
//
// Trách nhiệm: thu âm (MediaRecorder), POST /grade, tính %, phát SFX "sai" khi
// dưới ngưỡng, set status. KHÔNG điều hướng/đánh dấu hoàn thành — caller nhận pct
// (0-100) qua onGraded rồi tự gọi onCompleted(pct/100).

import { useRef, useState } from 'react';
import { apiFetch } from '@/lib/api';
import { useUiStore } from '@/store/ui';
import { playSfx } from '@/lib/sfx';
import type { LessonContent, PronWord } from './courseApi';

// % chính xác: (ok + low-severity) / non-skipped — khớp practicePct của PracticeDialog.
export function practicePct(phonemes: any[]): number | null {
  const scored = (phonemes || []).filter((p) => p.status !== 'skipped');
  if (!scored.length) return null;
  const pass = scored.filter((p) => p.status === 'ok' || p.severity === 'low').length;
  return Math.round((100 * pass) / scored.length);
}

export interface LessonPronGrade {
  recording: boolean;
  grading: boolean;
  status: string;
  drill: string;
  toggle: () => Promise<void>;
}

export function useLessonPronGrade(
  lesson: LessonContent,
  words: PronWord[],
  onGraded: (pct: number) => void,
): LessonPronGrade {
  const accent = useUiStore((s) => s.accent);
  const [recording, setRecording] = useState(false);
  const [grading, setGrading] = useState(false);
  const [status, setStatus] = useState('');
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const drill = words.map((w) => w.word).join(' ');

  function stopStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setRecording(false);
  }

  async function toggle() {
    if (grading) return;
    if (recording) {
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
    setRecording(true);
    setStatus('Đang ghi âm… bấm lần nữa để dừng.');
  }

  async function grade(blob: Blob, mime: string) {
    if (!blob || blob.size < 1024) {
      setStatus('Ghi âm quá ngắn — bấm 🎙️, đọc rõ các từ, rồi bấm dừng.');
      return;
    }
    setGrading(true);
    setStatus('Đang chấm…');
    try {
      const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm';
      const fd = new FormData();
      fd.append('audio', new File([blob], `lesson-${lesson.id}.${ext}`, { type: mime }));
      fd.append('text', drill);
      fd.append('mode', 'mock_test');
      fd.append('no_ai', 'true');
      fd.append('strict', 'true');
      fd.append('accent', accent);
      if (lesson.exam === 'topik') fd.append('exam', 'topik'); // pipeline tiếng Hàn
      const res = await apiFetch('/grade', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      const ws = result?.phoneme?.score?.words || [];
      const merged = ws.flatMap((w: any) => w.phonemes || []);
      const pct = practicePct(merged);
      if (pct == null) {
        setStatus('Chưa nghe rõ — hãy đọc to, rõ từng từ rồi thử lại.');
        return;
      }
      setStatus(pct >= 80 ? `Tuyệt vời — ${pct}%! 🎉` : `Được ${pct}% — luyện thêm rồi thử lại nhé.`);
      // 1 SFX/đợt: chưa đạt ngưỡng → tiếng "sai"; đạt → confetti+SFX ở onCompleted.
      if (pct / 100 < lesson.done_threshold) playSfx('wrong');
      onGraded(pct);
    } catch (e: any) {
      setStatus(`Lỗi chấm: ${e.message || e}`);
    } finally {
      setGrading(false);
    }
  }

  return { recording, grading, status, drill, toggle };
}
