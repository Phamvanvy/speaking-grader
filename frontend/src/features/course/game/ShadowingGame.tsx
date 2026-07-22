// 🎙️ Shadowing Challenge — nghe câu mẫu (TTS) rồi ĐỌC LẠI cả câu; chấm qua ĐÚNG hàm
// gradePronunciation dùng chung (không nhánh chấm thứ hai). Đây là game GIỮA vòng chơi:
// award word_recall như các mini-game khác, KHÔNG quyết định hoàn thành lesson (Boss
// mới quyết định). Component tự lo ghi âm (MediaRecorder) + gọi hàm chấm chung; session
// nhận onResult(correct) để award/chuyển bước.

import { useEffect, useRef, useState } from 'react';
import { Volume2 } from 'lucide-react';
import { playWordTts } from '@/features/grading/playback';
import { gradePronunciation } from '../gradePron';
import type { LessonContent } from '../courseApi';

type Stage = 'idle' | 'recording' | 'grading' | 'done';

export default function ShadowingGame({
  lesson,
  sentence,
  accent,
  threshold,
  onResult,
}: {
  lesson: LessonContent;
  sentence: string;
  accent: string;
  threshold: number; // pass = pct/100 >= threshold (dùng done_threshold của lesson)
  onResult: (correct: boolean) => void;
}) {
  const [stage, setStage] = useState<Stage>('idle');
  const [status, setStatus] = useState('');
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Tự phát câu mẫu khi vào bài (autoplay bị chặn thì im lặng — vẫn còn nút 🔊).
  useEffect(() => {
    const t = window.setTimeout(() => playWordTts(sentence), 300);
    return () => window.clearTimeout(t);
  }, [sentence]);

  function stopStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function toggle() {
    if (stage === 'grading' || stage === 'done') return;
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
      setStatus('Ghi âm quá ngắn — bấm 🎙️, đọc cả câu rồi bấm dừng.');
      return;
    }
    setStage('grading');
    setStatus('Đang chấm…');
    try {
      const pct = await gradePronunciation(lesson, sentence, blob, mime, accent);
      if (pct == null) {
        setStage('done');
        setStatus('Chưa nghe rõ — nhưng cứ tiếp tục nhé.');
        onResult(false);
        return;
      }
      const correct = pct / 100 >= threshold;
      setStage('done');
      setStatus(correct ? `Tuyệt — ${pct}%! 🎉` : `Được ${pct}% — cứ tiếp tục.`);
      onResult(correct);
    } catch (e: any) {
      setStage('done');
      setStatus(`Lỗi chấm: ${e.message || e}`);
      onResult(false); // lỗi mạng không chặn vòng chơi
    }
  }

  const recording = stage === 'recording';
  return (
    <div className="flex flex-col items-center gap-4">
      <div className="text-center text-sm font-medium text-muted-foreground">
        🎙️ Nghe câu mẫu rồi đọc lại thật giống
      </div>
      <button
        type="button"
        onClick={() => playWordTts(sentence)}
        className="flex items-center gap-2 rounded-full bg-primary px-4 py-2 text-primary-foreground shadow-sm transition-transform hover:scale-105 active:scale-95"
        title="Nghe lại câu"
      >
        <Volume2 className="h-5 w-5" /> Nghe câu
      </button>
      <p className="max-w-md text-center text-base font-medium">{sentence}</p>
      <button
        type="button"
        className={'practice-mic' + (recording ? ' recording' : '')}
        onClick={toggle}
        disabled={stage === 'grading' || stage === 'done'}
        title="Ghi âm đọc lại câu"
      >
        🎙️
      </button>
      {status && <div className="text-center text-sm text-muted-foreground">{status}</div>}
    </div>
  );
}
