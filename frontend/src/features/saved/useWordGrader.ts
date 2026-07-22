// Hook thu âm 1 từ → gọi /grade → parse phoneme → tính %. Tách từ PracticeDialog
// để phiên "Luyện nhanh" (QuickReviewDialog) tái dùng CÙNG đường chấm, không viết
// lại. Giữ nguyên các hằng đã tinh chỉnh: mode=mock_test, no_ai, strict, accent,
// exam=topik cho Hangul, guard blob <1KB, xử lý "tất cả skipped".
//
// Ranh giới trách nhiệm: hook sở hữu cơ chế ghi âm + POST + status GENERIC (đang
// ghi/đang chấm/lỗi mic/quá ngắn/không nghe rõ). Thông điệp THÀNH CÔNG + ăn mừng +
// cộng XP + cập nhật saved_word do CALLER quyết (qua onGraded) — mỗi nơi khác nhau.

import { useEffect, useRef, useState } from 'react';
import { apiFetch } from '@/lib/api';
import { hasHangul } from '@/lib/config';

export type GraderStatus = { text: string; kind: '' | 'err' | 'good' };

export interface GradedResult {
  word: string;
  /** Phoneme đã gộp mọi từ trong cụm (gắn _w = chỉ số từ) — như PracticeDialog. */
  phonemes: any[];
  /** % chính xác (0–100). */
  pct: number;
  transcript: string;
}

interface Options {
  accent: string;
  /** Gọi khi có kết quả phoneme hợp lệ. Caller tự set status thành công + ăn mừng. */
  onGraded: (r: GradedResult) => void;
}

// % chính xác: (ok + low-severity) / non-skipped — khớp ngưỡng isSignificant render.js.
export function practicePct(phonemes: any[]): number | null {
  const scored = (phonemes || []).filter((p) => p.status !== 'skipped');
  if (!scored.length) return null;
  const pass = scored.filter((p) => p.status === 'ok' || p.severity === 'low').length;
  return Math.round((100 * pass) / scored.length);
}

export function useWordGrader({ accent, onGraded }: Options) {
  const [status, setStatus] = useState<GraderStatus>({ text: '', kind: '' });
  const [recording, setRecording] = useState(false);
  const [grading, setGrading] = useState(false);
  const [replayUrl, setReplayUrl] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const replayAudioRef = useRef<HTMLAudioElement | null>(null);
  const wordRef = useRef<string>('');
  const recordingRef = useRef(false); // đọc sync trong stopStream (state async)

  function stopStream() {
    if (recorderRef.current && recordingRef.current) {
      try {
        recorderRef.current.stop();
      } catch {
        /* đã stop */
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    recordingRef.current = false;
    setRecording(false);
  }

  /** Dọn stream + revoke URL replay — gọi khi đóng dialog/hủy phiên. */
  function cleanup() {
    stopStream();
    setReplayUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
  }

  // Dọn khi unmount.
  useEffect(() => cleanup, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function toggleRecording(word: string) {
    if (grading) return;
    if (recordingRef.current) {
      recorderRef.current?.stop();
      return;
    }
    wordRef.current = word;
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setStatus({ text: 'Không truy cập được micro — kiểm tra quyền trình duyệt.', kind: 'err' });
      return;
    }
    chunksRef.current = [];
    const rec = new MediaRecorder(streamRef.current);
    recorderRef.current = rec;
    rec.addEventListener('dataavailable', (e) => {
      if (e.data && e.data.size) chunksRef.current.push(e.data);
    });
    rec.addEventListener('stop', () => {
      const mime = rec.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      stopStream();
      setReplayUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(blob);
      });
      gradeBlob(blob, mime);
    });
    rec.start();
    recordingRef.current = true;
    setRecording(true);
    setStatus({ text: 'Đang ghi âm… bấm lần nữa để dừng.', kind: '' });
  }

  async function gradeBlob(blob: Blob, mime: string) {
    const word = wordRef.current;
    if (!word) return;
    // Blob cụt (<1KB) = MediaRecorder nhả header rỗng → chặn khỏi tốn request.
    if (!blob || blob.size < 1024) {
      setStatus({ text: 'Ghi âm quá ngắn — bấm 🎙️, nói rõ từ, rồi mới bấm dừng.', kind: 'err' });
      return;
    }
    setGrading(true);
    setStatus({ text: 'Đang chấm…', kind: '' });
    try {
      const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm';
      const fd = new FormData();
      fd.append('audio', new File([blob], `practice-${word}.${ext}`, { type: mime }));
      fd.append('text', word);
      fd.append('mode', 'mock_test'); // ép bật phoneme analysis
      fd.append('no_ai', 'true'); // 1 từ không cần LLM
      fd.append('strict', 'true'); // chấm CHẶT (tắt leniency câu dài)
      fd.append('accent', accent);
      if (hasHangul(word)) fd.append('exam', 'topik'); // pipeline ko cho từ Hàn
      const res = await apiFetch('/grade', { method: 'POST', body: fd });
      if (!res.ok) {
        const raw = await res.text();
        let detail = `HTTP ${res.status}`;
        try {
          detail = JSON.parse(raw).detail || detail;
        } catch {
          /* giữ mã HTTP */
        }
        throw new Error(detail);
      }
      const result = await res.json();
      const ws = result?.phoneme?.score?.words || [];
      if (!ws.some((w: any) => (w.phonemes || []).length)) {
        throw new Error('Server không trả kết quả phoneme — thử lại.');
      }
      const wordSkipped = (w: any) => w.skip_reason || (w.phonemes || []).every((p: any) => p.status === 'skipped');
      if (ws.every(wordSkipped)) {
        const heard = (result.transcript || '').trim();
        const heardShort = heard.length > 60 ? `${heard.slice(0, 57)}…` : heard;
        setStatus({
          text: heardShort
            ? `Chưa nghe rõ — máy nghe thành “${heardShort}”. Hãy nói to, rõ và thử lại.`
            : 'Chưa nghe rõ — không thấy tiếng nói. Hãy nói to, rõ và thử lại.',
          kind: 'err',
        });
        return;
      }
      // Cụm nhiều từ: gộp phoneme tất cả các từ (gắn _w = chỉ số từ trong cụm).
      const merged =
        ws.length === 1
          ? ws[0].phonemes || []
          : ws.flatMap((w: any, i: number) => (w.phonemes || []).map((p: any) => ({ ...p, _w: i })));
      const pct = practicePct(merged) ?? 0;
      onGraded({ word, phonemes: merged, pct, transcript: (result.transcript || '').trim() });
    } catch (err: any) {
      setStatus({ text: `Lỗi chấm: ${err.message || err}`, kind: 'err' });
    } finally {
      setGrading(false);
    }
  }

  function playReplay() {
    if (!replayUrl) return;
    if (!replayAudioRef.current || replayAudioRef.current.src !== replayUrl) {
      replayAudioRef.current = new Audio(replayUrl);
    }
    replayAudioRef.current.currentTime = 0;
    replayAudioRef.current.play().catch(() => {});
  }

  return { status, setStatus, recording, grading, replayUrl, setReplayUrl, toggleRecording, playReplay, cleanup };
}
