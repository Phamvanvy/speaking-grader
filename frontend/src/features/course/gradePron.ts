// MỘT đường chấm phát âm DUY NHẤT cho khóa học. Boss + fallback PronBody (đọc list
// từ) và Shadowing (đọc theo câu mẫu) đều gọi hàm này → không tồn tại nhánh chấm thứ
// hai, hoàn thành lesson (qua Boss) không bao giờ lệch. Chỉ đổi `text` (drill) theo
// bài; TẤT CẢ tham số /grade (mode/no_ai/strict/accent + exam topik) giữ nguyên.

import { apiFetch } from '@/lib/api';
import type { LessonContent } from './courseApi';

// % chính xác: (ok + low-severity) / non-skipped — khớp practicePct của PracticeDialog.
export function practicePct(phonemes: any[]): number | null {
  const scored = (phonemes || []).filter((p) => p.status !== 'skipped');
  if (!scored.length) return null;
  const pass = scored.filter((p) => p.status === 'ok' || p.severity === 'low').length;
  return Math.round((100 * pass) / scored.length);
}

export function pickExt(mime: string): string {
  return mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm';
}

/**
 * POST /grade với đúng bộ tham số phát âm rồi rút % qua practicePct. Ném lỗi khi HTTP
 * != ok (caller bắt & hiển thị). `text` = chuỗi cần đọc (list từ cho Boss, câu cho
 * Shadowing). KHÔNG đổi tham số nào khác — đây là hợp đồng "một đường chấm".
 */
export async function gradePronunciation(
  lesson: LessonContent,
  text: string,
  blob: Blob,
  mime: string,
  accent: string,
): Promise<number | null> {
  const fd = new FormData();
  fd.append('audio', new File([blob], `lesson-${lesson.id}.${pickExt(mime)}`, { type: mime }));
  fd.append('text', text);
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
  return practicePct(merged);
}
