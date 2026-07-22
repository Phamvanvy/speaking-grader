// Chọn từ cho phiên "Luyện nhanh" — spaced repetition đơn giản, THUẦN client từ
// dữ liệu đã có trong store (last_score, last_practiced_at, saved_at) → không cần
// đổi schema/endpoint. Ưu tiên: điểm kém trước, rồi lâu chưa luyện; từ CHƯA từng
// luyện đẩy lên đầu (giá trị học cao nhất). Hàm thuần, tất định (dễ suy luận/test).

import type { SavedWord } from '@/store/savedWords';

// Điểm kém là tín hiệu mạnh nhất; "lâu chưa ôn" là phụ (cap để 1 từ cũ lâu không
// đè hết các từ điểm thấp).
const W_SCORE = 1.0;
const W_AGE = 0.25;
const AGE_CAP_DAYS = 30;

function ageDays(iso: string | null | undefined): number {
  if (!iso) return AGE_CAP_DAYS; // không rõ thời điểm → coi như đã lâu
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return AGE_CAP_DAYS;
  return Math.max(0, (Date.now() - t) / 86_400_000);
}

/** Điểm ưu tiên 0..~1.25 (cao = nên luyện trước). */
export function reviewPriority(w: SavedWord): number {
  const scorePart = 1 - (w.last_score ?? 0); // chưa luyện (null) → 1 (kém nhất)
  const refDate = w.last_practiced_at ?? w.saved_at;
  const agePart = Math.min(ageDays(refDate), AGE_CAP_DAYS) / AGE_CAP_DAYS;
  return scorePart * W_SCORE + agePart * W_AGE;
}

/**
 * Dựng hàng đợi phiên luyện: top-`size` từ theo ưu tiên giảm dần. Tie-break tất
 * định: chưa-luyện trước → luyện lâu hơn trước → A→Z. `size<=0` hoặc lớn hơn số
 * từ → trả toàn bộ (đã sắp).
 */
export function buildReviewQueue(words: SavedWord[], size: number): SavedWord[] {
  const withKey = words.map((w) => ({ w, p: reviewPriority(w) }));
  withKey.sort((a, b) => {
    if (b.p !== a.p) return b.p - a.p;
    // Tie-break: chưa từng luyện (null last_score) lên trước.
    const na = a.w.last_score == null ? 1 : 0;
    const nb = b.w.last_score == null ? 1 : 0;
    if (na !== nb) return nb - na;
    // Rồi tới luyện/ lưu lâu hơn (thời điểm sớm hơn) lên trước.
    const ta = Date.parse(a.w.last_practiced_at ?? a.w.saved_at ?? '') || 0;
    const tb = Date.parse(b.w.last_practiced_at ?? b.w.saved_at ?? '') || 0;
    if (ta !== tb) return ta - tb;
    return (a.w.word || '').localeCompare(b.w.word || '');
  });
  const sorted = withKey.map((x) => x.w);
  return size > 0 && size < sorted.length ? sorted.slice(0, size) : sorted;
}
