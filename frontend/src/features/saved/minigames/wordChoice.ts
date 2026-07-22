// Helper thuần cho mini-game 4-lựa-chọn (nghe & chọn / nghĩa → nhớ từ). Dựng bộ
// lựa chọn = từ đúng + tối đa (n-1) distractor lấy từ chính các từ đã lưu của
// người dùng (quen mắt hơn từ ngẫu nhiên), đã xáo trộn.

const norm = (w: string) => (w || '').trim().toLowerCase();

/** Xáo trộn bản sao (Fisher–Yates) — mini-game nên vị trí đáp án đúng ngẫu nhiên. */
function shuffled<T>(arr: T[]): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Bộ lựa chọn cho 1 lượt: `target` + tối đa `n-1` distractor lấy từ `pool` (khác
 * target, khử trùng lặp không phân biệt hoa/thường), tất cả đã xáo trộn. Ít từ
 * hơn `n` → trả ít lựa chọn hơn (tối thiểu chỉ có target).
 */
export function buildChoices(target: string, pool: string[], n = 4): string[] {
  const t = norm(target);
  const seen = new Set<string>([t]);
  const distractors: string[] = [];
  for (const w of shuffled(pool)) {
    const k = norm(w);
    if (!k || seen.has(k)) continue;
    seen.add(k);
    distractors.push(w);
    if (distractors.length >= n - 1) break;
  }
  return shuffled([target, ...distractors]);
}
