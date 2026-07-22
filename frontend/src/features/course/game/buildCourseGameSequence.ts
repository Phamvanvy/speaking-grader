// Hàm THUẦN sinh trình tự bước cho vòng chơi khóa học. Viết RIÊNG cho khóa học —
// KHÔNG tái dùng assignKinds của quickReview (vốn gắn với SavedWord + mastery-star).
// Nhận danh sách từ lesson + thông tin từ (nghĩa/câu) → trả mảng bước; tự bỏ bước
// khi thiếu dữ liệu để vòng chơi không bao giờ kẹt.

import type { PronWord } from '../courseApi';
import type { WordInfo } from './wordInfo';

export type GameKind = 'match' | 'listen' | 'build' | 'dictate' | 'shadow';

// match: nối một LÔ từ↔nghĩa cùng lúc. listen/build/dictate/shadow: một từ mỗi bước.
export type GameStep =
  | { kind: 'match'; words: PronWord[] }
  | { kind: 'listen'; word: PronWord }
  | { kind: 'build'; word: PronWord; example: string }
  | { kind: 'dictate'; word: PronWord }
  | { kind: 'shadow'; word: PronWord; example: string };

// Phase 2 toggles (mặc định bật) — truyền từ config để tắt riêng từng game khi cần.
export interface SequenceOpts {
  dictation?: boolean;
  shadowing?: boolean;
}

const MATCH_BATCH = 5; // số cặp tối đa mỗi màn Word Match
const MIN_EXAMPLE_WORDS = 3; // câu quá ngắn (<3 từ) không đáng để xếp lại
const MIN_SHADOW_WORDS = 4; // Shadowing cần câu đủ dài (≥4 từ) mới đáng đọc lại
const MAX_SHADOW = 1; // tối đa 1 bước Shadowing/phiên — giữ vòng chơi gọn, ít ghi âm

const wordCount = (s: string) => s.trim().split(/\s+/).filter(Boolean).length;

/**
 * Trình tự:
 * (1) một/nhiều màn Word Match cho các từ có nghĩa (chia lô ≤5).
 * (2) xen kẽ trên TỪNG từ theo thứ tự lesson với chu kỳ 3: vị trí ≡0 → Listen & Pick,
 *     ≡1 → Dictation (nghe→gõ, nếu bật), ≡2 → Sentence Builder (nếu câu đủ dài).
 *     Thiếu dữ liệu/tắt cờ → rơi về Listen & Pick (vòng chơi không bao giờ kẹt).
 * (3) đúng MỘT bước Shadowing (đọc lại câu mẫu, chấm qua đường chung) trước Boss, nếu
 *     bật và có câu đủ dài — đặt cuối để không rải nhiều lần ghi âm giữa chừng.
 */
export function buildCourseGameSequence(
  words: PronWord[],
  infoOf: (word: string) => WordInfo,
  opts: SequenceOpts = {},
): GameStep[] {
  const { dictation = true, shadowing = true } = opts;
  const steps: GameStep[] = [];

  const withMeaning = words.filter((w) => infoOf(w.word).meaning);
  if (withMeaning.length >= 2) {
    for (let i = 0; i < withMeaning.length; i += MATCH_BATCH) {
      steps.push({ kind: 'match', words: withMeaning.slice(i, i + MATCH_BATCH) });
    }
  }

  words.forEach((w, i) => {
    const example = infoOf(w.word).example;
    const canBuild = !!example && wordCount(example) >= MIN_EXAMPLE_WORDS;
    const cyc = i % 3;
    if (cyc === 2 && canBuild) {
      steps.push({ kind: 'build', word: w, example: example as string });
    } else if (cyc === 1 && dictation) {
      steps.push({ kind: 'dictate', word: w });
    } else {
      steps.push({ kind: 'listen', word: w });
    }
  });

  if (shadowing) {
    let added = 0;
    for (const w of words) {
      if (added >= MAX_SHADOW) break;
      const example = infoOf(w.word).example;
      if (example && wordCount(example) >= MIN_SHADOW_WORDS) {
        steps.push({ kind: 'shadow', word: w, example });
        added++;
      }
    }
  }

  return steps;
}
