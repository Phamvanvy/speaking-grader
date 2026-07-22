// Hàm THUẦN sinh trình tự bước cho vòng chơi khóa học. Viết RIÊNG cho khóa học —
// KHÔNG tái dùng assignKinds của quickReview (vốn gắn với SavedWord + mastery-star).
// Nhận danh sách từ lesson + thông tin từ (nghĩa/câu) → trả mảng bước; tự bỏ bước
// khi thiếu dữ liệu để vòng chơi không bao giờ kẹt.

import type { PronWord } from '../courseApi';
import type { WordInfo } from './wordInfo';

export type GameKind = 'match' | 'listen' | 'build';

// match: nối một LÔ từ↔nghĩa cùng lúc. listen/build: một từ mỗi bước.
export type GameStep =
  | { kind: 'match'; words: PronWord[] }
  | { kind: 'listen'; word: PronWord }
  | { kind: 'build'; word: PronWord; example: string };

const MATCH_BATCH = 5; // số cặp tối đa mỗi màn Word Match
const MIN_EXAMPLE_WORDS = 3; // câu quá ngắn (<3 từ) không đáng để xếp lại

const wordCount = (s: string) => s.trim().split(/\s+/).filter(Boolean).length;

/**
 * Trình tự: (1) một/nhiều màn Word Match cho các từ có nghĩa (chia lô ≤5), rồi
 * (2) xen kẽ Listen & Pick / Sentence Builder trên TỪNG từ theo thứ tự lesson —
 * vị trí lẻ ưu tiên Sentence Builder nếu có câu ví dụ đủ dài, còn lại Listen & Pick.
 * Thiếu nghĩa → bỏ khỏi Word Match; thiếu câu → không có bước build cho từ đó.
 */
export function buildCourseGameSequence(
  words: PronWord[],
  infoOf: (word: string) => WordInfo,
): GameStep[] {
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
    if (i % 2 === 1 && canBuild) {
      steps.push({ kind: 'build', word: w, example: example as string });
    } else {
      steps.push({ kind: 'listen', word: w });
    }
  });

  return steps;
}
