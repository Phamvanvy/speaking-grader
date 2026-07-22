// Nạp thông tin từ cho vòng chơi khóa học: nghĩa tiếng Việt (Word Match) + câu ví
// dụ (Sentence Builder). Cả hai lấy từ MỘT endpoint /word-info sẵn có (đã cache
// SQLite phía server: definition_en/example_en/meaning). Không thêm backend.
// Cache trong phiên để không gọi lại khi ôn/luyện lại.

import { apiFetch } from '@/lib/api';

export interface WordInfo {
  meaning: string | null; // nghĩa tiếng Việt (null nếu không có)
  example: string | null; // câu ví dụ tiếng Anh (null nếu không có)
}

const EMPTY: WordInfo = { meaning: null, example: null };
const cache = new Map<string, WordInfo>();

const key = (w: string) => (w || '').trim().toLowerCase();

/** Nạp (hoặc lấy cache) thông tin 1 từ. Lỗi mạng → trả rỗng (game tự bỏ bước thiếu dữ liệu). */
export async function fetchWordInfo(word: string): Promise<WordInfo> {
  const k = key(word);
  const hit = cache.get(k);
  if (hit) return hit;
  let wi: WordInfo = EMPTY;
  try {
    const r = await apiFetch(`/word-info?word=${encodeURIComponent(k)}`);
    if (r.ok) {
      const info = await r.json();
      wi = {
        meaning: (info?.meaning || '').trim() || null,
        example: (info?.example_en || '').trim() || null,
      };
    }
  } catch {
    /* im lặng — thiếu dữ liệu chỉ khiến bỏ bước tương ứng, không chặn vòng chơi */
  }
  cache.set(k, wi);
  return wi;
}

/** Nạp song song thông tin cho nhiều từ; trả map keyed lowercase. */
export async function fetchWordInfos(words: string[]): Promise<Map<string, WordInfo>> {
  const out = new Map<string, WordInfo>();
  await Promise.all(
    words.map(async (w) => {
      out.set(key(w), await fetchWordInfo(w));
    }),
  );
  return out;
}
