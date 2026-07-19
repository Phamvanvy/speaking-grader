// Từ đã lưu — client của API /words (server-side theo user_id, cùng cơ chế tab Lịch sử).
//
// GHI CHÚ kiến trúc (lệch có chủ đích so với "server state = TanStack Query" của plan):
// saved-words cần truy vấn `has(word)` ĐỒNG BỘ từ nhiều nơi KHÔNG phải React — renderer
// legacy (legacy/render.ts dựng sao ☆ bằng chuỗi HTML), popup luyện, review-toast. Một
// Zustand store giữ cache Map + Set khoá là nguồn sync tự nhiên cho các call site đó, đồng
// thời reactive cho SavedTab. Suggestions (LLM, chỉ đọc, không cần sync-has) vẫn qua
// TanStack Query. Xem [[word-suggestions-feature]] và [[saved-words-add-star]].

import { create } from 'zustand';
import { apiFetch } from '../lib/api';
import { getUserId } from '../lib/identity';
import { setSavedWords } from '../legacy/render';

export interface SavedWord {
  word: string;
  ipa?: string | null;
  phonemes?: any[];
  accuracy?: number | null;
  last_score?: number | null;
  saved_at?: string | null;
  last_practiced_at?: string | null;
}

const key = (w: string) => (w || '').trim().toLowerCase();

interface SavedWordsState {
  words: SavedWord[];
  keys: Set<string>; // khoá lowercase — has() sync
  loaded: boolean;
  has: (word: string) => boolean;
  get: (word: string) => SavedWord | null;
  refresh: () => Promise<SavedWord[]>;
  add: (entry: SavedWord) => Promise<SavedWord>;
  remove: (word: string) => Promise<void>;
}

function commit(words: SavedWord[]) {
  return { words, keys: new Set(words.map((w) => key(w.word))), loaded: true };
}

export const useSavedWords = create<SavedWordsState>((set, get) => ({
  words: [],
  keys: new Set<string>(),
  loaded: false,

  has: (word) => get().keys.has(key(word)),
  get: (word) => get().words.find((w) => key(w.word) === key(word)) || null,

  async refresh() {
    const data = await apiFetch(`/words?user_id=${encodeURIComponent(getUserId())}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      });
    const words: SavedWord[] = data.words || [];
    set(commit(words));
    return words;
  },

  // Upsert: server COALESCE (field không gửi giữ nguyên). Cập nhật cache tại chỗ để
  // has() phản ánh ngay (không chờ refresh) — mirror legacy SavedWords.add.
  async add(entry) {
    const fd = new FormData();
    fd.append('user_id', getUserId());
    fd.append('word', key(entry.word));
    if (entry.ipa) fd.append('ipa', entry.ipa);
    if (entry.phonemes) fd.append('phonemes', JSON.stringify(entry.phonemes));
    if (entry.accuracy != null) fd.append('accuracy', String(entry.accuracy));
    if (entry.last_score != null) fd.append('last_score', String(entry.last_score));
    const res = await apiFetch('/words', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const saved: SavedWord = await res.json();
    const others = get().words.filter((w) => key(w.word) !== key(saved.word));
    set(commit([...others, saved]));
    return saved;
  },

  async remove(word) {
    const res = await apiFetch(
      `/words/${encodeURIComponent(key(word))}?user_id=${encodeURIComponent(getUserId())}`,
      { method: 'DELETE' },
    );
    if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
    set(commit(get().words.filter((w) => key(w.word) !== key(word))));
  },
}));

// Bơm adapter {has} vào renderer legacy (sao ☆ trên bảng lỗi hỏi đồng bộ). Store là
// singleton nên has() luôn đọc state mới nhất.
setSavedWords({ has: (w: string) => useSavedWords.getState().has(w) });

/** Đồng bộ imperative các nút ☆/★ đã render dạng HTML tĩnh (bảng lỗi grading/exam/history
 *  do legacy/render.ts inject) — chúng KHÔNG re-render theo store. Mirror practice.js. */
export function syncBookmarkButtons(word: string) {
  const saved = useSavedWords.getState().has(word);
  document.querySelectorAll<HTMLElement>(`.word-bookmark[data-word="${(window as any).CSS.escape(word)}"]`).forEach((b) => {
    b.textContent = saved ? '★' : '☆';
    b.classList.toggle('saved', saved);
    b.title = saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập';
  });
}
