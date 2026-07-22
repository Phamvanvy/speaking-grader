// Mini-game "Nghĩa → nhớ từ": hiện nghĩa tiếng Việt (từ /word-info, đã cache SQLite
// phía server) rồi cho chọn 1 trong 4 từ. Không có nghĩa (từ hiếm / lỗi mạng) →
// tự lùi về "Nghe & chọn từ" để bài không bị kẹt. Mount lại mỗi bài (key=index).

import { useEffect, useState } from 'react';
import { apiFetch } from '@/lib/api';
import type { SavedWord } from '@/store/savedWords';
import { buildChoices } from './wordChoice';
import WordChoiceGame from './WordChoiceGame';
import ListenChoose from './ListenChoose';

// Cache nghĩa trong phiên (server cũng cache) — tránh gọi lại khi "Luyện lại".
const meaningCache = new Map<string, string | null>();

export default function MeaningRecall({
  word,
  pool,
  onResult,
}: {
  word: SavedWord;
  pool: string[];
  onResult: (correct: boolean) => void;
}) {
  const key = word.word.trim().toLowerCase();
  const [meaning, setMeaning] = useState<string | null | undefined>(
    meaningCache.has(key) ? meaningCache.get(key) : undefined, // undefined = đang tải
  );

  useEffect(() => {
    if (meaningCache.has(key)) return; // đã có (kể cả null)
    let alive = true;
    apiFetch(`/word-info?word=${encodeURIComponent(key)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((info: { meaning?: string }) => {
        const m = (info?.meaning || '').trim() || null;
        meaningCache.set(key, m);
        if (alive) setMeaning(m);
      })
      .catch(() => {
        meaningCache.set(key, null);
        if (alive) setMeaning(null);
      });
    return () => {
      alive = false;
    };
  }, [key]);

  // Không có nghĩa → chơi "Nghe & chọn từ" thay thế (vẫn 1 lượt word_recall).
  if (meaning === null) return <ListenChoose word={word} pool={pool} onResult={onResult} />;

  if (meaning === undefined) {
    return <div className="py-8 text-center text-sm text-muted-foreground">Đang tải nghĩa…</div>;
  }

  const options = buildChoices(word.word, pool, 4);
  return (
    <WordChoiceGame
      hint="🧠 Từ nào mang nghĩa này?"
      prompt={
        <div className="rounded-xl border bg-muted/30 px-4 py-4 text-center text-lg font-medium">
          🇻🇳 {meaning}
        </div>
      }
      options={options}
      answer={word.word}
      onResult={onResult}
    />
  );
}
