// ⚡ Word Match — nối từ (tiếng Anh) ↔ nghĩa (tiếng Việt) theo lô. Presentational
// THUẦN: nhận `pairs` + `onDone(results)` với đúng/sai TỪNG từ (đúng = nối trúng
// ngay, không bấm sai lần nào cho từ đó). Không biết gì về session/XP — điều phối
// (award, chuyển bước) do CourseGameSession lo, giống khuôn WordChoiceGame.

import { useMemo, useState } from 'react';
import { Check, X } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface MatchPair {
  word: string;
  meaning: string;
}

export interface MatchResult {
  word: string;
  correct: boolean;
}

function shuffle<T>(arr: T[]): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export default function WordMatchGame({
  pairs,
  onDone,
}: {
  pairs: MatchPair[];
  onDone: (results: MatchResult[]) => void;
}) {
  const meanings = useMemo(() => shuffle(pairs.map((p) => p.meaning)), [pairs]);
  const meaningOf = useMemo(() => new Map(pairs.map((p) => [p.word, p.meaning])), [pairs]);

  const [selected, setSelected] = useState<string | null>(null); // từ đang chọn
  const [matched, setMatched] = useState<Record<string, true>>({});
  const [errored, setErrored] = useState<Record<string, true>>({});
  const [wrong, setWrong] = useState<string | null>(null); // meaning vừa bấm sai (nháy đỏ)
  const [done, setDone] = useState(false);

  const matchedMeanings = new Set(Object.keys(matched).map((w) => meaningOf.get(w)));

  function finishIfComplete(nextMatched: Record<string, true>, nextErrored: Record<string, true>) {
    if (Object.keys(nextMatched).length !== pairs.length || done) return;
    setDone(true);
    onDone(pairs.map((p) => ({ word: p.word, correct: !nextErrored[p.word] })));
  }

  function pickWord(word: string) {
    if (matched[word] || done) return;
    setSelected(word);
  }

  function pickMeaning(meaning: string) {
    if (done || matchedMeanings.has(meaning)) return;
    if (!selected) return;
    if (meaningOf.get(selected) === meaning) {
      const nm = { ...matched, [selected]: true as const };
      setMatched(nm);
      setSelected(null);
      finishIfComplete(nm, errored);
    } else {
      const ne = { ...errored, [selected]: true as const };
      setErrored(ne);
      setWrong(meaning);
      setSelected(null);
      window.setTimeout(() => setWrong(null), 450);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="text-center text-sm font-medium text-muted-foreground">
        ⚡ Nối từ với nghĩa đúng
      </div>
      <div className="grid grid-cols-2 gap-3">
        {/* Cột từ tiếng Anh */}
        <div className="flex flex-col gap-2">
          {pairs.map((p) => {
            const isMatched = !!matched[p.word];
            const isSel = selected === p.word;
            return (
              <button
                key={p.word}
                type="button"
                disabled={isMatched || done}
                onClick={() => pickWord(p.word)}
                className={cn(
                  'flex items-center justify-center gap-1.5 rounded-xl border px-3 py-3 text-base font-semibold transition-colors',
                  isMatched && 'border-green-500 bg-green-50 text-green-700 dark:bg-green-950/40 dark:text-green-300',
                  !isMatched && isSel && 'border-primary bg-primary/10',
                  !isMatched && !isSel && 'hover:border-primary hover:bg-primary/5',
                )}
              >
                {isMatched && <Check className="h-4 w-4 shrink-0" />}
                <span className="truncate">{p.word}</span>
              </button>
            );
          })}
        </div>
        {/* Cột nghĩa tiếng Việt (đã xáo) */}
        <div className="flex flex-col gap-2">
          {meanings.map((m) => {
            const isMatched = matchedMeanings.has(m);
            const isWrong = wrong === m;
            return (
              <button
                key={m}
                type="button"
                disabled={isMatched || done}
                onClick={() => pickMeaning(m)}
                className={cn(
                  'flex items-center justify-center gap-1.5 rounded-xl border px-3 py-3 text-sm font-medium transition-colors',
                  isMatched && 'border-green-500 bg-green-50 text-green-700 dark:bg-green-950/40 dark:text-green-300',
                  isWrong && 'border-red-500 bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300',
                  !isMatched && !isWrong && 'hover:border-primary hover:bg-primary/5',
                )}
              >
                {isWrong && <X className="h-4 w-4 shrink-0" />}
                <span className="truncate">{m}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
