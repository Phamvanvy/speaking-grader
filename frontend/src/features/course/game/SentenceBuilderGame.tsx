// 🧩 Sentence Builder — xếp lại câu ví dụ từ các từ xáo trộn. Presentational THUẦN:
// nhận `sentence` (câu ví dụ tiếng Anh), người học bấm từ ở kho → xếp vào hàng trả
// lời; đủ từ thì tự chấm (đúng = đúng thứ tự gốc) rồi gọi `onResult(correct)` MỘT
// LẦN. Logic chơi hoàn toàn offline/tất định (no-AI). Điều phối do session lo.

import { useMemo, useState } from 'react';
import { Check, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Token {
  id: number;
  text: string;
}

function shuffle<T>(arr: T[]): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export default function SentenceBuilderGame({
  sentence,
  onResult,
}: {
  sentence: string;
  onResult: (correct: boolean) => void;
}) {
  const tokens = useMemo<Token[]>(
    () => sentence.trim().split(/\s+/).filter(Boolean).map((text, id) => ({ id, text })),
    [sentence],
  );
  const bankOrder = useMemo(() => shuffle(tokens), [tokens]);

  const [answer, setAnswer] = useState<Token[]>([]);
  const [verdict, setVerdict] = useState<'idle' | 'right' | 'wrong'>('idle');

  const answered = verdict !== 'idle';
  const placed = new Set(answer.map((t) => t.id));

  function place(tok: Token) {
    if (answered || placed.has(tok.id)) return;
    const next = [...answer, tok];
    setAnswer(next);
    if (next.length === tokens.length) {
      const correct = next.map((t) => t.text).join(' ') === tokens.map((t) => t.text).join(' ');
      setVerdict(correct ? 'right' : 'wrong');
      onResult(correct);
    }
  }

  function unplace(tok: Token) {
    if (answered) return;
    setAnswer((a) => a.filter((t) => t.id !== tok.id));
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="text-center text-sm font-medium text-muted-foreground">
        🧩 Xếp các từ thành câu đúng
      </div>

      {/* Hàng trả lời */}
      <div
        className={cn(
          'flex min-h-[3.25rem] flex-wrap content-start items-start gap-2 rounded-xl border border-dashed p-2',
          verdict === 'right' && 'border-green-500 bg-green-50 dark:bg-green-950/30',
          verdict === 'wrong' && 'border-red-500 bg-red-50 dark:bg-red-950/30',
        )}
      >
        {answer.length === 0 && (
          <span className="px-1 py-1.5 text-sm text-muted-foreground">Bấm các từ bên dưới…</span>
        )}
        {answer.map((t) => (
          <button
            key={t.id}
            type="button"
            disabled={answered}
            onClick={() => unplace(t)}
            className="rounded-lg border bg-background px-3 py-1.5 text-base font-medium shadow-sm"
          >
            {t.text}
          </button>
        ))}
        {verdict === 'right' && <Check className="ml-auto h-5 w-5 self-center text-green-600" />}
        {verdict === 'wrong' && <X className="ml-auto h-5 w-5 self-center text-red-600" />}
      </div>

      {/* Câu đúng khi xếp sai */}
      {verdict === 'wrong' && (
        <div className="text-center text-sm text-muted-foreground">
          Câu đúng: <b className="text-foreground">{tokens.map((t) => t.text).join(' ')}</b>
        </div>
      )}

      {/* Kho từ */}
      <div className="flex flex-wrap justify-center gap-2">
        {bankOrder.map((t) => {
          const used = placed.has(t.id);
          return (
            <button
              key={t.id}
              type="button"
              disabled={used || answered}
              onClick={() => place(t)}
              className={cn(
                'rounded-lg border px-3 py-2 text-base font-medium transition-colors',
                used ? 'invisible' : 'hover:border-primary hover:bg-primary/5',
              )}
            >
              {t.text}
            </button>
          );
        })}
      </div>
    </div>
  );
}
