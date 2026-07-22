// Khung mini-game 4-lựa-chọn dùng chung cho "Nghe & chọn từ" và "Nghĩa → nhớ từ".
// Presentational: nhận `prompt` (nút 🔊 hoặc nghĩa tiếng Việt), danh sách `options`
// và từ đúng `answer`; xử lý chọn + hiện đúng/sai rồi gọi `onResult(correct)` ĐÚNG
// MỘT LẦN. Chấm điểm/combo/XP nằm ở QuickReviewDialog (giống đường nói).

import { useState } from 'react';
import { Check, X } from 'lucide-react';
import { cn } from '@/lib/utils';

const norm = (w: string) => (w || '').trim().toLowerCase();

export default function WordChoiceGame({
  prompt,
  hint,
  options,
  answer,
  onResult,
}: {
  prompt: React.ReactNode; // phần gợi ý (audio hoặc nghĩa)
  hint: string; // dòng hướng dẫn nhỏ ("Nghe rồi chọn từ đúng")
  options: string[];
  answer: string;
  onResult: (correct: boolean) => void;
}) {
  const [picked, setPicked] = useState<string | null>(null);
  const answered = picked != null;
  const isRight = (o: string) => norm(o) === norm(answer);

  function choose(o: string) {
    if (answered) return;
    setPicked(o);
    onResult(isRight(o)); // báo kết quả ngay; dialog lo award + tự sang bài kế
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="text-center text-sm font-medium text-muted-foreground">{hint}</div>
      <div className="flex justify-center">{prompt}</div>
      <div className="grid grid-cols-2 gap-2">
        {options.map((o) => {
          const right = isRight(o);
          const chosen = picked != null && norm(picked) === norm(o);
          // Sau khi trả lời: tô xanh đáp án đúng, tô đỏ lựa chọn sai đã bấm.
          const state = !answered ? 'idle' : right ? 'right' : chosen ? 'wrong' : 'dim';
          return (
            <button
              key={o}
              type="button"
              disabled={answered}
              onClick={() => choose(o)}
              className={cn(
                'flex items-center justify-center gap-1.5 rounded-xl border px-3 py-3 text-base font-semibold transition-colors',
                state === 'idle' && 'hover:border-primary hover:bg-primary/5',
                state === 'right' && 'border-green-500 bg-green-50 text-green-700 dark:bg-green-950/40 dark:text-green-300',
                state === 'wrong' && 'border-red-500 bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300',
                state === 'dim' && 'opacity-50',
              )}
            >
              {state === 'right' && <Check className="h-4 w-4 shrink-0" />}
              {state === 'wrong' && <X className="h-4 w-4 shrink-0" />}
              <span className="truncate">{o}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
