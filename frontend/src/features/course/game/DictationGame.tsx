// ⌨️ Dictation Sprint — nghe TTS rồi GÕ LẠI từ (không hiện từ). Presentational THUẦN:
// nhận `word` (+ `ipa`) → tự phát mẫu khi vào bài, người học gõ, so khớp tất định
// (chuẩn hóa hoa/thường + bỏ dấu câu) rồi gọi `onResult(correct)` MỘT LẦN. Không
// mic, không AI, không backend — chỉ tái dùng /tts. Điều phối (award/chuyển bước) do
// CourseGameSession lo, giống ListenChoose.

import { useEffect, useRef, useState } from 'react';
import { Volume2, Check, X } from 'lucide-react';
import { playWordTts } from '@/features/grading/playback';
import { cn } from '@/lib/utils';

// So khớp chính tả: bỏ hoa/thường, dấu câu, chuẩn hóa khoảng trắng. Giữ chữ cái/số/
// nháy đơn (để "don't", "o'clock" khớp). Đủ chặt cho từ/cụm ngắn của lesson.
function normalize(s: string): string {
  return (s || '')
    .toLowerCase()
    .normalize('NFC')
    .replace(/[^\p{L}\p{N}\s']/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export default function DictationGame({
  word,
  ipa = '',
  onResult,
}: {
  word: string;
  ipa?: string;
  onResult: (correct: boolean) => void;
}) {
  const [value, setValue] = useState('');
  const [verdict, setVerdict] = useState<'idle' | 'right' | 'wrong'>('idle');
  const inputRef = useRef<HTMLInputElement>(null);

  // Tự phát mẫu khi vào bài (autoplay bị chặn thì im lặng — vẫn còn nút 🔊).
  useEffect(() => {
    const t = window.setTimeout(() => playWordTts(word, ipa), 250);
    inputRef.current?.focus();
    return () => window.clearTimeout(t);
  }, [word]); // eslint-disable-line react-hooks/exhaustive-deps

  const answered = verdict !== 'idle';

  function submit() {
    if (answered || !value.trim()) return;
    const correct = normalize(value) === normalize(word);
    setVerdict(correct ? 'right' : 'wrong');
    onResult(correct);
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="text-center text-sm font-medium text-muted-foreground">
        ⌨️ Nghe rồi gõ lại từ bạn nghe được
      </div>
      <button
        type="button"
        onClick={() => playWordTts(word, ipa)}
        className="flex h-16 w-16 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm transition-transform hover:scale-105 active:scale-95"
        title="Nghe lại"
      >
        <Volume2 className="h-7 w-7" />
      </button>

      <input
        ref={inputRef}
        type="text"
        value={value}
        disabled={answered}
        autoCapitalize="off"
        autoCorrect="off"
        spellCheck={false}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && submit()}
        placeholder="Gõ từ…"
        className={cn(
          'w-full max-w-xs rounded-xl border px-4 py-3 text-center text-lg font-semibold outline-none transition-colors',
          verdict === 'right' && 'border-green-500 bg-green-50 text-green-700 dark:bg-green-950/40 dark:text-green-300',
          verdict === 'wrong' && 'border-red-500 bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300',
          verdict === 'idle' && 'border-input focus:border-primary',
        )}
      />

      {verdict === 'wrong' && (
        <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <X className="h-4 w-4 text-red-600" /> Đáp án: <b className="text-foreground">{word}</b>
        </div>
      )}
      {verdict === 'right' && (
        <div className="flex items-center gap-1.5 text-sm text-green-600">
          <Check className="h-4 w-4" /> Chính xác!
        </div>
      )}

      {!answered && (
        <button
          type="button"
          onClick={submit}
          disabled={!value.trim()}
          className="rounded-xl bg-primary px-5 py-2 text-sm font-semibold text-primary-foreground disabled:opacity-40"
        >
          Kiểm tra
        </button>
      )}
    </div>
  );
}
