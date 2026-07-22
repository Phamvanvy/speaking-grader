// Mini-game "Nghe & chọn từ": tự phát TTS từ đúng khi vào bài, người dùng chọn 1
// trong 4 từ (distractor lấy từ chính từ đã lưu). Không nói/không chấm mic — tái
// dùng /tts (playWordTts) + khung WordChoiceGame. Mount lại mỗi bài (key=index).

import { useEffect, useMemo } from 'react';
import { Volume2 } from 'lucide-react';
import { playWordTts } from '@/features/grading/playback';
import type { SavedWord } from '@/store/savedWords';
import { buildChoices } from './wordChoice';
import WordChoiceGame from './WordChoiceGame';

function wordIpa(w: SavedWord): string {
  return (w.ipa || '').trim();
}

export default function ListenChoose({
  word,
  pool,
  onResult,
}: {
  word: SavedWord;
  pool: string[]; // các từ đã lưu (nguồn distractor)
  onResult: (correct: boolean) => void;
}) {
  const options = useMemo(() => buildChoices(word.word, pool, 4), [word.word, pool]);

  // Tự phát mẫu khi vào bài (autoplay bị chặn thì im lặng — vẫn còn nút 🔊 để bấm).
  useEffect(() => {
    const t = window.setTimeout(() => playWordTts(word.word, wordIpa(word)), 250);
    return () => window.clearTimeout(t);
  }, [word.word]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <WordChoiceGame
      hint="🎧 Nghe rồi chọn từ bạn nghe được"
      prompt={
        <button
          type="button"
          onClick={() => playWordTts(word.word, wordIpa(word))}
          className="flex h-16 w-16 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-sm transition-transform hover:scale-105 active:scale-95"
          title="Nghe lại"
        >
          <Volume2 className="h-7 w-7" />
        </button>
      }
      options={options}
      answer={word.word}
      onResult={onResult}
    />
  );
}
