// Vòng chơi cho lesson phát âm: Learn → Play (mini-game) → Review (ôn từ sai, KHÔNG
// XP) → Boss (đọc to → chấm thật → hoàn thành lesson). Component này CHỈ ĐIỀU PHỐI:
// dựng trình tự, đổi bước, tính combo, award word_recall (đúng MỘT lần/từ ở LẦN chơi
// đầu — Review không award → không farm được), rồi ủy thác chấm Boss cho hook DÙNG
// CHUNG useLessonPronGrade (cùng đường /grade với màn cũ). Logic chơi của từng game
// nằm trong component game độc lập, không ở đây.

import { useEffect, useMemo, useRef, useState } from 'react';
import { useXp } from '@/store/xp';
import { useUiStore } from '@/store/ui';
import { COURSE_GAME_DICTATION, COURSE_GAME_SHADOWING } from '@/lib/config';
import ListenChoose from '@/features/saved/minigames/ListenChoose';
import type { SavedWord } from '@/store/savedWords';
import type { LessonContent, PronWord } from '../courseApi';
import { useLessonPronGrade } from '../useLessonPronGrade';
import { buildCourseGameSequence, type GameStep } from './buildCourseGameSequence';
import { fetchWordInfos, type WordInfo } from './wordInfo';
import WordMatchGame, { type MatchResult } from './WordMatchGame';
import SentenceBuilderGame from './SentenceBuilderGame';
import DictationGame from './DictationGame';
import ShadowingGame from './ShadowingGame';

const EMPTY_INFO: WordInfo = { meaning: null, example: null };
const key = (w: string) => (w || '').trim().toLowerCase();
const asSaved = (w: PronWord): SavedWord => ({ word: w.word, ipa: w.ipa });

type Phase = 'loading' | 'learn' | 'play' | 'review' | 'boss';

export default function CourseGameSession({
  lesson,
  onCompleted,
}: {
  lesson: LessonContent;
  onCompleted: (score: number) => void;
}) {
  const accent = useUiStore((s) => s.accent);
  const words = useMemo(() => lesson.words || [], [lesson.words]);
  const pool = useMemo(() => words.map((w) => w.word), [words]);
  const byWord = useMemo(() => new Map(words.map((w) => [key(w.word), w])), [words]);

  const [phase, setPhase] = useState<Phase>('loading');
  const [infos, setInfos] = useState<Map<string, WordInfo>>(new Map());
  const [steps, setSteps] = useState<GameStep[]>([]);
  const [index, setIndex] = useState(0);
  const [combo, setCombo] = useState(0);

  // Từ sai trong pha Play → gom để ôn ở Review (dedupe, giữ thứ tự xuất hiện).
  const missedRef = useRef<Map<string, PronWord>>(new Map());
  const [missed, setMissed] = useState<PronWord[]>([]);
  const [reviewIdx, setReviewIdx] = useState(0);

  // Từ đã award word_recall — đảm bảo mỗi từ chỉ cộng XP ở LẦN chơi đầu.
  const awardedRef = useRef<Set<string>>(new Set());

  // Nạp nghĩa + câu ví dụ cho mọi từ (1 fetch /từ, cache server) rồi dựng trình tự.
  useEffect(() => {
    let alive = true;
    fetchWordInfos(pool).then((map) => {
      if (!alive) return;
      setInfos(map);
      const infoOf = (w: string) => map.get(key(w)) || EMPTY_INFO;
      setSteps(
        buildCourseGameSequence(words, infoOf, {
          dictation: COURSE_GAME_DICTATION,
          shadowing: COURSE_GAME_SHADOWING,
        }),
      );
      setPhase('learn');
    });
    return () => {
      alive = false;
    };
  }, [pool, words]);

  const infoOf = (w: string) => infos.get(key(w)) || EMPTY_INFO;

  function awardWords(ws: string[], correctOf: (w: string) => boolean) {
    for (const w of ws) {
      const k = key(w);
      if (awardedRef.current.has(k)) continue;
      awardedRef.current.add(k);
      void useXp.getState().award('word_recall', correctOf(k) ? 1 : 0);
    }
  }

  function markMissed(k: string) {
    const w = byWord.get(k);
    if (w && !missedRef.current.has(k)) missedRef.current.set(k, w);
  }

  // Kết thúc pha Play → có từ sai thì Review, không thì vào Boss.
  function afterPlay() {
    const list = Array.from(missedRef.current.values());
    setMissed(list);
    setPhase(list.length ? 'review' : 'boss');
  }

  function advancePlay(stepCorrect: boolean) {
    setCombo((c) => (stepCorrect ? c + 1 : 0));
    window.setTimeout(() => {
      if (index + 1 < steps.length) setIndex(index + 1);
      else afterPlay();
    }, 900);
  }

  // Kết quả một bước trên MỘT từ (Listen / Dictation / Sentence Builder / Shadowing):
  // award word_recall (1 lần/từ), gom từ sai, tính combo, chuyển bước.
  function onWordResult(word: PronWord, correct: boolean) {
    const k = key(word.word);
    awardWords([k], () => correct);
    if (!correct) markMissed(k);
    advancePlay(correct);
  }

  function onMatchDone(results: MatchResult[]) {
    const correctSet = new Set(results.filter((r) => r.correct).map((r) => key(r.word)));
    awardWords(
      results.map((r) => r.word),
      (k) => correctSet.has(k),
    );
    for (const r of results) if (!r.correct) markMissed(key(r.word));
    advancePlay(results.every((r) => r.correct));
  }

  // ── Render ────────────────────────────────────────────────────────────────

  if (phase === 'loading') {
    return <p className="history-empty">⏳ Đang chuẩn bị vòng chơi…</p>;
  }

  if (phase === 'learn') {
    return (
      <div className="course-game">
        <div className="course-section-label">🎯 {words.length} từ mới — nghe rồi bắt đầu</div>
        <div className="course-pron-words">
          {words.map((w) => (
            <div className="course-pron-word" key={w.word}>
              <button
                type="button"
                className="tts-play course-pron-word__play"
                data-word={w.word}
                data-ipa={w.ipa || undefined}
                title={`Nghe “${w.word}”`}
              >
                🔊
              </button>
              <span className="course-pron-word__text">
                <b>{w.word}</b> {w.ipa && <span className="course-pron-word__ipa">/{w.ipa}/</span>}
                {infoOf(w.word).meaning && (
                  <span className="course-pron-word__reason">{infoOf(w.word).meaning}</span>
                )}
              </span>
            </div>
          ))}
        </div>
        <div className="course-complete">
          <button type="button" className="btn btn-primary" onClick={() => setPhase('play')}>
            ▶ Bắt đầu chơi
          </button>
        </div>
      </div>
    );
  }

  if (phase === 'play') {
    const step = steps[index];
    return (
      <div className="course-game">
        <div className="course-game__bar">
          <span>Bước {index + 1}/{steps.length}</span>
          {combo >= 2 && <span className="course-game__combo">🔥 Combo {combo}</span>}
        </div>
        <div className="course-game__stage" key={index}>
          {step.kind === 'match' && (
            <WordMatchGame
              pairs={step.words.map((w) => ({ word: w.word, meaning: infoOf(w.word).meaning || '' }))}
              onDone={onMatchDone}
            />
          )}
          {step.kind === 'listen' && (
            <ListenChoose
              word={asSaved(step.word)}
              pool={pool}
              onResult={(c) => onWordResult(step.word, c)}
            />
          )}
          {step.kind === 'dictate' && (
            <DictationGame
              word={step.word.word}
              ipa={step.word.ipa}
              onResult={(c) => onWordResult(step.word, c)}
            />
          )}
          {step.kind === 'build' && (
            <SentenceBuilderGame
              sentence={step.example}
              onResult={(c) => onWordResult(step.word, c)}
            />
          )}
          {step.kind === 'shadow' && (
            <ShadowingGame
              lesson={lesson}
              sentence={step.example}
              accent={accent}
              threshold={lesson.done_threshold}
              onResult={(c) => onWordResult(step.word, c)}
            />
          )}
        </div>
      </div>
    );
  }

  if (phase === 'review') {
    const w = missed[reviewIdx];
    return (
      <div className="course-game">
        <div className="course-section-label">🔁 Ôn nhanh từ chưa chắc (không tính XP)</div>
        <div className="course-game__bar">
          <span>Từ {reviewIdx + 1}/{missed.length}</span>
          <button type="button" className="btn btn-secondary btn-inline" onClick={() => setPhase('boss')}>
            Bỏ qua →
          </button>
        </div>
        <div className="course-game__stage" key={`rv-${reviewIdx}`}>
          <ListenChoose
            word={asSaved(w)}
            pool={pool}
            onResult={() => {
              window.setTimeout(() => {
                if (reviewIdx + 1 < missed.length) setReviewIdx(reviewIdx + 1);
                else setPhase('boss');
              }, 900);
            }}
          />
        </div>
      </div>
    );
  }

  // phase === 'boss'
  return <BossStage lesson={lesson} words={words} onCompleted={onCompleted} />;
}

// Boss — CHỈ là lớp UI quanh hook chấm DÙNG CHUNG (không engine chấm mới).
function BossStage({
  lesson,
  words,
  onCompleted,
}: {
  lesson: LessonContent;
  words: PronWord[];
  onCompleted: (score: number) => void;
}) {
  const { recording, grading, status, toggle } = useLessonPronGrade(lesson, words, (pct) =>
    onCompleted(pct / 100),
  );
  return (
    <div className="course-game">
      <div className="course-section-label">👾 Boss — đọc to tất cả các từ để chiến thắng</div>
      <div className="course-game__boss-words">{words.map((w) => w.word).join(' · ')}</div>
      <div className="course-complete">
        <div className="course-complete__hint">
          Đọc to, rõ tất cả các từ. Đạt {Math.round(lesson.done_threshold * 100)}% âm đúng để hạ Boss.
        </div>
        <button
          type="button"
          className={'practice-mic' + (recording ? ' recording' : '')}
          onClick={toggle}
          disabled={grading}
          title="Ghi âm đọc các từ"
        >
          🎙️
        </button>
        {status && <div className="course-complete__status">{status}</div>}
      </div>
    </div>
  );
}
