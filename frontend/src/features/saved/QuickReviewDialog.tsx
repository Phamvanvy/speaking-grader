// Phiên "Luyện nhanh" — biến bảng Từ đã lưu thành vòng chơi có nhịp: luyện liên
// tiếp N từ (ưu tiên yếu/đến hạn), combo 🔥 khi đúng liên tục, màn tổng kết. Tái
// dùng CÙNG đường chấm với popup luyện qua useWordGrader (không viết lại thu âm/
// /grade). Combo là ĐỘNG LỰC/cosmetic — KHÔNG đổi cách tính XP (backend cấp XP mỗi
// từ, có cap ngày; y hệt popup luyện).

import { useEffect, useRef, useState } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Mic, Play, SkipForward, ChevronRight, Trophy } from 'lucide-react';
import { useUiStore } from '@/store/ui';
import { useSavedWords } from '@/store/savedWords';
import { useXp } from '@/store/xp';
import {
  useQuickReview,
  comboAt,
  maxCombo,
  correctCount,
  PASS_PCT,
  type ReviewResult,
} from '@/store/quickReview';
import { useWordGrader } from './useWordGrader';
import ListenChoose from './minigames/ListenChoose';
import MeaningRecall from './minigames/MeaningRecall';
import { celebrateGood, celebratePerfect, celebrateComplete, bigCelebrate } from '@/lib/celebrate';
import { playSfx } from '@/lib/sfx';
import { phonemeTip } from '@/lib/phonemeTips';
import { ipaStressString } from '@/legacy/render';

const ADVANCE_MS = 1500; // tự sang từ kế sau khi chấm (kịp đọc điểm), bấm "Tiếp" để bỏ chờ

function practiceIsBad(p: any): boolean {
  return p.status !== 'skipped' && !(p.status === 'ok' || p.severity === 'low');
}
function ringColor(pct: number): string {
  return pct >= 80 ? '#16a34a' : pct >= 50 ? '#f59e0b' : '#dc2626';
}
/** Mức thành thạo 0–3 sao theo điểm (đồng bộ SavedTab). */
function masteryStars(v: number | null): number {
  if (v == null) return 0;
  if (v >= 0.85) return 3;
  if (v >= 0.7) return 2;
  if (v >= 0.5) return 1;
  return 0;
}
function wordIpa(w: { ipa?: string | null; phonemes?: any[] }): string {
  const s = (w.phonemes || []).length ? ipaStressString(w.phonemes) : w.ipa || '';
  return s ? `/${s}/` : '';
}

export default function QuickReviewDialog() {
  const open = useQuickReview((s) => s.open);
  const queue = useQuickReview((s) => s.queue);
  const kinds = useQuickReview((s) => s.kinds);
  const index = useQuickReview((s) => s.index);
  const results = useQuickReview((s) => s.results);
  const phase = useQuickReview((s) => s.phase);
  const record = useQuickReview((s) => s.record);
  const advance = useQuickReview((s) => s.advance);
  const close = useQuickReview((s) => s.close);
  const accent = useUiStore((s) => s.accent);
  const swAdd = useSavedWords((s) => s.add);
  const savedList = useSavedWords((s) => s.words); // nguồn distractor cho mini-game

  // Phoneme + % của từ ĐANG luyện (để vẽ vòng điểm + chip lỗi); reset mỗi từ.
  const [graded, setGraded] = useState<{ pct: number; phonemes: any[] } | null>(null);
  const advanceTimer = useRef<number | null>(null);

  const cur = queue[index];
  const word = cur?.word || '';

  const { status, setStatus, recording, grading, replayUrl, setReplayUrl, toggleRecording, playReplay, cleanup } =
    useWordGrader({
      accent,
      onGraded: ({ word: gw, phonemes, pct }) => {
        record(pct); // cập nhật store trước để comboAt đọc được kết quả mới
        setGraded({ pct, phonemes });
        // Cộng XP (backend tự tính + cap ngày) + cập nhật điểm từ đã lưu (im lặng).
        useXp.getState().award('word_practice', pct / 100);
        if (useSavedWords.getState().has(gw)) {
          swAdd({ word: gw, last_score: pct / 100, phonemes }).catch(() => {});
        }
        // Ăn mừng theo combo: đúng liên tiếp càng dài càng "đã".
        const combo = comboAt(useQuickReview.getState().results, useQuickReview.getState().index);
        if (pct === 100) celebratePerfect();
        else if (pct >= PASS_PCT) {
          celebrateGood();
          if (combo >= 3) bigCelebrate();
        } else playSfx('wrong');
        setStatus({
          text:
            pct >= PASS_PCT
              ? combo >= 2
                ? `Tuyệt! ${pct}% · combo x${combo} 🔥`
                : `Tốt lắm — ${pct}%!`
              : `Được ${pct}% — nghe mẫu 🔊 rồi luyện lại từ này sau nhé.`,
          kind: pct >= PASS_PCT ? 'good' : '',
        });
        scheduleAdvance();
      },
    });

  function clearAdvance() {
    if (advanceTimer.current != null) {
      window.clearTimeout(advanceTimer.current);
      advanceTimer.current = null;
    }
  }
  function scheduleAdvance() {
    clearAdvance();
    advanceTimer.current = window.setTimeout(() => {
      advanceTimer.current = null;
      advance();
    }, ADVANCE_MS);
  }
  function goNext() {
    clearAdvance();
    advance();
  }

  // Kết quả 1 lượt mini-game (nghe-chọn / nghĩa-nhớ): chấm nhị phân đúng/sai. Cộng
  // XP qua event 'word_recall' (CHUNG cap ngày). KHÔNG đụng last_score/phonemes —
  // đó là điểm phát âm (mastery/sao), mini-game recall không phải phát âm.
  function finishMinigame(correct: boolean) {
    const pct = correct ? 100 : 0;
    record(pct);
    useXp.getState().award('word_recall', pct / 100);
    const combo = comboAt(useQuickReview.getState().results, useQuickReview.getState().index);
    if (correct) {
      celebrateGood();
      if (combo >= 3) bigCelebrate();
    } else playSfx('wrong');
    scheduleAdvance();
  }

  // Đổi từ (index) → reset trạng thái chấm của từ trước.
  useEffect(() => {
    setGraded(null);
    setStatus({ text: '', kind: '' });
    setReplayUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    cleanup(); // dừng mic nếu còn ghi dở
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index]);

  // Vào màn tổng kết → dọn mic + ăn mừng nếu làm tốt.
  useEffect(() => {
    if (phase !== 'summary') return;
    clearAdvance();
    cleanup();
    const done = results.filter(Boolean) as ReviewResult[];
    const ratio = done.length ? correctCount(results) / done.length : 0;
    if (ratio >= 0.8) bigCelebrate();
    else celebrateComplete();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  // Dọn timer khi unmount.
  useEffect(() => clearAdvance, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (!open) return null;

  const total = queue.length;
  const combo = comboAt(results, index);
  const kind = kinds[index] ?? 'speak';
  // Distractor cho mini-game = các từ đã lưu (bỏ chính từ đang hỏi); dự phòng dùng
  // các từ trong queue nếu store rỗng.
  const pool = (savedList.length ? savedList : queue).map((w) => w.word);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && close()}>
      <DialogContent className="max-h-[92vh] max-w-md gap-4 overflow-y-auto">
        {phase === 'summary' ? (
          <Summary results={results} onClose={close} />
        ) : (
          <>
            {/* Tiến độ + combo */}
            <div className="flex items-center gap-3 pr-7">
              <span className="text-sm font-semibold text-muted-foreground">
                Luyện nhanh · {Math.min(index + 1, total)}/{total}
              </span>
              {combo >= 2 && (
                <span className="rounded-full bg-orange-100 px-2 py-0.5 text-sm font-bold text-orange-600 dark:bg-orange-950/40 dark:text-orange-400">
                  🔥 combo x{combo}
                </span>
              )}
              <span className="flex-1" />
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${total ? (index / total) * 100 : 0}%` }}
              />
            </div>

            {kind === 'speak' ? (
              <>
                {/* Từ + IPA + nghe mẫu + vòng điểm */}
                <div className="flex items-center gap-2">
                  <h3 className="text-2xl font-bold">{word}</h3>
                  <span className="flex-1" />
                  {graded && (
                    <div
                      className="practice-ring"
                      style={
                        {
                          ['--pct' as any]: graded.pct,
                          ['--ring-color' as any]: ringColor(graded.pct),
                        } as React.CSSProperties
                      }
                      title={`Độ chính xác: ${graded.pct}%`}
                    >
                      <div className="practice-ring__inner">{graded.pct}%</div>
                    </div>
                  )}
                </div>
                {wordIpa(cur) && (
                  <div className="practice-ipa">
                    {wordIpa(cur)}
                    <button type="button" className="tts-play" data-word={word} data-ipa={wordIpa(cur).replace(/\//g, '') || undefined} title="Nghe phát âm chuẩn">
                      🔊
                    </button>
                  </div>
                )}

                {/* Chip lỗi rút gọn sau khi chấm */}
                {graded && graded.phonemes.length > 0 && <CompactChips phonemes={graded.phonemes} />}

                {/* Ghi âm */}
                <div className="practice-rec">
                  <div className="practice-rec__hint">Chạm để nói từ này</div>
                  <button
                    type="button"
                    className={`practice-mic${recording ? ' recording' : ''}`}
                    onClick={() => toggleRecording(word)}
                    disabled={grading}
                    title="Ghi âm luyện tập"
                  >
                    <Mic className="h-5 w-5" />
                  </button>
                  <div className={`practice-status${status.kind ? ' ' + status.kind : ''}`}>{status.text}</div>
                  {replayUrl && (
                    <Button type="button" variant="outline" size="sm" className="mt-1" onClick={playReplay}>
                      <Play className="h-3.5 w-3.5" /> Nghe lại bạn vừa nói
                    </Button>
                  )}
                </div>
              </>
            ) : cur && kind === 'listen' ? (
              <ListenChoose key={index} word={cur} pool={pool} onResult={finishMinigame} />
            ) : cur ? (
              <MeaningRecall key={index} word={cur} pool={pool} onResult={finishMinigame} />
            ) : null}

            {/* Điều hướng */}
            <div className="flex items-center justify-between gap-2">
              <Button type="button" variant="ghost" size="sm" className="text-muted-foreground" onClick={goNext} disabled={grading}>
                <SkipForward className="h-4 w-4" /> Bỏ qua
              </Button>
              <Button type="button" size="sm" onClick={goNext} disabled={grading}>
                {index + 1 < total ? 'Từ tiếp theo' : 'Xem kết quả'}
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** Chip phoneme rút gọn — chỉ nhấn màu ok/bad/skip, bấm để nghe âm mẫu trong 1 từ. */
function CompactChips({ phonemes }: { phonemes: any[] }) {
  const shown = (phonemes || []).filter((p: any) => !p._hidden);
  if (!shown.length) return null;
  return (
    <div className="practice-chips">
      {shown.map((p: any, idx: number) => {
        const cls = p.status === 'skipped' ? 'skip' : practiceIsBad(p) ? 'bad' : 'ok';
        const info = phonemeTip(p.symbol);
        const sym = `/${p.symbol}/`;
        return info && info.example ? (
          <button key={idx} type="button" className={`practice-chip ${cls} tts-play`} data-word={info.example} title={`Nghe âm này trong “${info.example}”`}>
            {sym}
          </button>
        ) : (
          <span key={idx} className={`practice-chip ${cls}`}>
            {sym}
          </span>
        );
      })}
    </div>
  );
}

/** Màn tổng kết cuối phiên: số đúng, combo cao nhất, số sao tăng thêm. */
function Summary({ results, onClose }: { results: (ReviewResult | null)[]; onClose: () => void }) {
  const start = useQuickReview((s) => s.start);
  const queue = useQuickReview((s) => s.queue);
  const done = results.filter(Boolean) as ReviewResult[];
  const correct = correctCount(results);
  const best = maxCombo(results);
  // Sao thành thạo chỉ tính bài NÓI (last_score là điểm phát âm); mini-game recall
  // không nâng sao dù chọn đúng.
  const starsGained = done.reduce(
    (sum, r) =>
      sum + (r.kind === 'speak' ? Math.max(0, masteryStars(r.pct / 100) - masteryStars(r.prevScore)) : 0),
    0,
  );
  const ratio = done.length ? correct / done.length : 0;

  return (
    <div className="flex flex-col items-center gap-4 py-2 text-center">
      <div className="text-4xl">{ratio >= 0.8 ? '🏆' : ratio >= 0.5 ? '🎉' : '💪'}</div>
      <div>
        <h3 className="text-xl font-bold">Hoàn thành phiên!</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          {ratio >= 0.8 ? 'Xuất sắc — giữ nhịp này nhé!' : ratio >= 0.5 ? 'Tiến bộ tốt, luyện thêm là chuẩn!' : 'Cứ luyện đều, sẽ lên nhanh thôi!'}
        </p>
      </div>
      <div className="grid w-full grid-cols-3 gap-2">
        <Stat label="Đúng" value={`${correct}/${done.length}`} icon="✅" />
        <Stat label="Combo cao nhất" value={best >= 2 ? `x${best}` : '—'} icon="🔥" />
        <Stat label="Sao mới" value={starsGained > 0 ? `+${starsGained}` : '0'} icon="⭐" />
      </div>
      <div className="flex w-full gap-2">
        <Button variant="outline" className="flex-1" onClick={() => start(queue)}>
          <Trophy className="h-4 w-4" /> Luyện lại
        </Button>
        <Button className="flex-1" onClick={onClose}>
          Xong
        </Button>
      </div>
    </div>
  );
}

function Stat({ label, value, icon }: { label: string; value: string; icon: string }) {
  return (
    <div className="rounded-xl border bg-muted/30 p-3">
      <div className="text-lg">{icon}</div>
      <div className="mt-0.5 text-lg font-bold tabular-nums">{value}</div>
      <div className="text-[0.7rem] leading-tight text-muted-foreground">{label}</div>
    </div>
  );
}
