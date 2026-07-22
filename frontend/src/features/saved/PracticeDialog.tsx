// Popup luyện từ (ELSA-style) — port web/js/practice.js sang React + shadcn Dialog.
// Mở từ mọi tab qua usePractice (savedInterop bơm từ .practice-open, hoặc bảng Từ đã
// lưu/review-toast gọi trực tiếp). Vỏ modal = shadcn Dialog; phần chip phoneme/thẻ lỗi
// GIỮ class .practice-* của CSS legacy (đã tinh chỉnh kỹ) để bám sát hình cũ.

import { useEffect, useState } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Mic, Play } from 'lucide-react';
import { apiFetch } from '@/lib/api';
import { hasHangul } from '@/lib/config';
import { useUiStore } from '@/store/ui';
import { usePractice } from '@/store/practice';
import { useSavedWords, syncBookmarkButtons } from '@/store/savedWords';
import { useXp } from '@/store/xp';
import { celebrateGood, celebratePerfect } from '@/lib/celebrate';
import { playSfx } from '@/lib/sfx';
import { phonemeTip } from '@/lib/phonemeTips';
import { useWordGrader, practicePct } from './useWordGrader';
import { ipaStressString, toBritishWord } from '@/legacy/render';

// Cache word-info trong phiên (server cũng cache SQLite — đây chỉ đỡ round-trip).
const wordInfoCache = new Map<string, WordInfo | null>();
interface WordInfo {
  meaning?: string;
  definition_en?: string;
  example_en?: string;
  // Phiên âm UK/US từ /word-info (cascade cache→CMUdict→Cambridge→eSpeak). Chỉ có
  // khi bật TOEIC_IPA_CACHE_ENABLED; uk_ipa thường chỉ có khi nguồn là Cambridge.
  uk_ipa?: string;
  us_ipa?: string;
  // Dạng weak (function word: at /æt/→/ət/) + biến thể phụ (marry us /ˈmer.i/·/ˈmær.i/).
  // CHỈ có từ nguồn Cambridge; null ở nhánh CMUdict/eSpeak → không render hàng đó.
  uk_ipa_weak?: string | null;
  us_ipa_weak?: string | null;
  uk_ipa_alt?: string | null;
  us_ipa_alt?: string | null;
}

function practiceIsBad(p: any): boolean {
  return p.status !== 'skipped' && !(p.status === 'ok' || p.severity === 'low');
}
function ringColor(pct: number): string {
  return pct >= 80 ? '#16a34a' : pct >= 50 ? '#f59e0b' : '#dc2626';
}

export default function PracticeDialog() {
  const open = usePractice((s) => s.open);
  const data = usePractice((s) => s.data);
  const setData = usePractice((s) => s.setData);
  const close = usePractice((s) => s.close);
  const accent = useUiStore((s) => s.accent);
  const savedHas = useSavedWords((s) => s.keys); // re-render khi cache đổi
  const swAdd = useSavedWords((s) => s.add);
  const swRemove = useSavedWords((s) => s.remove);

  const [wordInfo, setWordInfo] = useState<WordInfo | null | undefined>(undefined);

  const word = data?.word || '';
  const isKo = hasHangul(word);
  const saved = !!word && savedHas.has(word.trim().toLowerCase());

  // Đường chấm dùng chung (thu âm → /grade → phoneme). onGraded set kết quả + ăn
  // mừng + cộng XP + cập nhật saved_word — phần ĐẶC THÙ của popup này.
  const { status, setStatus, recording, grading, replayUrl, setReplayUrl, toggleRecording, playReplay, cleanup } =
    useWordGrader({
      accent,
      onGraded: ({ word: gw, phonemes: merged, pct }) => {
        const d = usePractice.getState().data;
        if (!d) return;
        setData({ ...d, phonemes: merged, skip_reason: null });
        setStatus({
          text: pct >= 80 ? `Tuyệt vời — ${pct}%! 🎉` : `Được ${pct}% — nghe mẫu 🔊 rồi thử lại nhé.`,
          kind: pct >= 80 ? 'good' : '',
        });
        // Game hóa: ăn mừng theo mức điểm (1 SFX/lần chấm).
        if (pct === 100) celebratePerfect();
        else if (pct >= 80) celebrateGood();
        else if (pct < 50) playSfx('wrong');
        // Cộng XP luyện từ — client CHỈ gửi event+score, backend tự tính (RB#5) + cap ngày.
        useXp.getState().award('word_practice', pct / 100);
        // Từ đã lưu → cập nhật điểm + snapshot phonemes mới (im lặng).
        if (useSavedWords.getState().has(gw)) {
          swAdd({ word: gw, last_score: pct / 100, phonemes: merged }).catch(() => {});
        }
      },
    });

  // Reset khi mở từ khác / đổi data.
  useEffect(() => {
    if (!open || !word) return;
    setStatus({
      text: data?.skip_reason ? 'Lần chấm trước chưa nghe rõ từ này — bấm mic để luyện và chấm lại.' : '',
      kind: '',
    });
    setReplayUrl(null);
    // word-info (lazy). Từ Hangul: /word-info là từ điển EN → bỏ qua.
    if (isKo) {
      setWordInfo(null);
    } else {
      const key = word.toLowerCase();
      if (wordInfoCache.has(key)) {
        setWordInfo(wordInfoCache.get(key));
      } else {
        setWordInfo(undefined); // loading
        // /word-info chờ Cambridge đồng bộ (wait_cambridge) nên response đầu đã có CẢ
        // uk_ipa lẫn us_ipa → popup hiện cả hai ngay, không cần refetch/upgrade.
        apiFetch(`/word-info?word=${encodeURIComponent(key)}`)
          .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
          .then((info: WordInfo) => {
            wordInfoCache.set(key, info);
            if (usePractice.getState().data?.word.toLowerCase() === key) setWordInfo(info);
          })
          .catch(() => {
            if (usePractice.getState().data?.word.toLowerCase() === key) setWordInfo(null);
          });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, word]);

  // Dọn stream + replay URL khi đóng (component vẫn mounted, chỉ đổi `open`).
  useEffect(() => {
    if (!open) cleanup();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  async function toggleBookmark() {
    if (!word || isKo) return;
    try {
      if (saved) await swRemove(word);
      else await swAdd({ word, ipa: data?.ipa, phonemes: data?.phonemes, accuracy: data?.accuracy });
      syncBookmarkButtons(word);
    } catch (err: any) {
      alert(`Lỗi lưu từ: ${err.message || err}`);
    }
  }

  if (!data) return null;

  // ── IPA + phoneme hiển thị (áp GB nếu chọn, trừ tiếng Hàn) ──
  const phonemes: any[] = data.phonemes || [];
  const disp = accent === 'gb' && !isKo ? toBritishWord({ phonemes }).phonemes : phonemes;
  const shown = (disp || []).filter((p: any) => !p._hidden);
  const ipaStr = (ipaStressString(disp) || data.ipa || '').trim();
  // Phiên âm theo TỪNG accent (độc lập với accent đang chọn) để hiện + đọc CẢ UK và US.
  // US: ưu tiên từ điển (Cambridge us_ipa) → CMUdict phonemes → data.ipa.
  // UK: ưu tiên từ điển (uk_ipa) → suy từ phonemes qua toBritishWord (như chip lỗi) khi
  //     nguồn là CMUdict (uk_ipa thường trống). Rỗng cả hai → không hiện hàng đó.
  // Phiên âm theo từng accent — CHỈ hiện hàng có dữ liệu (không bịa/không mượn chéo).
  // US: từ điển us_ipa → suy từ phonemes → data.ipa.
  const usIpa = (wordInfo?.us_ipa || ipaStressString(phonemes) || data.ipa || '').trim();
  // UK: từ điển uk_ipa → suy từ phonemes (toBritishWord). Rỗng (mở từ 1 từ đã lưu,
  // chưa có phonemes + nguồn CMUdict chưa có uk_ipa) → ẩn hàng UK. Cambridge warm nền
  // sau đó điền uk_ipa → lần mở lại (cache đã đổi) hiện cả hai.
  const ukIpa = (
    wordInfo?.uk_ipa ||
    (phonemes.length ? ipaStressString(toBritishWord({ phonemes }).phonemes) : '') ||
    ''
  ).trim();
  // Biến thể phụ mỗi accent (weak + variant không nhãn) — CHỈ từ Cambridge (wordInfo).
  // Lọc rỗng + trùng phiên âm chính để không hiện dòng thừa. tag → nhãn nhỏ hiển thị.
  const extraRows = (
    primary: string,
    weak?: string | null,
    alt?: string | null,
  ): { tag: string; ipa: string }[] =>
    [
      { tag: 'weak', ipa: (weak || '').trim() },
      { tag: 'var', ipa: (alt || '').trim() },
    ].filter((v) => v.ipa && v.ipa !== primary);
  const ukExtras = extraRows(ukIpa, wordInfo?.uk_ipa_weak, wordInfo?.uk_ipa_alt);
  const usExtras = extraRows(usIpa, wordInfo?.us_ipa_weak, wordInfo?.us_ipa_alt);
  const pct = practicePct(phonemes);
  const bad = shown.filter(practiceIsBad);
  const skippedCount = shown.filter((p: any) => p.status === 'skipped').length;
  const phraseWords = String(word).split(' ');

  return (
    <Dialog open={open} onOpenChange={(o) => !o && close()}>
      <DialogContent className="max-h-[90vh] max-w-md gap-3 overflow-y-auto">
        {/* Header */}
        <div className="flex items-center gap-2">
          <h3 className="practice-head__word text-xl font-bold">{word}</h3>
          {!isKo && (
            <button
              type="button"
              className={`practice-bookmark${saved ? ' saved' : ''}`}
              onClick={toggleBookmark}
              title={saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}
            >
              {saved ? '★' : '☆'}
            </button>
          )}
          <span className="flex-1" />
          {/* Chưa có điểm thì KHÔNG vẽ vòng tròn rỗng (trông như widget hỏng) — chỉ
              hiện khi đã chấm ít nhất 1 lần. Chừa chỗ cho nút đóng của Dialog. */}
          {pct != null && (
            <div
              className="practice-ring mr-7"
              style={
                {
                  ['--pct' as any]: pct,
                  ['--ring-color' as any]: ringColor(pct),
                } as React.CSSProperties
              }
              title={`Độ chính xác lần chấm gần nhất: ${pct}%`}
            >
              <div className="practice-ring__inner">{pct}%</div>
            </div>
          )}
        </div>

        {/* IPA — tiếng Hàn: 1 dòng (không có UK/US). Tiếng Anh: 2 hàng UK + US, mỗi
            hàng đọc ĐÚNG giọng của nó qua data-accent (playWordTts). */}
        {isKo ? (
          <div className="practice-ipa">
            {ipaStr ? `/${ipaStr}/` : ''}
            <button type="button" className="tts-play" data-word={word} data-ipa={ipaStr || undefined} title="Nghe phát âm chuẩn">
              🔊
            </button>
          </div>
        ) : (
          <div className="practice-ipa practice-ipa--dual">
            {ukIpa && (
              <span className="practice-ipa__row">
                <span className="practice-ipa__accent">UK</span>&nbsp;/{ukIpa}/
                <button type="button" className="tts-play" data-word={word} data-ipa={ukIpa} data-accent="gb" title="Nghe giọng Anh (UK)">
                  🔊
                </button>
              </span>
            )}
            {/* Biến thể UK (weak/variant) — đọc cùng giọng gb, dữ liệu = IPA của biến thể. */}
            {ukIpa &&
              ukExtras.map((v) => (
                <span className="practice-ipa__row practice-ipa__row--variant" key={`uk-${v.tag}-${v.ipa}`}>
                  <span className="practice-ipa__variant-tag">{v.tag}</span>&nbsp;/{v.ipa}/
                  <button type="button" className="tts-play" data-word={word} data-ipa={v.ipa} data-accent="gb" title={`Nghe biến thể (${v.tag})`}>
                    🔊
                  </button>
                </span>
              ))}
            {usIpa && (
              <span className="practice-ipa__row">
                <span className="practice-ipa__accent">US</span>&nbsp;/{usIpa}/
                <button type="button" className="tts-play" data-word={word} data-ipa={usIpa} data-accent="us" title="Nghe giọng Mỹ (US)">
                  🔊
                </button>
              </span>
            )}
            {usIpa &&
              usExtras.map((v) => (
                <span className="practice-ipa__row practice-ipa__row--variant" key={`us-${v.tag}-${v.ipa}`}>
                  <span className="practice-ipa__variant-tag">{v.tag}</span>&nbsp;/{v.ipa}/
                  <button type="button" className="tts-play" data-word={word} data-ipa={v.ipa} data-accent="us" title={`Nghe biến thể (${v.tag})`}>
                    🔊
                  </button>
                </span>
              ))}
          </div>
        )}

        {/* Định nghĩa + ví dụ */}
        <div className="practice-info">
          {wordInfo === undefined && <span className="practice-info__loading">Đang tải định nghĩa…</span>}
          {wordInfo && (
            <>
              {wordInfo.meaning && <div className="practice-info__meaning">🇻🇳 {wordInfo.meaning}</div>}
              <div className="practice-info__label">Định nghĩa</div>
              <div>{wordInfo.definition_en || ''}</div>
              <div className="practice-info__label">Ví dụ</div>
              <div>{wordInfo.example_en || ''}</div>
            </>
          )}
        </div>

        {/* Chip phoneme + thẻ lỗi */}
        {shown.length > 0 && (
          <div>
            <div className="practice-chips">{renderChips(shown)}</div>
            {bad.length ? (
              <>
                <div className="practice-fix-list__title">Cần cải thiện</div>
                {bad.map((p: any, i: number) => renderFixCard(p, i, phraseWords))}
              </>
            ) : (
              <div className="practice-allok">🎉 Tất cả các âm đều chính xác!</div>
            )}
            {skippedCount > 0 && (
              <div className="practice-skipnote">
                Âm màu xám chưa chấm được ở lần nói này — bấm mic thử lại.
              </div>
            )}
          </div>
        )}

        {/* Ghi âm + chấm lại */}
        <div className="practice-rec">
          <div className="practice-rec__hint">Chạm để nói — chấm lại riêng từ này</div>
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
      </DialogContent>
    </Dialog>
  );

  // ── Helpers render (JSX thay chuỗi HTML của legacy) ──
  function renderChips(list: any[]) {
    const out: React.ReactNode[] = [];
    let prevW: any;
    list.forEach((p, idx) => {
      if (out.length && p._w !== prevW) out.push(<span key={`gap-${idx}`} className="practice-chip-gap" />);
      prevW = p._w;
      const cls = p.status === 'skipped' ? 'skip' : practiceIsBad(p) ? 'bad' : 'ok';
      const info = phonemeTip(p.symbol);
      const sym = `/${p.symbol}/`;
      out.push(
        info && info.example ? (
          <button
            key={idx}
            type="button"
            className={`practice-chip ${cls} tts-play`}
            data-word={info.example}
            title={`Nghe âm này trong từ “${info.example}”`}
          >
            {sym}
          </button>
        ) : (
          <span key={idx} className={`practice-chip ${cls}`}>
            {sym}
          </span>
        ),
      );
    });
    return out;
  }

  function renderFixCard(p: any, i: number, words: string[]) {
    const info = phonemeTip(p.symbol);
    const heard = p.status === 'del' ? '∅ thiếu âm' : `/${p.heard ?? '?'}/`;
    const tip = info ? info.tip : 'Nghe mẫu và bắt chước khẩu hình — chú ý vị trí lưỡi và môi.';
    const inWord = p._w != null && words.length > 1 && words[p._w] ? words[p._w] : null;
    return (
      <div className="practice-fix" key={`fix-${i}`}>
        <div className="practice-fix__row">
          <span className="practice-fix__target">/{p.symbol}/</span>
          {inWord && <span className="practice-fix__word">trong “{inWord}”</span>}
          <span className="practice-fix__label">bạn nói</span>
          <span className="practice-fix__heard">{heard}</span>
          {info && info.example && (
            <button
              type="button"
              className="tts-play practice-fix__play"
              data-word={info.example}
              title={`Nghe âm này trong từ “${info.example}”`}
            >
              🔊 {info.example}
            </button>
          )}
        </div>
        <div className="practice-tip">{tip}</div>
      </div>
    );
  }
}
