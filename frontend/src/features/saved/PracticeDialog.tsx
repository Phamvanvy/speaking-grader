// Popup luyện từ (ELSA-style) — port web/js/practice.js sang React + shadcn Dialog.
// Mở từ mọi tab qua usePractice (savedInterop bơm từ .practice-open, hoặc bảng Từ đã
// lưu/review-toast gọi trực tiếp). Vỏ modal = shadcn Dialog; phần chip phoneme/thẻ lỗi
// GIỮ class .practice-* của CSS legacy (đã tinh chỉnh kỹ) để bám sát hình cũ.

import { useEffect, useRef, useState } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Mic, Play } from 'lucide-react';
import { apiFetch } from '@/lib/api';
import { hasHangul } from '@/lib/config';
import { useUiStore } from '@/store/ui';
import { usePractice } from '@/store/practice';
import { useSavedWords, syncBookmarkButtons } from '@/store/savedWords';
import { phonemeTip } from '@/lib/phonemeTips';
import { ipaStressString, toBritishWord } from '@/legacy/render';

// Cache word-info trong phiên (server cũng cache SQLite — đây chỉ đỡ round-trip).
const wordInfoCache = new Map<string, WordInfo | null>();
interface WordInfo {
  meaning?: string;
  definition_en?: string;
  example_en?: string;
}

// % chính xác: (ok + low-severity) / non-skipped — khớp ngưỡng isSignificant render.js.
function practicePct(phonemes: any[]): number | null {
  const scored = (phonemes || []).filter((p) => p.status !== 'skipped');
  if (!scored.length) return null;
  const pass = scored.filter((p) => p.status === 'ok' || p.severity === 'low').length;
  return Math.round((100 * pass) / scored.length);
}
function practiceIsBad(p: any): boolean {
  return p.status !== 'skipped' && !(p.status === 'ok' || p.severity === 'low');
}
function ringColor(pct: number): string {
  return pct >= 80 ? '#16a34a' : pct >= 50 ? '#f59e0b' : '#dc2626';
}

type Status = { text: string; kind: '' | 'err' | 'good' };

export default function PracticeDialog() {
  const open = usePractice((s) => s.open);
  const data = usePractice((s) => s.data);
  const setData = usePractice((s) => s.setData);
  const close = usePractice((s) => s.close);
  const accent = useUiStore((s) => s.accent);
  const savedHas = useSavedWords((s) => s.keys); // re-render khi cache đổi
  const swAdd = useSavedWords((s) => s.add);
  const swRemove = useSavedWords((s) => s.remove);

  const [status, setStatus] = useState<Status>({ text: '', kind: '' });
  const [recording, setRecording] = useState(false);
  const [grading, setGrading] = useState(false);
  const [wordInfo, setWordInfo] = useState<WordInfo | null | undefined>(undefined);
  const [replayUrl, setReplayUrl] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const replayAudioRef = useRef<HTMLAudioElement | null>(null);

  const word = data?.word || '';
  const isKo = hasHangul(word);
  const saved = !!word && savedHas.has(word.trim().toLowerCase());

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

  // Dọn stream + replay URL khi đóng.
  useEffect(() => {
    if (open) return;
    stopStream();
    setReplayUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  function stopStream() {
    if (recorderRef.current && recording) {
      try {
        recorderRef.current.stop();
      } catch {
        /* đã stop */
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setRecording(false);
  }

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

  async function toggleRecording() {
    if (grading) return;
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setStatus({ text: 'Không truy cập được micro — kiểm tra quyền trình duyệt.', kind: 'err' });
      return;
    }
    chunksRef.current = [];
    const rec = new MediaRecorder(streamRef.current);
    recorderRef.current = rec;
    rec.addEventListener('dataavailable', (e) => {
      if (e.data && e.data.size) chunksRef.current.push(e.data);
    });
    rec.addEventListener('stop', () => {
      const mime = rec.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      stopStream();
      setReplayUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(blob);
      });
      gradeAttempt(blob, mime);
    });
    rec.start();
    setRecording(true);
    setStatus({ text: 'Đang ghi âm… bấm lần nữa để dừng.', kind: '' });
  }

  async function gradeAttempt(blob: Blob, mime: string) {
    const d = usePractice.getState().data;
    if (!d) return;
    // Blob cụt (<1KB) = MediaRecorder nhả header rỗng → chặn khỏi tốn request.
    if (!blob || blob.size < 1024) {
      setStatus({ text: 'Ghi âm quá ngắn — bấm 🎙️, nói rõ từ, rồi mới bấm dừng.', kind: 'err' });
      return;
    }
    setGrading(true);
    setStatus({ text: 'Đang chấm…', kind: '' });
    try {
      const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm';
      const fd = new FormData();
      fd.append('audio', new File([blob], `practice-${d.word}.${ext}`, { type: mime }));
      fd.append('text', d.word);
      fd.append('mode', 'mock_test'); // ép bật phoneme analysis
      fd.append('no_ai', 'true'); // 1 từ không cần LLM
      fd.append('strict', 'true'); // chấm CHẶT (tắt leniency câu dài)
      fd.append('accent', accent);
      if (hasHangul(d.word)) fd.append('exam', 'topik'); // pipeline ko cho từ Hàn
      const res = await apiFetch('/grade', { method: 'POST', body: fd });
      if (!res.ok) {
        const raw = await res.text();
        let detail = `HTTP ${res.status}`;
        try {
          detail = JSON.parse(raw).detail || detail;
        } catch {
          /* giữ mã HTTP */
        }
        throw new Error(detail);
      }
      const result = await res.json();
      const ws = result?.phoneme?.score?.words || [];
      if (!ws.some((w: any) => (w.phonemes || []).length)) {
        throw new Error('Server không trả kết quả phoneme — thử lại.');
      }
      const wordSkipped = (w: any) => w.skip_reason || (w.phonemes || []).every((p: any) => p.status === 'skipped');
      if (ws.every(wordSkipped)) {
        const heard = (result.transcript || '').trim();
        const heardShort = heard.length > 60 ? `${heard.slice(0, 57)}…` : heard;
        setStatus({
          text: heardShort
            ? `Chưa nghe rõ — máy nghe thành “${heardShort}”. Hãy nói to, rõ và thử lại.`
            : 'Chưa nghe rõ — không thấy tiếng nói. Hãy nói to, rõ và thử lại.',
          kind: 'err',
        });
        return;
      }
      // Cụm nhiều từ: gộp phoneme tất cả các từ (gắn _w = chỉ số từ trong cụm).
      const merged =
        ws.length === 1
          ? ws[0].phonemes || []
          : ws.flatMap((w: any, i: number) => (w.phonemes || []).map((p: any) => ({ ...p, _w: i })));
      const pct = practicePct(merged) ?? 0;
      setData({ ...d, phonemes: merged, skip_reason: null });
      setStatus({
        text: pct >= 80 ? `Tuyệt vời — ${pct}%! 🎉` : `Được ${pct}% — nghe mẫu 🔊 rồi thử lại nhé.`,
        kind: pct >= 80 ? 'good' : '',
      });
      // Từ đã lưu → cập nhật điểm + snapshot phonemes mới (im lặng).
      if (useSavedWords.getState().has(d.word)) {
        swAdd({ word: d.word, last_score: pct / 100, phonemes: merged }).catch(() => {});
      }
    } catch (err: any) {
      setStatus({ text: `Lỗi chấm: ${err.message || err}`, kind: 'err' });
    } finally {
      setGrading(false);
    }
  }

  function playReplay() {
    if (!replayUrl) return;
    if (!replayAudioRef.current || replayAudioRef.current.src !== replayUrl) {
      replayAudioRef.current = new Audio(replayUrl);
    }
    replayAudioRef.current.currentTime = 0;
    replayAudioRef.current.play().catch(() => {});
  }

  if (!data) return null;

  // ── IPA + phoneme hiển thị (áp GB nếu chọn, trừ tiếng Hàn) ──
  const phonemes: any[] = data.phonemes || [];
  const disp = accent === 'gb' && !isKo ? toBritishWord({ phonemes }).phonemes : phonemes;
  const shown = (disp || []).filter((p: any) => !p._hidden);
  const ipaStr = (ipaStressString(disp) || data.ipa || '').trim();
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
          <div
            className="practice-ring"
            style={
              {
                ['--pct' as any]: pct ?? 0,
                ['--ring-color' as any]: pct != null ? ringColor(pct) : undefined,
              } as React.CSSProperties
            }
          >
            <div className="practice-ring__inner">{pct != null ? `${pct}%` : '–'}</div>
          </div>
        </div>

        {/* IPA */}
        <div className="practice-ipa">
          {ipaStr ? `/${ipaStr}/` : ''}
          <button type="button" className="tts-play" data-word={word} title="Nghe phát âm chuẩn">
            🔊
          </button>
        </div>

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
            onClick={toggleRecording}
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
