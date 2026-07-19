// Tab "Từ đã lưu" — port web/js/saved.js sang React + shadcn. Gộp 4 luồng render
// saved-word cũ (bảng, hàng gợi ý, popup, review-toast) về CHUNG 1 nguồn state
// (useSavedWords + usePractice) — trực tiếp trị "bug state" mà plan nhắm tới.

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Trash2, Volume2, Mic, Bell, BellOff, RotateCw, ChevronUp, ChevronDown, ChevronsUpDown, Star, Search, Target } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { apiGet } from '@/lib/api';
import { getUserId } from '@/lib/identity';
import { useSavedWords, syncBookmarkButtons, type SavedWord } from '@/store/savedWords';
import { usePractice } from '@/store/practice';
import { useReviewToast, type ReviewSettings } from './reviewToast';
import { ipaStressString } from '@/legacy/render';
import { phonemeTip } from '@/lib/phonemeTips';

// ── Sort ─────────────────────────────────────────────────────────────────
type SortKey = 'remind' | 'date' | 'word' | 'score';
type SortDir = 'asc' | 'desc';
const SORT_KEY_LS = 'speaking-grader-saved-sort';
const PAGESIZE_LS = 'speaking-grader-saved-pagesize';
const PAGE_OPTIONS = [10, 20, 50, 0]; // 0 = tất cả
const SORT_DEFAULT_DIR: Record<SortKey, SortDir> = { remind: 'desc', date: 'desc', score: 'desc', word: 'asc' };

function loadSort(): { key: SortKey; dir: SortDir } {
  try {
    const s = JSON.parse(localStorage.getItem(SORT_KEY_LS) || '');
    if (s && ['remind', 'date', 'word', 'score'].includes(s.key) && (s.dir === 'asc' || s.dir === 'desc')) return s;
  } catch {
    /* mặc định */
  }
  return { key: 'remind', dir: 'desc' };
}
function loadPageSize(): number {
  const v = parseInt(localStorage.getItem(PAGESIZE_LS) || '', 10);
  return PAGE_OPTIONS.includes(v) ? v : 10;
}

function wordIpa(w: SavedWord): string {
  const s = (w.phonemes || []).length ? ipaStressString(w.phonemes) : w.ipa || '';
  return s ? `/${s}/` : '';
}
function scoreVal(w: SavedWord): number | null {
  if (w.last_score != null) return w.last_score;
  if (w.accuracy != null) return w.accuracy;
  return null;
}
function scorePct(w: SavedWord): string {
  const v = scoreVal(w);
  return v == null ? '—' : `${Math.round(v * 100)}%`;
}
/** Màu điểm luyện gần nhất: xanh ≥80, hổ phách ≥60, đỏ dưới đó (khớp thang chấm). */
function scoreTone(v: number | null): string {
  if (v == null) return 'text-muted-foreground';
  if (v >= 0.8) return 'text-emerald-600 dark:text-emerald-400';
  if (v >= 0.6) return 'text-amber-600 dark:text-amber-400';
  return 'text-rose-600 dark:text-rose-400';
}

const ADD_WORD_RE = /^[A-Za-z][A-Za-z' -]{0,39}$/;

export default function SavedTab() {
  const words = useSavedWords((s) => s.words);
  const refresh = useSavedWords((s) => s.refresh);
  const removeWord = useSavedWords((s) => s.remove);
  const addWord = useSavedWords((s) => s.add);
  const muted = useReviewToast((s) => s.muted);
  const toggleMuted = useReviewToast((s) => s.toggleMuted);
  const openPractice = usePractice((s) => s.openPractice);

  const [sort, setSort] = useState(loadSort);
  const [pageSize, setPageSize] = useState(loadPageSize);
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState('');
  const [remindOpen, setRemindOpen] = useState(false);
  const remindOn = useReviewToast((s) => s.settings.enabled);

  // Nạp lại mỗi lần mở tab (giữ đúng thứ tự server "mới lưu trước").
  useEffect(() => {
    refresh().catch(() => {});
  }, [refresh]);

  const isMuted = (w: string) => muted.has((w || '').trim().toLowerCase());

  // Sort (port _sortSavedItems) — tie-break mới→cũ rồi A→Z (tất định).
  const sorted = useMemo(() => {
    const dateVal = (w: SavedWord) => (w.saved_at ? Date.parse(w.saved_at) : 0);
    const scoreVal = (w: SavedWord) => (w.last_score != null ? w.last_score : w.accuracy != null ? w.accuracy : -1);
    const { key, dir } = sort;
    return [...words].sort((a, b) => {
      if (key === 'remind') {
        const ma = isMuted(a.word),
          mb = isMuted(b.word);
        if (ma !== mb) return (ma ? 1 : -1) * (dir === 'asc' ? -1 : 1);
        return dateVal(b) - dateVal(a);
      }
      let base: number;
      if (key === 'word') base = a.word.localeCompare(b.word);
      else if (key === 'score') base = scoreVal(a) - scoreVal(b);
      else base = dateVal(a) - dateVal(b);
      if (base === 0) base = dateVal(b) - dateVal(a);
      if (base === 0) base = a.word.localeCompare(b.word);
      return dir === 'asc' ? base : -base;
    });
  }, [words, sort, muted]);

  const filtered = useMemo(
    () => (filter ? sorted.filter((w) => (w.word || '').toLowerCase().includes(filter)) : sorted),
    [sorted, filter],
  );

  const total = filtered.length;
  const pageCount = pageSize ? Math.ceil(total / pageSize) : 1;
  const curPage = Math.min(Math.max(page, 0), Math.max(pageCount - 1, 0));
  const slice = pageSize ? filtered.slice(curPage * pageSize, curPage * pageSize + pageSize) : filtered;

  function applySort(key: SortKey) {
    const next =
      sort.key === key ? { key, dir: (sort.dir === 'asc' ? 'desc' : 'asc') as SortDir } : { key, dir: SORT_DEFAULT_DIR[key] };
    setSort(next);
    localStorage.setItem(SORT_KEY_LS, JSON.stringify(next));
    setPage(0);
  }
  function changePageSize(v: number) {
    setPageSize(v);
    localStorage.setItem(PAGESIZE_LS, String(v));
    setPage(0);
  }

  async function del(w: string) {
    if (!confirm(`Bỏ lưu từ "${w}"?`)) return;
    try {
      await removeWord(w);
      syncBookmarkButtons(w);
    } catch (err: any) {
      alert(`Lỗi xoá: ${err.message || err}`);
    }
  }

  return (
    <div id="mode-saved" className="flex flex-col gap-5">
      <Card className="overflow-hidden">
        <SectionHead
          icon="📚"
          title="Từ đã lưu"
          desc="Các từ bạn đã đánh dấu ☆ khi chấm bài — bấm 🎙️ để luyện lại riêng từng từ."
          right={
            <div className="flex items-center gap-2">
              {words.length > 0 && <Badge variant="secondary">{words.length} từ</Badge>}
              {/* Nhắc ôn là 1 tuỳ chọn CỦA danh sách này (không còn card riêng ở dưới). */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setRemindOpen(true)}
                title="Cài đặt nhắc ôn từ đã lưu"
                className={
                  remindOn
                    ? 'gap-1.5 border-primary/60 bg-primary/10 font-semibold text-primary shadow-sm ring-1 ring-primary/25 hover:bg-primary/15 hover:text-primary'
                    : 'gap-1.5 text-muted-foreground'
                }
              >
                {remindOn ? (
                  <span className="relative flex h-4 w-4 items-center justify-center">
                    <Bell className="h-4 w-4" />
                    <span className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-primary" />
                  </span>
                ) : (
                  <BellOff className="h-4 w-4 opacity-60" />
                )}
                Nhắc ôn
                {remindOn ? (
                  <span className="rounded-full bg-primary px-1.5 py-0.5 text-[0.65rem] font-bold uppercase leading-none tracking-wide text-primary-foreground">
                    Bật
                  </span>
                ) : (
                  <span className="text-xs font-normal text-muted-foreground">Tắt</span>
                )}
              </Button>
            </div>
          }
        />

        <div className="p-5">
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
            <AddWordInline
              onAdd={async (word) => {
                await addWord({ word });
                syncBookmarkButtons(word);
              }}
              has={(w) => useSavedWords.getState().has(w)}
            />
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Tìm trong từ đã lưu…"
                value={filter}
                onChange={(e) => {
                  setFilter(e.target.value.trim().toLowerCase());
                  setPage(0);
                }}
                className="w-56 pl-8"
              />
            </div>
          </div>

          {total === 0 ? (
            <div className="rounded-lg border border-dashed px-6 py-10 text-center text-sm text-muted-foreground">
              {filter
                ? `Không tìm thấy từ nào khớp “${filter}”.`
                : 'Chưa có từ nào. Gõ từ vào ô "Thêm từ" ở trên, hoặc khi xem kết quả chấm bấm ☆ trên bảng lỗi để lưu từ vào đây luyện tập.'}
            </div>
          ) : (
            <>
              <div className="overflow-hidden rounded-xl border shadow-sm">
                <Table>
                  <TableHeader>
                    <TableRow className="border-b-2 bg-muted/60 hover:bg-muted/60 [&>th]:h-10 [&>th]:text-[0.78rem] [&>th]:font-semibold [&>th]:uppercase [&>th]:tracking-wide">
                      <SortHead label="Từ" col="word" sort={sort} onSort={applySort} />
                      <TableHead>Phát âm</TableHead>
                      <SortHead label="Điểm" col="score" sort={sort} onSort={applySort} className="w-20" />
                      <SortHead label="Ngày lưu" col="date" sort={sort} onSort={applySort} className="w-28" />
                      <SortHead label="Nhắc" col="remind" sort={sort} onSort={applySort} className="w-16" />
                      <TableHead className="w-24 text-right">Luyện</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {slice.map((w) => {
                      const ipa = wordIpa(w);
                      const m = isMuted(w.word);
                      const sv = scoreVal(w);
                      return (
                        <TableRow key={w.word} className="group odd:bg-muted/20">
                          {/* Bấm vào từ = mở luyện tập (hàng nào cũng có nút 🎙️, nhưng từ là target to nhất). */}
                          <TableCell className="py-3">
                            <button
                              type="button"
                              className="text-[0.95rem] font-semibold text-foreground hover:text-primary hover:underline"
                              title="Bấm để luyện tập từ này"
                              onClick={() =>
                                openPractice({ word: w.word, ipa: w.ipa ?? null, accuracy: w.accuracy ?? null, phonemes: w.phonemes || [] })
                              }
                            >
                              {w.word}
                            </button>
                          </TableCell>
                          <TableCell className="py-3">
                            <span className="flex items-center gap-1.5">
                              <span className="rounded-md bg-muted/70 px-1.5 py-0.5 font-mono text-xs text-muted-foreground">{ipa}</span>
                              <IconBtn className="tts-play h-7 w-7" data-word={w.word} title="Nghe phát âm chuẩn">
                                <Volume2 className="h-4 w-4" />
                              </IconBtn>
                            </span>
                          </TableCell>
                          <TableCell className="py-3" title="Điểm luyện gần nhất">
                            <ScoreCell value={sv} label={scorePct(w)} />
                          </TableCell>
                          <TableCell className="py-3 text-xs tabular-nums text-muted-foreground">
                            {w.saved_at ? new Date(w.saved_at).toLocaleDateString('vi-VN') : ''}
                          </TableCell>
                          <TableCell className="py-3">
                            <IconBtn
                              title={m ? 'Đã tắt nhắc ôn — bấm để bật lại' : 'Đang nhắc ôn — bấm để tắt nhắc từ này'}
                              onClick={() => toggleMuted(w.word)}
                            >
                              {m ? <BellOff className="h-4 w-4 opacity-50" /> : <Bell className="h-4 w-4 text-primary" />}
                            </IconBtn>
                          </TableCell>
                          <TableCell className="py-3">
                            {/* Hiện mờ, rõ khi hover hàng — bảng đỡ rối vì 2 nút/hàng. */}
                            <div className="flex items-center justify-end gap-0.5 opacity-60 transition-opacity group-hover:opacity-100">
                              <IconBtn
                                title="Luyện tập từ này"
                                onClick={() =>
                                  openPractice({ word: w.word, ipa: w.ipa ?? null, accuracy: w.accuracy ?? null, phonemes: w.phonemes || [] })
                                }
                              >
                                <Mic className="h-4 w-4" />
                              </IconBtn>
                              <IconBtn title="Bỏ lưu" className="hover:text-destructive" onClick={() => del(w.word)}>
                                <Trash2 className="h-4 w-4" />
                              </IconBtn>
                            </div>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

              {/* Pager */}
              <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground">
                <label className="mb-0 flex items-center gap-1.5 whitespace-nowrap font-normal text-muted-foreground">
                  Hiện
                  {/* `!w-auto`: base.css legacy có `select { width:100% }` (element, thắng 1 class utility). */}
                  <select
                    className="!w-auto rounded-md border border-input bg-background px-2 py-1 text-sm text-foreground"
                    value={pageSize}
                    onChange={(e) => changePageSize(parseInt(e.target.value, 10))}
                  >
                    {PAGE_OPTIONS.map((o) => (
                      <option key={o} value={o}>
                        {o === 0 ? 'Tất cả' : o}
                      </option>
                    ))}
                  </select>
                  từ mới nhất
                </label>
                {pageCount > 1 && (
                  <span className="flex items-center gap-1">
                    <PagerBtn disabled={curPage <= 0} onClick={() => setPage(0)} label="Trang đầu" glyph="«" />
                    <PagerBtn disabled={curPage <= 0} onClick={() => setPage(curPage - 1)} label="Trang trước" glyph="‹" />
                    <span className="px-1.5 tabular-nums">
                      Trang {curPage + 1}/{pageCount}
                    </span>
                    <PagerBtn disabled={curPage >= pageCount - 1} onClick={() => setPage(curPage + 1)} label="Trang sau" glyph="›" />
                    <PagerBtn disabled={curPage >= pageCount - 1} onClick={() => setPage(pageCount - 1)} label="Trang cuối" glyph="»" />
                  </span>
                )}
                <span className="ml-auto tabular-nums">
                  {total} / {words.length} từ
                </span>
              </div>
            </>
          )}
        </div>
      </Card>

      <SuggestionsCard onPractice={openPractice} />
      <ReviewSettingsDialog open={remindOpen} onOpenChange={setRemindOpen} />
    </div>
  );
}

/** Điểm luyện gần nhất dạng pill — dễ quét mắt hơn text trần. */
function ScoreCell({ value, label }: { value: number | null; label: string }) {
  if (value == null) return <span className="text-sm text-muted-foreground">—</span>;
  const tone =
    value >= 0.8
      ? 'bg-emerald-50 dark:bg-emerald-950/40'
      : value >= 0.6
        ? 'bg-amber-50 dark:bg-amber-950/40'
        : 'bg-rose-50 dark:bg-rose-950/40';
  return (
    <span className={`inline-block rounded-md px-2 py-0.5 text-sm font-semibold tabular-nums ${tone} ${scoreTone(value)}`}>
      {label}
    </span>
  );
}

// ── Khối dùng chung cho 3 card của tab ───────────────────────────────────
function SectionHead({
  icon,
  title,
  desc,
  right,
}: {
  icon: string;
  title: string;
  desc?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3 border-b bg-muted/30 px-5 py-4">
      <div>
        <h2 className="flex items-center gap-2 text-base font-semibold">
          <span aria-hidden>{icon}</span>
          {title}
        </h2>
        {desc && <p className="mt-1 text-sm text-muted-foreground">{desc}</p>}
      </div>
      {right && <div className="shrink-0">{right}</div>}
    </div>
  );
}

/** Nút icon 32px — dùng <button> trần (đã reset UA ở tailwind.css) cho gọn hơn Button ghost. */
function IconBtn({
  className,
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={`inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${className || ''}`}
      {...props}
    >
      {children}
    </button>
  );
}

function PagerBtn({
  disabled,
  onClick,
  label,
  glyph,
}: {
  disabled: boolean;
  onClick: () => void;
  label: string;
  glyph: string;
}) {
  return (
    <Button variant="outline" size="icon" className="h-8 w-8 text-base" disabled={disabled} onClick={onClick} aria-label={label} title={label}>
      {glyph}
    </Button>
  );
}

// ── Header cột sắp xếp ───────────────────────────────────────────────────
function SortHead({
  label,
  col,
  sort,
  onSort,
  className,
}: {
  label: string;
  col: SortKey;
  sort: { key: SortKey; dir: SortDir };
  onSort: (k: SortKey) => void;
  className?: string;
}) {
  const active = sort.key === col;
  const Icon = !active ? ChevronsUpDown : sort.dir === 'asc' ? ChevronUp : ChevronDown;
  return (
    <TableHead
      className={`cursor-pointer select-none hover:text-foreground ${active ? 'text-foreground' : ''} ${className || ''}`}
      onClick={() => onSort(col)}
      title={`Bấm để sắp xếp theo ${label}`}
      aria-sort={active ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <Icon className={`h-3.5 w-3.5 ${active ? '' : 'opacity-40'}`} />
      </span>
    </TableHead>
  );
}

// ── Form thêm từ (1 ô, inline trên bảng) ─────────────────────────────────
function AddWordInline({ onAdd, has }: { onAdd: (word: string) => Promise<void>; has: (w: string) => boolean }) {
  const [value, setValue] = useState('');
  const [msg, setMsg] = useState<{ text: string; err: boolean }>({ text: '', err: false });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const word = value.trim().toLowerCase().replace(/\s+/g, ' ');
    if (!ADD_WORD_RE.test(word) || word.split(' ').length > 4) {
      setMsg({ text: 'Từ/cụm chỉ gồm chữ cái tiếng Anh (nháy đơn, gạch nối; cụm tối đa 4 từ).', err: true });
      return;
    }
    if (has(word)) {
      setMsg({ text: `"${word}" đã có trong danh sách.`, err: false });
      return;
    }
    try {
      await onAdd(word);
      setValue('');
      setMsg({ text: `Đã lưu "${word}".`, err: false });
    } catch (err: any) {
      setMsg({ text: `Lỗi lưu từ: ${err.message || err}`, err: true });
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Thêm từ… (ví dụ: bookstore / gain valuable insights)"
          maxLength={40}
          autoComplete="off"
          spellCheck={false}
          className="w-[22rem] max-w-full"
        />
        <Button type="submit">
          <Star className="h-4 w-4" /> Lưu
        </Button>
      </div>
      {msg.text && <div className={`text-xs ${msg.err ? 'text-destructive' : 'text-muted-foreground'}`}>{msg.text}</div>}
    </form>
  );
}

// ── Gợi ý luyện âm (API /words/suggestions, TanStack Query) ───────────────
interface WeakPhoneme {
  symbol: string;
  error_rate?: number;
}
interface Suggestion {
  word: string;
  ipa?: string | null;
  target_phonemes?: string[];
  reason?: string;
}
interface SuggestResponse {
  source?: string;
  weak_phonemes?: WeakPhoneme[];
  suggestions?: Suggestion[];
}

function SuggestionsCard({ onPractice }: { onPractice: (d: any) => void }) {
  const savedHas = useSavedWords((s) => s.keys);
  const swAdd = useSavedWords((s) => s.add);
  const swRemove = useSavedWords((s) => s.remove);

  const { data, isFetching, refetch } = useQuery<SuggestResponse>({
    queryKey: ['wordSuggestions', getUserId()],
    queryFn: () => apiGet(`/words/suggestions?user_id=${encodeURIComponent(getUserId())}`),
    staleTime: Infinity, // legacy: fetch 1 lần/phiên; nút ↻ ép mới
  });

  async function toggleStar(s: Suggestion) {
    try {
      if (useSavedWords.getState().has(s.word)) await swRemove(s.word);
      else await swAdd({ word: s.word, ipa: s.ipa, phonemes: [], accuracy: null });
      syncBookmarkButtons(s.word);
    } catch (err: any) {
      alert(`Lỗi lưu từ: ${err.message || err}`);
    }
  }

  return (
    <Card className="overflow-hidden">
      <SectionHead
        icon="🎯"
        title="Gợi ý luyện âm"
        desc="Từ được chọn theo các âm bạn hay sai — bấm 🎙️ để luyện, ☆ để lưu vào danh sách trên."
        right={
          <IconBtn title="Tải lại gợi ý" onClick={() => refetch()} disabled={isFetching}>
            <RotateCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
          </IconBtn>
        }
      />

      <div className="p-5">
        {isFetching && !data ? (
          <div className="text-sm text-muted-foreground">Đang tải gợi ý… (lần đầu có thể mất vài giây — AI chọn từ cho từng âm)</div>
        ) : !data ? (
          <div className="text-sm text-muted-foreground">Không tải được gợi ý — thử lại.</div>
        ) : (
          <>
            {/* Khối "âm yếu" tách nền riêng: đây là CHẨN ĐOÁN, khác với danh sách từ ở dưới. */}
            <div className="mb-5 rounded-lg border bg-muted/30 p-3.5">
              <p className="mb-2 flex items-center gap-1.5 text-sm font-medium">
                <Target className="h-4 w-4 text-primary" />
                Âm bạn hay sai
              </p>
              <p className="mb-2.5 text-sm text-muted-foreground">
                {data.source === 'fallback'
                  ? 'Chưa đủ dữ liệu chấm điểm — gợi ý theo các âm người Việt thường gặp khó. Chấm thêm bài để gợi ý bám sát bạn hơn.'
                  : 'Tính từ lịch sử chấm — bấm chip để nghe âm mẫu trong một từ.'}
              </p>
              <div className="flex flex-wrap gap-1.5">
                {(data.weak_phonemes || []).map((w, i) => {
                  const info = phonemeTip(w.symbol);
                  const pct = w.error_rate != null ? ` · sai ${Math.round(w.error_rate * 100)}%` : '';
                  const cls = 'practice-chip bad';
                  return info && info.example ? (
                    <button
                      key={i}
                      type="button"
                      className={`${cls} tts-play`}
                      data-word={info.example}
                      title={`${info.tip} — nghe trong từ “${info.example}”`}
                    >
                      /{w.symbol}/{pct}
                    </button>
                  ) : (
                    <span key={i} className={cls} title={info ? info.tip : ''}>
                      /{w.symbol}/{pct}
                    </span>
                  );
                })}
              </div>
            </div>

            <div className="grid gap-2.5 sm:grid-cols-2">
              {(data.suggestions || []).length ? (
                data.suggestions!.map((s, i) => {
                  const saved = savedHas.has(s.word.trim().toLowerCase());
                  return (
                    // Cả thẻ = target mở luyện tập; nút ☆ / 🔊 stopPropagation để không bị nuốt.
                    <div
                      key={i}
                      role="button"
                      tabIndex={0}
                      onClick={() => onPractice({ word: s.word, ipa: s.ipa ?? null, accuracy: null, phonemes: [] })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          onPractice({ word: s.word, ipa: s.ipa ?? null, accuracy: null, phonemes: [] });
                        }
                      }}
                      title="Bấm để luyện từ này"
                      className="group flex cursor-pointer flex-col rounded-xl border bg-muted/25 p-3.5 shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/50 hover:bg-card hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <div className="flex items-start gap-2">
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                            <span className="text-[0.95rem] font-semibold">{s.word}</span>
                            {(s.target_phonemes || []).map((p, j) => (
                              <span key={j} className="practice-chip bad">
                                /{p}/
                              </span>
                            ))}
                          </div>
                          {s.ipa && <div className="mt-0.5 font-mono text-xs text-muted-foreground">/{s.ipa}/</div>}
                        </div>
                        <div className="flex shrink-0 items-center gap-0.5">
                          <IconBtn
                            className="tts-play h-8 w-8"
                            data-word={s.word}
                            title="Nghe phát âm chuẩn"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <Volume2 className="h-4 w-4" />
                          </IconBtn>
                          <button
                            type="button"
                            className={`word-bookmark${saved ? ' saved' : ''}`}
                            title={saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleStar(s);
                            }}
                          >
                            {saved ? '★' : '☆'}
                          </button>
                        </div>
                      </div>
                      {s.reason && <div className="mt-2 text-sm leading-snug text-muted-foreground">{s.reason}</div>}
                      <div className="mt-3 flex items-center gap-1.5 text-sm font-medium text-primary opacity-70 transition-opacity group-hover:opacity-100">
                        <Mic className="h-4 w-4" /> Luyện tập
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="text-sm text-muted-foreground">Chưa có gợi ý — thử lại sau.</div>
              )}
            </div>
          </>
        )}
      </div>
    </Card>
  );
}

// ── Cài đặt nhắc ôn (review-toast) — popup mở từ header card "Từ đã lưu" ──
function ReviewSettingsDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const settings = useReviewToast((s) => s.settings);
  const setSettings = useReviewToast((s) => s.setSettings);

  // `!w-20`: base.css có `input[type="number"] { width:100% }` — selector attribute
  // (0,1,1) thắng utility 1 class (0,1,0), nên phải ép important tại chỗ.
  const numField = (label: string, key: keyof ReviewSettings, min: number, max: number, unit: string) => (
    <label className="mb-0 flex items-center gap-2 text-sm font-normal">
      <span className="w-44 text-muted-foreground">{label}</span>
      <Input
        type="number"
        min={min}
        max={max}
        value={String(settings[key])}
        onChange={(e) => setSettings({ [key]: parseInt(e.target.value, 10) } as Partial<ReviewSettings>)}
        className="!w-20 tabular-nums"
      />
      <span className="text-xs text-muted-foreground">{unit}</span>
    </label>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Bell className="h-4 w-4 text-primary" /> Nhắc ôn từ đã lưu
          </DialogTitle>
          <DialogDescription>Thỉnh thoảng hiện vài từ ở góc màn hình (dưới nút đổi giao diện) để bạn ôn lại.</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3">
          <label className="mb-0 flex items-center gap-3 rounded-lg border bg-muted/30 px-3 py-2.5 text-sm font-normal">
            <Switch checked={settings.enabled} onCheckedChange={(v) => setSettings({ enabled: v })} />
            <span className="font-medium">Bật nhắc ôn định kỳ</span>
          </label>
          {settings.enabled && (
            <div className="flex flex-col gap-3 border-l-2 border-muted pl-4">
              {numField('Số từ mỗi lần', 'count', 1, 10, 'từ')}
              {numField('Tự ẩn sau', 'hideSec', 5, 120, 'giây')}
              {numField('Khoảng cách nhắc', 'intervalMin', 1, 120, 'phút')}
              <label className="mb-0 flex items-center gap-3 text-sm font-normal">
                <Switch checked={settings.noMobile} onCheckedChange={(v) => setSettings({ noMobile: v })} />
                <span>Không nhắc trên điện thoại (màn hình nhỏ)</span>
              </label>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
