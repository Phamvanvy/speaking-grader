// Tab "Từ đã lưu" — port web/js/saved.js sang React + shadcn. Gộp 4 luồng render
// saved-word cũ (bảng, hàng gợi ý, popup, review-toast) về CHUNG 1 nguồn state
// (useSavedWords + usePractice) — trực tiếp trị "bug state" mà plan nhắm tới.

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Trash2, Volume2, Mic, Bell, BellOff, RotateCw, ChevronUp, ChevronDown, ChevronsUpDown, Star } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
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
function scorePct(w: SavedWord): string {
  if (w.last_score != null) return `${Math.round(w.last_score * 100)}%`;
  if (w.accuracy != null) return `${Math.round(w.accuracy * 100)}%`;
  return '—';
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
    <div id="mode-saved" className="flex flex-col gap-4">
      <Card className="p-5">
        <h2 className="mb-1 text-lg font-semibold">📚 Từ đã lưu</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          Các từ bạn đã đánh dấu ☆ khi chấm bài — bấm 🎙️ để luyện lại riêng từng từ.
        </p>

        <AddWordInline
          onAdd={async (word) => {
            await addWord({ word });
            syncBookmarkButtons(word);
          }}
          has={(w) => useSavedWords.getState().has(w)}
        />

        <div className="mb-3 mt-4">
          <Input
            placeholder="🔍 Tìm trong từ đã lưu…"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value.trim().toLowerCase());
              setPage(0);
            }}
            className="max-w-xs"
          />
        </div>

        {total === 0 ? (
          <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
            {filter
              ? `Không tìm thấy từ nào khớp “${filter}”.`
              : 'Chưa có từ nào. Gõ từ vào ô "Thêm từ" ở trên, hoặc khi xem kết quả chấm bấm ☆ trên bảng lỗi để lưu từ vào đây luyện tập.'}
          </div>
        ) : (
          <>
            <Table>
              <TableHeader>
                <TableRow>
                  <SortHead label="Từ" col="word" sort={sort} onSort={applySort} />
                  <TableHead>Phát âm</TableHead>
                  <SortHead label="Điểm" col="score" sort={sort} onSort={applySort} />
                  <SortHead label="Ngày lưu" col="date" sort={sort} onSort={applySort} />
                  <SortHead label="Nhắc" col="remind" sort={sort} onSort={applySort} />
                  <TableHead className="text-right">Luyện</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {slice.map((w) => {
                  const ipa = wordIpa(w);
                  const m = isMuted(w.word);
                  return (
                    <TableRow key={w.word}>
                      <TableCell className="font-medium">{w.word}</TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5">
                          <span className="font-mono text-xs text-muted-foreground">{ipa}</span>
                          <button type="button" className="tts-play opacity-70 hover:opacity-100" data-word={w.word} title="Nghe phát âm chuẩn">
                            <Volume2 className="h-4 w-4" />
                          </button>
                        </span>
                      </TableCell>
                      <TableCell title="Điểm luyện gần nhất">{scorePct(w)}</TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {w.saved_at ? new Date(w.saved_at).toLocaleDateString('vi-VN') : ''}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          title={m ? 'Đã tắt nhắc ôn — bấm để bật lại' : 'Đang nhắc ôn — bấm để tắt nhắc từ này'}
                          onClick={() => toggleMuted(w.word)}
                        >
                          {m ? <BellOff className="h-4 w-4 opacity-60" /> : <Bell className="h-4 w-4" />}
                        </Button>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          title="Luyện tập từ này"
                          onClick={() => openPractice({ word: w.word, ipa: w.ipa ?? null, accuracy: w.accuracy ?? null, phonemes: w.phonemes || [] })}
                        >
                          <Mic className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" title="Bỏ lưu" onClick={() => del(w.word)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>

            {/* Pager */}
            <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
              <label className="flex items-center gap-1.5">
                Hiện
                <select
                  className="rounded-md border border-input bg-background px-2 py-1 text-sm text-foreground"
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
                  <Button variant="outline" size="sm" disabled={curPage <= 0} onClick={() => setPage(0)} aria-label="Trang đầu">
                    «
                  </Button>
                  <Button variant="outline" size="sm" disabled={curPage <= 0} onClick={() => setPage(curPage - 1)} aria-label="Trang trước">
                    ‹
                  </Button>
                  <span className="px-1">
                    Trang {curPage + 1}/{pageCount}
                  </span>
                  <Button variant="outline" size="sm" disabled={curPage >= pageCount - 1} onClick={() => setPage(curPage + 1)} aria-label="Trang sau">
                    ›
                  </Button>
                  <Button variant="outline" size="sm" disabled={curPage >= pageCount - 1} onClick={() => setPage(pageCount - 1)} aria-label="Trang cuối">
                    »
                  </Button>
                </span>
              )}
              <span className="ml-auto">{total} từ</span>
            </div>
          </>
        )}
      </Card>

      <SuggestionsCard onPractice={openPractice} />
      <ReviewSettingsCard />
    </div>
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
    <form onSubmit={submit} className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Thêm từ… (ví dụ: bookstore / gain valuable insights)"
          maxLength={40}
          autoComplete="off"
          spellCheck={false}
          className="max-w-sm"
        />
        <Button type="submit" size="sm">
          <Star className="h-4 w-4" /> Lưu
        </Button>
      </div>
      {msg.text && <div className={`text-sm ${msg.err ? 'text-destructive' : 'text-muted-foreground'}`}>{msg.text}</div>}
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
    <Card className="p-5">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-lg font-semibold">🎯 Gợi ý luyện âm</h2>
        <Button variant="ghost" size="icon" title="Tải lại gợi ý" onClick={() => refetch()} disabled={isFetching}>
          <RotateCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {isFetching && !data ? (
        <div className="text-sm text-muted-foreground">Đang tải gợi ý… (lần đầu có thể mất vài giây — AI chọn từ cho từng âm)</div>
      ) : !data ? (
        <div className="text-sm text-muted-foreground">Không tải được gợi ý — thử lại.</div>
      ) : (
        <>
          <p className="mb-2 text-sm text-muted-foreground">
            {data.source === 'fallback'
              ? 'Chưa đủ dữ liệu chấm điểm — gợi ý theo các âm người Việt thường gặp khó. Chấm thêm bài để gợi ý bám sát bạn hơn.'
              : 'Các âm bạn hay sai (tính từ lịch sử chấm) — bấm chip để nghe âm mẫu:'}
          </p>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {(data.weak_phonemes || []).map((w, i) => {
              const info = phonemeTip(w.symbol);
              const pct = w.error_rate != null ? ` · sai ${Math.round(w.error_rate * 100)}%` : '';
              const cls = 'practice-chip bad';
              return info && info.example ? (
                <button key={i} type="button" className={`${cls} tts-play`} data-word={info.example} title={`${info.tip} — nghe trong từ “${info.example}”`}>
                  /{w.symbol}/{pct}
                </button>
              ) : (
                <span key={i} className={cls} title={info ? info.tip : ''}>
                  /{w.symbol}/{pct}
                </span>
              );
            })}
          </div>

          <div className="flex flex-col gap-2">
            {(data.suggestions || []).length ? (
              data.suggestions!.map((s, i) => {
                const saved = savedHas.has(s.word.trim().toLowerCase());
                return (
                  <div key={i} className="rounded-md border p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium">{s.word}</span>
                      <button
                        type="button"
                        className={`word-bookmark${saved ? ' saved' : ''}`}
                        title={saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}
                        onClick={() => toggleStar(s)}
                      >
                        {saved ? '★' : '☆'}
                      </button>
                      {s.ipa && <span className="font-mono text-xs text-muted-foreground">/{s.ipa}/</span>}
                      <button type="button" className="tts-play opacity-70 hover:opacity-100" data-word={s.word} title="Nghe phát âm chuẩn">
                        <Volume2 className="h-4 w-4" />
                      </button>
                      {(s.target_phonemes || []).map((p, j) => (
                        <span key={j} className="practice-chip bad">
                          /{p}/
                        </span>
                      ))}
                    </div>
                    {s.reason && <div className="mt-1 text-sm text-muted-foreground">{s.reason}</div>}
                    <Button
                      variant="secondary"
                      size="sm"
                      className="mt-2"
                      onClick={() => onPractice({ word: s.word, ipa: s.ipa ?? null, accuracy: null, phonemes: [] })}
                    >
                      <Mic className="h-4 w-4" /> Luyện tập
                    </Button>
                  </div>
                );
              })
            ) : (
              <div className="text-sm text-muted-foreground">Chưa có gợi ý — thử lại sau.</div>
            )}
          </div>
        </>
      )}
    </Card>
  );
}

// ── Cài đặt nhắc ôn (review-toast) ───────────────────────────────────────
function ReviewSettingsCard() {
  const settings = useReviewToast((s) => s.settings);
  const setSettings = useReviewToast((s) => s.setSettings);

  const numField = (label: string, key: keyof ReviewSettings, min: number, max: number, unit: string) => (
    <label className="flex items-center gap-2 text-sm">
      <span className="w-40 text-muted-foreground">{label}</span>
      <Input
        type="number"
        min={min}
        max={max}
        value={String(settings[key])}
        onChange={(e) => setSettings({ [key]: parseInt(e.target.value, 10) } as Partial<ReviewSettings>)}
        className="w-20"
      />
      <span className="text-xs text-muted-foreground">{unit}</span>
    </label>
  );

  return (
    <Card className="p-5">
      <h2 className="mb-3 text-lg font-semibold">🔔 Nhắc ôn từ đã lưu</h2>
      <div className="flex flex-col gap-3">
        <label className="flex items-center gap-3 text-sm">
          <Switch checked={settings.enabled} onCheckedChange={(v) => setSettings({ enabled: v })} />
          <span>Bật nhắc ôn định kỳ (hiện toast vài từ để ôn lại)</span>
        </label>
        {settings.enabled && (
          <>
            {numField('Số từ mỗi lần', 'count', 1, 10, 'từ')}
            {numField('Tự ẩn sau', 'hideSec', 5, 120, 'giây')}
            {numField('Khoảng cách nhắc', 'intervalMin', 1, 120, 'phút')}
            <label className="flex items-center gap-3 text-sm">
              <Switch checked={settings.noMobile} onCheckedChange={(v) => setSettings({ noMobile: v })} />
              <span>Không nhắc trên điện thoại (màn hình nhỏ)</span>
            </label>
          </>
        )}
      </div>
    </Card>
  );
}
