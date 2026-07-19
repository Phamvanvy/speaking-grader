// Toast "Ôn lại từ đã lưu" — port web/js/review-toast.js sang sonner + Zustand.
// Định kỳ hiện ~N từ đã lưu (ưu tiên yếu / lâu chưa ôn). Client-side hoàn toàn: đọc
// cache useSavedWords, mở popup luyện qua usePractice, phát mẫu qua nút .tts-play
// (delegated playback). sonner lo animate + hover-pause + vị trí (bỏ CSS .review-toast
// tự viết + logic hover/hide thủ công của legacy).

import { useState } from 'react';
import { create } from 'zustand';
import { toast } from '@/components/ui/sonner';
import { Volume2, BellOff } from 'lucide-react';
import { apiFetch } from '@/lib/api';
import { getUserId } from '@/lib/identity';
import { useSavedWords, type SavedWord } from '@/store/savedWords';
import { usePractice } from '@/store/practice';
import { ipaStressString } from '@/legacy/render';

// ── Settings + mute (localStorage; store reactive cho SavedTab) ───────────
const REVIEW_TOAST_KEY = 'speaking-grader-review-toast';
const REVIEW_MUTED_KEY = 'speaking-grader-review-muted';
const REVIEW_SETTING_SERVER_KEY = 'review_toast';

const RT_FIRST_DELAY_MS = 1200;
const RT_LOAD_RETRY_MS = 800;
const RT_BUSY_RETRY_MS = 60000;

export interface ReviewSettings {
  enabled: boolean;
  count: number;
  hideSec: number;
  intervalMin: number;
  noMobile: boolean;
}

const clamp = (v: any, lo: number, hi: number, dflt: number) => {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? Math.min(Math.max(n, lo), hi) : dflt;
};

function readSettings(): ReviewSettings {
  let s: any = {};
  try {
    s = JSON.parse(localStorage.getItem(REVIEW_TOAST_KEY) || '{}') || {};
  } catch {
    s = {};
  }
  return {
    enabled: s.enabled !== false, // default ON
    count: clamp(s.count, 1, 10, 5),
    hideSec: clamp(s.hideSec, 5, 120, 15),
    intervalMin: clamp(s.intervalMin, 1, 120, 10),
    noMobile: s.noMobile === true,
  };
}

function readMuted(): Set<string> {
  try {
    return new Set<string>(JSON.parse(localStorage.getItem(REVIEW_MUTED_KEY) || '[]'));
  } catch {
    return new Set<string>();
  }
}

const mkey = (w: string) => (w || '').trim().toLowerCase();

interface ReviewToastState {
  settings: ReviewSettings;
  muted: Set<string>;
  isMuted: (word: string) => boolean;
  setMuted: (word: string, muted: boolean) => void;
  toggleMuted: (word: string) => boolean;
  setSettings: (patch: Partial<ReviewSettings>) => void;
  /** Nạp cài đặt từ server (bản ghi server thắng), gọi lúc mở app. */
  loadFromServer: () => Promise<void>;
}

export const useReviewToast = create<ReviewToastState>((set, get) => ({
  settings: readSettings(),
  muted: readMuted(),

  isMuted: (word) => get().muted.has(mkey(word)),

  setMuted: (word, muted) => {
    const s = new Set(get().muted);
    if (muted) s.add(mkey(word));
    else s.delete(mkey(word));
    localStorage.setItem(REVIEW_MUTED_KEY, JSON.stringify([...s]));
    set({ muted: s });
  },

  toggleMuted: (word) => {
    const next = !get().isMuted(word);
    get().setMuted(word, next);
    return next;
  },

  setSettings: (patch) => {
    const next = { ...get().settings, ...patch };
    // clamp lại qua readSettings sau khi ghi (nhập ngoài range tự sửa).
    localStorage.setItem(REVIEW_TOAST_KEY, JSON.stringify(next));
    const clamped = readSettings();
    set({ settings: clamped });
    reArm();
    saveSettingsToServer();
  },

  async loadFromServer() {
    try {
      const res = await apiFetch(
        `/settings?key=${REVIEW_SETTING_SERVER_KEY}&user_id=${encodeURIComponent(getUserId())}`,
      );
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.value) {
        localStorage.setItem(REVIEW_TOAST_KEY, data.value); // server thắng
        set({ settings: readSettings() });
        reconcile(); // KHÔNG re-arm full — giữ lần hiện đầu ~1.2s
      }
    } catch {
      /* offline/khách → dùng localStorage */
    }
  },
}));

function saveSettingsToServer() {
  const fd = new FormData();
  fd.append('user_id', getUserId());
  fd.append('key', REVIEW_SETTING_SERVER_KEY);
  fd.append('value', localStorage.getItem(REVIEW_TOAST_KEY) || '{}');
  // fire-and-forget; apiBase() để chắc chắn cùng origin backend.
  apiFetch('/settings', { method: 'POST', body: fd }).catch(() => {});
}

// ── Picker: chọn N từ ưu tiên yếu / lâu chưa ôn (mirror legacy) ───────────
let lastShown = new Set<string>();

function pickReviewWords(n: number): SavedWord[] {
  const sw = useSavedWords.getState();
  if (!sw.loaded) return [];
  const muted = useReviewToast.getState().muted;
  let pool = sw.words.filter((w) => !muted.has(mkey(w.word)));
  if (!pool.length) return [];
  const fresh = pool.filter((w) => !lastShown.has(w.word));
  if (fresh.length >= n) pool = fresh;

  const now = Date.now();
  const scored = pool.map((w) => {
    const s = w.last_score != null ? w.last_score : w.accuracy != null ? w.accuracy : 0;
    const ref = w.last_practiced_at || w.saved_at;
    const days = ref ? (now - Date.parse(ref)) / 86400000 : 999;
    const staleness = Math.min(Math.max(days, 0) / 14, 1);
    const need = (1 - s) * 0.6 + staleness * 0.4 + Math.random() * 0.08;
    return { w, need };
  });
  scored.sort((a, b) => b.need - a.need);
  const picked = scored.slice(0, n).map((x) => x.w);
  lastShown = new Set(picked.map((w) => w.word));
  return picked;
}

// ── Nội dung toast (JSX) ─────────────────────────────────────────────────
function reviewIpa(w: SavedWord): string {
  const s = (w.phonemes || []).length ? ipaStressString(w.phonemes) : w.ipa || '';
  return s ? `/${s}/` : '';
}

function ReviewToastCard({ words, toastId }: { words: SavedWord[]; toastId: string | number }) {
  const [rows, setRows] = useState(words);
  const openPractice = usePractice((s) => s.openPractice);
  const setWordMuted = useReviewToast((s) => s.setMuted);

  const muteRow = (w: SavedWord) => {
    setWordMuted(w.word, true);
    const next = rows.filter((r: SavedWord) => mkey(r.word) !== mkey(w.word));
    setRows(next);
    document.dispatchEvent(new CustomEvent('reviewmute:changed'));
    if (!next.length) toast.dismiss(toastId);
  };

  // toast.custom KHÔNG mang style mặc định của sonner (classNames.toast chỉ áp cho
  // toast dựng sẵn) → phải tự dựng mặt card: nền, viền, bo, đổ bóng.
  // `mt-[3.4rem]`: toast neo top-right, chừa chỗ cho .theme-toggle (top 1rem, cao 42px)
  // để rơi ngay DƯỚI nút dark mode thay vì đè lên.
  return (
    <div className="mt-[3.4rem] w-[272px] max-w-[80vw] overflow-hidden rounded-xl border border-border bg-popover text-popover-foreground shadow-xl">
      <div className="flex items-center gap-1.5 border-b bg-muted/40 px-3 py-2 text-sm font-semibold">
        <span aria-hidden>📖</span> Ôn lại từ đã lưu
        <span className="ml-auto text-xs font-normal text-muted-foreground">{rows.length} từ</span>
      </div>
      <div className="flex flex-col p-1">
        {rows.map((w: SavedWord) => (
          <div key={w.word} className="group flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent">
            <button
              type="button"
              className="flex-1 truncate rounded text-left font-medium text-primary hover:underline"
              title="Bấm để luyện từ này"
              onClick={() => {
                openPractice({ word: w.word, ipa: w.ipa ?? null, accuracy: w.accuracy ?? null, phonemes: w.phonemes || [] });
                toast.dismiss(toastId);
              }}
            >
              {w.word}
            </button>
            <span className="shrink-0 font-mono text-xs text-muted-foreground">{reviewIpa(w)}</span>
            {/* .tts-play + data-word → delegated playback handler phát mẫu qua /tts. */}
            <button
              type="button"
              className="tts-play shrink-0 rounded p-0.5 text-muted-foreground opacity-70 hover:bg-background hover:text-foreground hover:opacity-100"
              data-word={w.word}
              title="Nghe phát âm chuẩn"
            >
              <Volume2 className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              className="shrink-0 rounded p-0.5 text-muted-foreground opacity-60 hover:bg-background hover:text-foreground hover:opacity-100"
              title="Dừng nhắc ôn từ này"
              onClick={() => muteRow(w)}
            >
              <BellOff className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Scheduler (timers module-level) ──────────────────────────────────────
let rtTimer: ReturnType<typeof setTimeout> | null = null;
let pendingHidden = false;
let toastVisible = false;

function isMobileViewport() {
  return typeof window.matchMedia === 'function' && window.matchMedia('(max-width: 640px)').matches;
}

function schedule(delayMs: number) {
  if (rtTimer) clearTimeout(rtTimer);
  rtTimer = null;
  if (!useReviewToast.getState().settings.enabled) return;
  rtTimer = setTimeout(maybeShow, delayMs);
}

function reArm() {
  const cfg = useReviewToast.getState().settings;
  if (!cfg.enabled) {
    if (rtTimer) clearTimeout(rtTimer);
    rtTimer = null;
    return;
  }
  schedule(cfg.intervalMin * 60000);
}

function reconcile() {
  const cfg = useReviewToast.getState().settings;
  if (!cfg.enabled) {
    if (rtTimer) clearTimeout(rtTimer);
    rtTimer = null;
    return;
  }
  if (!rtTimer) schedule(cfg.intervalMin * 60000);
}

function maybeShow() {
  const cfg = useReviewToast.getState().settings;
  if (!cfg.enabled) return;
  const intervalMs = cfg.intervalMin * 60000;

  if (document.hidden) {
    pendingHidden = true;
    return;
  }
  if (cfg.noMobile && isMobileViewport()) {
    schedule(intervalMs);
    return;
  }
  if (!useSavedWords.getState().loaded) {
    schedule(RT_LOAD_RETRY_MS);
    return;
  }
  // Bận: popup luyện đang mở hoặc toast đang hiện → hoãn.
  if (usePractice.getState().open || toastVisible) {
    schedule(RT_BUSY_RETRY_MS);
    return;
  }
  const words = pickReviewWords(cfg.count);
  if (!words.length) {
    schedule(intervalMs);
    return;
  }
  toastVisible = true;
  toast.custom((id) => <ReviewToastCard words={words} toastId={id} />, {
    duration: cfg.hideSec * 1000,
    // Neo góc trên-phải (ngay dưới nút dark mode) thay vì bottom-right mặc định
    // của Toaster — chỉ toast này đổi vị trí, các toast khác giữ nguyên.
    position: 'top-right',
    // Bỏ khung mặc định: `unstyled` tắt CSS riêng của sonner, còn classNames.toast của
    // Toaster (bg/border/shadow) vẫn áp cho toast.custom → phải ghi đè !important,
    // nếu không lớp nền ngoài cao hơn card và đè lên nút dark mode.
    unstyled: true,
    className: '!bg-transparent !border-0 !p-0 !shadow-none',
    onDismiss: () => {
      toastVisible = false;
    },
    onAutoClose: () => {
      toastVisible = false;
    },
  });
  schedule(intervalMs);
}

/** Gắn 1 lần lúc mở app (main.tsx): mồi lịch + sync server + xử lý visibility. */
export function installReviewToast() {
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && pendingHidden) {
      pendingHidden = false;
      schedule(3000);
    }
  });
  schedule(RT_FIRST_DELAY_MS);
  useReviewToast.getState().loadFromServer();
}
