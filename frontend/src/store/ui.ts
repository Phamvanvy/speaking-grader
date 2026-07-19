// Zustand — CHỈ UI/client state (theme, accent). Server state thuộc TanStack Query.
//
// Theme: giữ đúng cơ chế legacy — class `body.dark` + key 'toeic-grader-theme'
// (form.js). Accent: key 'pron_accent' (state.js) — 'default' chấp nhận cả Anh-Anh/
// Anh-Mỹ lúc CHẤM; 'gb'/'us' chỉ đổi hiển thị IPA.

import { create } from 'zustand';
import { ACCENT_KEY, type AuthInfo } from '../lib/identity';
import { VALID_ACCENTS, type Accent } from '../lib/config';

const THEME_KEY = 'toeic-grader-theme';
export type Theme = 'light' | 'dark';

function initialTheme(): Theme {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === 'dark' || saved === 'light') return saved;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme: Theme): void {
  document.body.classList.toggle('dark', theme === 'dark');
}

function initialAccent(): Accent {
  const saved = localStorage.getItem(ACCENT_KEY);
  return VALID_ACCENTS.includes(saved as Accent) ? (saved as Accent) : 'default';
}

interface UiState {
  theme: Theme;
  accent: Accent;
  toggleTheme: () => void;
  setAccent: (a: Accent) => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  theme: initialTheme(),
  accent: initialAccent(),
  toggleTheme: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
    set({ theme: next });
  },
  setAccent: (a) => {
    const accent = VALID_ACCENTS.includes(a) ? a : 'default';
    localStorage.setItem(ACCENT_KEY, accent);
    set({ accent });
  },
}));

// Áp theme ngay khi module nạp (trước first paint của React đủ sớm cho body class).
applyTheme(useUiStore.getState().theme);

// Delegated: mọi <select class="accent-select"> trong HTML do renderer legacy inject
// (Pronunciation detail) — đổi giọng gọi setAccent → store update → kết quả re-render
// với accent mới (không chấm lại). Cài 1 lần. Port state.js:206.
let _accentDelegated = false;
export function installAccentDelegation() {
  if (_accentDelegated) return;
  _accentDelegated = true;
  document.addEventListener('change', (e) => {
    const t = e.target as HTMLElement;
    if (t instanceof HTMLSelectElement && t.classList.contains('accent-select')) {
      useUiStore.getState().setAccent(t.value as Accent);
    }
  });
}

export type { AuthInfo };
