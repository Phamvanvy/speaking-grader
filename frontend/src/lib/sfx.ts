// SFX game hóa — phát file âm thanh ngắn (public/sounds/*.wav) khi đúng/sai/lên
// cấp/hoàn thành/mở huy hiệu. File bundled → precache qua Workbox (xem vite.config,
// globPatterns có wav) nên chạy cả offline (PWA). Tôn trọng cờ TẮT TIẾNG lưu ở
// localStorage. Pattern audio giống playback.ts (module-level Audio, .catch nuốt
// AbortError/NotAllowed để tự-play không văng lỗi).

import { create } from 'zustand';

const MUTE_KEY = 'sg-sound-muted';

export type SfxName = 'correct' | 'wrong' | 'levelup' | 'complete' | 'badge' | 'tap';

// Đường dẫn TƯƠNG ĐỐI theo origin trang (frontend serve public/), KHÔNG qua apiBase
// (API có thể ở origin khác lúc dev). Vite dev + FastAPI build đều serve tại '/'.
const FILES: Record<SfxName, string> = {
  correct: '/sounds/correct.wav',
  wrong: '/sounds/wrong.wav',
  levelup: '/sounds/levelup.wav',
  complete: '/sounds/complete.wav',
  badge: '/sounds/badge.wav',
  tap: '/sounds/tap.wav',
};

// Âm lượng mặc định theo loại (tap nhỏ để không chói khi bấm liên tục).
const VOL: Record<SfxName, number> = {
  correct: 0.55,
  wrong: 0.4,
  levelup: 0.6,
  complete: 0.6,
  badge: 0.5,
  tap: 0.25,
};

const cache = new Map<SfxName, HTMLAudioElement>();

function initialMuted(): boolean {
  return localStorage.getItem(MUTE_KEY) === '1';
}

interface SoundState {
  muted: boolean;
  toggle: () => void;
}

/** Trạng thái tắt tiếng (zustand để nút SoundToggle re-render). */
export const useSound = create<SoundState>((set, get) => ({
  muted: initialMuted(),
  toggle: () => {
    const next = !get().muted;
    localStorage.setItem(MUTE_KEY, next ? '1' : '0');
    set({ muted: next });
  },
}));

/** Phát 1 SFX (no-op khi đang tắt tiếng). Tái dùng cùng <Audio> theo tên. */
export function playSfx(name: SfxName): void {
  if (useSound.getState().muted) return;
  let a = cache.get(name);
  if (!a) {
    a = new Audio(FILES[name]);
    a.preload = 'auto';
    cache.set(name, a);
  }
  try {
    a.currentTime = 0;
    a.volume = VOL[name];
    const p = a.play();
    if (p && typeof p.catch === 'function') p.catch(() => {});
  } catch {
    /* trình duyệt chặn autoplay trước tương tác — bỏ qua */
  }
}
