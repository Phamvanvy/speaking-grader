// Trạng thái XP/level/huy hiệu (gamification) — server là nguồn sự thật. Store này
// giữ snapshot mới nhất + kích hiệu ứng ăn mừng khi lên cấp / mở huy hiệu.
//
// Backend TỰ tính XP: client chỉ gửi (event, score) tới /course/xp/award (RB#5).
// Payload award / lesson-complete / course-state đều mang cùng shape `XpState` (kèm
// cờ leveled_up + new_badges) → ingest() dùng chung, tránh double-fetch.

import { create } from 'zustand';
import { toast } from 'sonner';
import { apiGet, apiPostForm } from '@/lib/api';
import { getUserId } from '@/lib/identity';
import { celebrateLevelUp, celebrateBadge } from '@/lib/celebrate';
import { badgeMeta } from '@/features/gamify/badges';

export interface Badge {
  id: string;
  earned_at: string;
}

export interface XpState {
  enabled: boolean;
  xp: number;
  level: number;
  level_floor: number;
  level_ceil: number;
  into_level: number;
  span: number;
  coins: number;
  badges: Badge[];
  // Kèm ở /course/xp (không ở award) để tab Từ đã lưu hiện streak mà không gọi /course/state.
  streak?: {
    streak_days: number;
    longest_streak: number;
    last_active_day: string | null;
    total_completed: number;
  };
  // Chỉ có ở payload award/complete — dùng để quyết định ăn mừng.
  leveled_up?: boolean;
  new_badges?: string[];
  awarded?: number;
}

interface XpStore {
  data: XpState | null;
  fetch: () => Promise<void>;
  /** Gọi sau mỗi lần luyện từ thành công — server cộng XP + trả state mới. */
  award: (event: 'word_practice', score: number) => Promise<void>;
  /** Nhận state XP từ payload có sẵn (course-state / lesson-complete) + ăn mừng. */
  ingest: (state: XpState | null | undefined) => void;
}

function celebrate(state: XpState): void {
  if (state.leveled_up) celebrateLevelUp();
  const newBadges = state.new_badges || [];
  if (newBadges.length) {
    if (!state.leveled_up) celebrateBadge();
    for (const id of newBadges) {
      const m = badgeMeta(id);
      toast.success(`Mở khóa huy hiệu ${m.icon} ${m.label}!`, { description: m.desc });
    }
  }
}

export const useXp = create<XpStore>((set, get) => ({
  data: null,
  fetch: async () => {
    try {
      const state = await apiGet<XpState>(`/course/xp?user_id=${encodeURIComponent(getUserId())}`);
      set({ data: state.enabled ? state : null });
    } catch {
      /* im lặng — gamification không được chặn luồng chính */
    }
  },
  award: async (event, score) => {
    try {
      const fd = new FormData();
      fd.append('user_id', getUserId());
      fd.append('event', event);
      fd.append('score', String(score));
      const state = await apiPostForm<XpState>('/course/xp/award', fd);
      get().ingest(state);
    } catch {
      /* im lặng */
    }
  },
  ingest: (state) => {
    if (!state || !state.enabled) return;
    celebrate(state);
    set({ data: state });
  },
}));
