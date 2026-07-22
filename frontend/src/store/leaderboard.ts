// Bảng xếp hạng tuần (Phase 5 game hóa) — CHỈ tài khoản đã opt-in xuất hiện. Server
// là nguồn sự thật của hạng/XP tuần; opt-in cần token đăng nhập (withAuth tự gắn).
// Ẩn danh vẫn xem được bảng nhưng không có hạng. Mọi lỗi nuốt im lặng + toast.

import { create } from 'zustand';
import { toast } from 'sonner';
import { apiGet, apiPostForm } from '@/lib/api';
import { getUserId } from '@/lib/identity';

export interface LeaderboardEntry {
  rank: number;
  username: string;
  weekly_xp: number;
  level: number;
  is_me: boolean;
}

export interface LeaderboardData {
  enabled: boolean;
  week_start: string;
  goal: number;
  opted_in: boolean;
  entries: LeaderboardEntry[];
  me: LeaderboardEntry | null;
}

interface LeaderboardStore {
  open: boolean;
  data: LeaderboardData | null;
  loading: boolean;
  saving: boolean;
  openBoard: () => void;
  close: () => void;
  refresh: () => Promise<void>;
  setOptIn: (optIn: boolean) => Promise<void>;
}

export const useLeaderboard = create<LeaderboardStore>((set, get) => ({
  open: false,
  data: null,
  loading: false,
  saving: false,

  openBoard: () => {
    set({ open: true, loading: get().data == null });
    void get().refresh();
  },
  close: () => set({ open: false }),

  refresh: async () => {
    try {
      const d = await apiGet<LeaderboardData>(
        `/course/leaderboard?user_id=${encodeURIComponent(getUserId())}`,
      );
      set({ data: d.enabled ? d : null, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  setOptIn: async (optIn) => {
    set({ saving: true });
    try {
      const fd = new FormData();
      fd.append('user_id', getUserId());
      fd.append('opt_in', String(optIn));
      await apiPostForm('/course/leaderboard/optin', fd);
      await get().refresh();
      toast.success(optIn ? '🏆 Đã tham gia bảng xếp hạng' : 'Đã ẩn khỏi bảng xếp hạng');
    } catch {
      toast.error('Không đổi được cài đặt', { description: 'Cần đăng nhập để tham gia.' });
    } finally {
      set({ saving: false });
    }
  },
}));
