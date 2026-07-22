// Cửa hàng cosmetic (Phase 4 game hóa). Server là nguồn sự thật của xu/giá/sở hữu
// (client CHỈ gửi item_id — backend giữ giá, RB#5). Store giữ snapshot danh mục +
// điều khiển mở/đóng dialog. Sau mua/trang bị, đồng bộ ví + cosmetic sang useXp để
// CoinChip / thanh XP / ngọn lửa cập nhật ngay. Mọi lỗi nuốt im lặng (gamify không
// được chặn luồng chính) và báo người dùng qua toast.

import { create } from 'zustand';
import { toast } from 'sonner';
import { apiGet, apiPostForm } from '@/lib/api';
import { getUserId } from '@/lib/identity';
import { useXp } from './xp';
import { playSfx } from '@/lib/sfx';
import { burstConfetti } from '@/lib/celebrate';

export interface ShopItem {
  id: string;
  slot: string;
  price: number;
  icon: string;
  label: string;
  desc: string;
  owned: boolean;
  equipped: boolean;
  affordable: boolean;
}

export interface ShopState {
  enabled: boolean;
  coins: number;
  items: ShopItem[];
  cosmetics: Record<string, string>;
}

interface ShopStore {
  open: boolean;
  data: ShopState | null;
  loading: boolean;
  busyId: string | null; // item đang mua/trang bị (khóa nút, tránh double-submit)
  openShop: () => void;
  close: () => void;
  buy: (itemId: string) => Promise<void>;
  equip: (itemId: string, equipped: boolean) => Promise<void>;
}

/** Đồng bộ snapshot cửa hàng vào store + ví/cosmetic của useXp. */
function absorb(state: ShopState | null): ShopState | null {
  if (!state || !state.enabled) return null;
  useXp.getState().syncWallet(state.coins, state.cosmetics || {});
  return state;
}

export const useShop = create<ShopStore>((set, get) => ({
  open: false,
  data: null,
  loading: false,
  busyId: null,

  openShop: () => {
    set({ open: true, loading: get().data == null });
    apiGet<ShopState>(`/course/shop?user_id=${encodeURIComponent(getUserId())}`)
      .then((s) => set({ data: absorb(s), loading: false }))
      .catch(() => set({ loading: false }));
  },

  close: () => set({ open: false }),

  buy: async (itemId) => {
    if (get().busyId) return;
    set({ busyId: itemId });
    try {
      const fd = new FormData();
      fd.append('user_id', getUserId());
      fd.append('item_id', itemId);
      const s = await apiPostForm<ShopState>('/course/shop/buy', fd);
      set({ data: absorb(s) });
      playSfx('correct');
      burstConfetti();
      toast.success('🛍️ Đã mua vật phẩm!', { description: 'Chạm "Trang bị" để dùng ngay.' });
    } catch (e) {
      toast.error('Không mua được', {
        description: e instanceof Error ? e.message : 'Kiểm tra lại số xu.',
      });
    } finally {
      set({ busyId: null });
    }
  },

  equip: async (itemId, equipped) => {
    if (get().busyId) return;
    set({ busyId: itemId });
    try {
      const fd = new FormData();
      fd.append('user_id', getUserId());
      fd.append('item_id', itemId);
      fd.append('equipped', String(equipped));
      const s = await apiPostForm<ShopState>('/course/shop/equip', fd);
      set({ data: absorb(s) });
      if (equipped) playSfx('correct');
    } catch (e) {
      toast.error('Không trang bị được', {
        description: e instanceof Error ? e.message : undefined,
      });
    } finally {
      set({ busyId: null });
    }
  },
}));
