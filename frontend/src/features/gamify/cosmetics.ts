// Ánh xạ item_id (cosmetic mua ở cửa hàng) → style hiển thị. Backend chỉ giữ
// id/slot/giá/nhãn; PHẦN NHÌN (màu thanh XP, màu ngọn lửa) là việc của frontend.
// Item thuần trang trí — KHÔNG chạm chấm điểm/XP.
//
// NB: các chuỗi class Tailwind phải viết ĐẦY ĐỦ (không nối động) để JIT quét thấy.

export interface XpTheme {
  from: string; // class gradient bắt đầu
  to: string; // class gradient kết thúc
}

const DEFAULT_XP_THEME: XpTheme = { from: 'from-amber-400', to: 'to-orange-500' };

const XP_THEMES: Record<string, XpTheme> = {
  xp_ocean: { from: 'from-cyan-400', to: 'to-blue-500' },
  xp_forest: { from: 'from-lime-400', to: 'to-emerald-500' },
  xp_sunset: { from: 'from-pink-400', to: 'to-rose-500' },
  xp_royal: { from: 'from-violet-400', to: 'to-fuchsia-500' },
};

/** Gradient thanh XP theo cosmetic đang trang bị (slot 'xp_theme'); mặc định cam. */
export function xpTheme(itemId?: string | null): XpTheme {
  return (itemId && XP_THEMES[itemId]) || DEFAULT_XP_THEME;
}

// Ngọn lửa streak dùng emoji 🔥 (cam) — đổi màu bằng CSS hue-rotate (rẻ, không
// cần asset). '' = giữ màu cam mặc định.
const FLAME_FILTERS: Record<string, string> = {
  flame_azure: 'hue-rotate(190deg) saturate(1.3)',
  flame_violet: 'hue-rotate(255deg) saturate(1.4)',
  flame_emerald: 'hue-rotate(95deg) saturate(1.2)',
};

/** CSS filter cho ngọn lửa theo cosmetic (slot 'streak_flame'); '' nếu mặc định. */
export function flameFilter(itemId?: string | null): string {
  return (itemId && FLAME_FILTERS[itemId]) || '';
}
