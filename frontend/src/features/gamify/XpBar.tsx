// Thanh XP + level (game hóa). Đọc snapshot từ useXp (server là nguồn sự thật).
// Ẩn hoàn toàn khi chưa có dữ liệu / tắt cờ COURSE_XP_ENABLED.

import { motion } from 'motion/react';
import { NumberTicker } from '@/components/ui/number-ticker';
import { useXp } from '@/store/xp';
import { xpTheme } from './cosmetics';

export default function XpBar({ className = '' }: { className?: string }) {
  const data = useXp((s) => s.data);
  if (!data) return null;
  const pct = data.span > 0 ? Math.min(100, Math.round((data.into_level / data.span) * 100)) : 0;
  const theme = xpTheme(data.cosmetics?.xp_theme); // cosmetic cửa hàng (mặc định cam)

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Huy chương cấp độ */}
      <div className="relative flex h-11 w-11 shrink-0 items-center justify-center">
        <div className={`absolute inset-0 rounded-full bg-gradient-to-br ${theme.from} ${theme.to} shadow-md`} />
        <div className="absolute inset-[3px] rounded-full bg-background" />
        <span className="relative text-sm font-extrabold tabular-nums text-orange-600 dark:text-orange-400">
          {data.level}
        </span>
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
          <span className="font-semibold text-foreground">Cấp {data.level}</span>
          <span className="tabular-nums text-muted-foreground">
            <NumberTicker value={data.into_level} className="text-foreground" /> / {data.span} XP
          </span>
        </div>
        <div className="h-2.5 overflow-hidden rounded-full bg-muted">
          <motion.div
            className={`h-full rounded-full bg-gradient-to-r ${theme.from} ${theme.to}`}
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ type: 'spring', stiffness: 120, damping: 20 }}
          />
        </div>
      </div>
    </div>
  );
}
