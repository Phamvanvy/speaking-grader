// Ngọn lửa streak (số ngày học liên tiếp). Đọc từ course.streak (server). Lửa
// "sống" (animate) khi streak > 0, xám khi = 0.

import { motion } from 'motion/react';
import { useXp } from '@/store/xp';
import { flameFilter } from './cosmetics';

export default function StreakFlame({
  days,
  longest,
  className = '',
}: {
  days: number;
  longest?: number;
  className?: string;
}) {
  const active = days > 0;
  const cosmetic = useXp((s) => s.data?.cosmetics?.streak_flame);
  const tint = active ? flameFilter(cosmetic) : ''; // đổi màu lửa theo cosmetic (chỉ khi đang cháy)
  return (
    <div
      className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 ${
        active
          ? 'border-orange-300 bg-orange-50 dark:border-orange-500/40 dark:bg-orange-950/40'
          : 'border-border bg-muted/40'
      } ${className}`}
      title={
        longest && longest > days
          ? `Chuỗi ${days} ngày · kỷ lục ${longest} ngày`
          : `Chuỗi học ${days} ngày liên tiếp`
      }
    >
      <motion.span
        aria-hidden
        className="text-lg leading-none"
        animate={active ? { scale: [1, 1.18, 1], rotate: [0, -4, 4, 0] } : {}}
        transition={active ? { duration: 1.4, repeat: Infinity, ease: 'easeInOut' } : {}}
        style={active ? (tint ? { filter: tint } : {}) : { filter: 'grayscale(1)', opacity: 0.5 }}
      >
        🔥
      </motion.span>
      <span className="text-sm font-bold tabular-nums text-foreground">{days}</span>
      <span className="text-xs text-muted-foreground">ngày</span>
    </div>
  );
}
