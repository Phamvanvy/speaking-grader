// Nhiệm vụ hằng ngày (Phase 2 game hóa) + chip xu. Đọc snapshot từ useXp (server
// là nguồn sự thật của count/goal/coins). Vòng tiến độ đầy khi luyện đủ goal từ;
// ăn mừng chạm mốc do store/xp.ts xử lý (cờ daily_goal_hit) — component chỉ VẼ.

import { motion } from 'motion/react';
import { useXp } from '@/store/xp';

/** Vòng tiến độ "luyện N/goal từ hôm nay" + nhãn. Ẩn khi thiếu dữ liệu daily. */
export function DailyGoalRing({ className = '' }: { className?: string }) {
  const daily = useXp((s) => s.data?.daily);
  if (!daily || daily.goal <= 0) return null;
  const { count, goal, done } = daily;
  const pct = Math.min(1, count / goal);
  const r = 16;
  const c = 2 * Math.PI * r;
  return (
    <div
      className={`flex items-center gap-2.5 rounded-full border px-3 py-1.5 ${
        done
          ? 'border-emerald-300 bg-emerald-50 dark:border-emerald-500/40 dark:bg-emerald-950/40'
          : 'border-border bg-muted/40'
      } ${className}`}
      title={done ? 'Đã hoàn thành nhiệm vụ hôm nay 🎉' : `Luyện ${count}/${goal} từ hôm nay`}
    >
      <span className="relative flex h-9 w-9 shrink-0 items-center justify-center">
        <svg className="h-9 w-9 -rotate-90" viewBox="0 0 40 40" aria-hidden>
          <circle cx="20" cy="20" r={r} fill="none" strokeWidth="4" className="stroke-muted" />
          <motion.circle
            cx="20"
            cy="20"
            r={r}
            fill="none"
            strokeWidth="4"
            strokeLinecap="round"
            className={done ? 'stroke-emerald-500' : 'stroke-indigo-500'}
            strokeDasharray={c}
            initial={{ strokeDashoffset: c }}
            animate={{ strokeDashoffset: c * (1 - pct) }}
            transition={{ type: 'spring', stiffness: 120, damping: 20 }}
          />
        </svg>
        <span className="absolute text-xs" aria-hidden>
          {done ? '🎯' : '📆'}
        </span>
      </span>
      <span className="flex flex-col leading-tight">
        <span className="text-xs font-medium text-muted-foreground">Nhiệm vụ hôm nay</span>
        <span className="text-sm font-bold tabular-nums text-foreground">
          {count}/{goal} từ
        </span>
      </span>
    </div>
  );
}

/** Icon xu (SVG inline — không phụ thuộc font emoji của hệ điều hành). */
function CoinIcon({ className = '' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={`text-amber-500 dark:text-amber-400 ${className}`}
      aria-hidden
    >
      <circle cx="12" cy="12" r="9" fill="currentColor" />
      <circle cx="12" cy="12" r="6.5" fill="none" stroke="#fff" strokeOpacity="0.55" strokeWidth="1.2" />
      <path
        d="M12 8.2v7.6M10.2 9.2h2.7a1.5 1.5 0 0 1 0 3h-2.4m0 0h2.7a1.5 1.5 0 0 1 0 3h-2.7"
        fill="none"
        stroke="#fff"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Chip số xu tích lũy (xu chỉ dùng cho cửa hàng cosmetic — Phase 4). */
export function CoinChip({ className = '' }: { className?: string }) {
  const coins = useXp((s) => s.data?.coins);
  if (coins == null) return null;
  return (
    <div
      className={`flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3 py-1.5 dark:border-amber-500/40 dark:bg-amber-950/40 ${className}`}
      title={`${coins} xu`}
    >
      <CoinIcon className="h-4 w-4 shrink-0" />
      <span className="text-sm font-bold tabular-nums text-foreground">{coins}</span>
    </div>
  );
}
