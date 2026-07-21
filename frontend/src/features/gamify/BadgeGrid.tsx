// Lưới huy hiệu — mở khóa (nổi bật) vs chưa (xám). `compact` = hàng ngang nhỏ cho
// header; mặc định lưới đầy đủ. Nguồn earned = server (useXp badges / props).

import { motion } from 'motion/react';
import { BADGE_ORDER, badgeMeta } from './badges';

export default function BadgeGrid({
  earned,
  compact = false,
  className = '',
}: {
  earned: string[];
  compact?: boolean;
  className?: string;
}) {
  const earnedSet = new Set(earned);
  const ids = compact
    ? // Compact: chỉ hiện huy hiệu ĐÃ mở (tối đa 6) — header gọn.
      BADGE_ORDER.filter((id) => earnedSet.has(id)).slice(0, 6)
    : BADGE_ORDER;

  if (compact && ids.length === 0) return null;

  return (
    <div className={`flex flex-wrap ${compact ? 'gap-1.5' : 'gap-2.5'} ${className}`}>
      {ids.map((id) => {
        const m = badgeMeta(id);
        const has = earnedSet.has(id);
        if (compact) {
          return (
            <motion.span
              key={id}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 300, damping: 18 }}
              className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-amber-300 bg-amber-50 text-lg shadow-sm dark:border-amber-500/40 dark:bg-amber-950/40"
              title={`${m.label} — ${m.desc}`}
            >
              {m.icon}
            </motion.span>
          );
        }
        return (
          <div
            key={id}
            title={`${m.label} — ${m.desc}`}
            className={`flex w-20 flex-col items-center gap-1 rounded-xl border p-2.5 text-center transition-colors ${
              has
                ? 'border-amber-300 bg-amber-50 dark:border-amber-500/40 dark:bg-amber-950/30'
                : 'border-dashed bg-muted/30 opacity-60'
            }`}
          >
            <span className={`text-2xl ${has ? '' : 'grayscale'}`} aria-hidden>
              {has ? m.icon : '🔒'}
            </span>
            <span className="text-[0.7rem] font-medium leading-tight text-foreground">{m.label}</span>
          </div>
        );
      })}
    </div>
  );
}
