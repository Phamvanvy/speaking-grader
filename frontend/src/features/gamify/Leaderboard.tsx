// Bảng xếp hạng tuần (Phase 5): top tài khoản đã opt-in theo XP-practice 7 ngày.
// Quyền riêng tư: CHỈ tài khoản đăng nhập tham gia (opt-in), ẩn danh chỉ xem. Toggle
// tham gia hiện khi đã đăng nhập; chưa đăng nhập → nhắc đăng nhập.

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { Loader2, Trophy } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLeaderboard, type LeaderboardEntry } from '@/store/leaderboard';
import { useAuthStore } from '@/store/auth';

const MEDAL = ['🥇', '🥈', '🥉'];

export default function Leaderboard() {
  const open = useLeaderboard((s) => s.open);
  const close = useLeaderboard((s) => s.close);
  const data = useLeaderboard((s) => s.data);
  const loading = useLeaderboard((s) => s.loading);
  const saving = useLeaderboard((s) => s.saving);
  const setOptIn = useLeaderboard((s) => s.setOptIn);
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn);

  const me = data?.me ?? null;
  const goalPct = data && me ? Math.min(100, Math.round((me.weekly_xp / data.goal) * 100)) : 0;

  return (
    <Dialog open={open} onOpenChange={(o) => !o && close()}>
      <DialogContent className="max-h-[92vh] max-w-md gap-4 overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Trophy className="h-5 w-5 text-amber-500" /> Bảng xếp hạng tuần
          </DialogTitle>
        </DialogHeader>

        {/* Thử thách tuần: tiến độ XP của tôi so với mục tiêu */}
        {data && (
          <div className="rounded-xl border bg-muted/30 p-3">
            <div className="mb-1 flex items-baseline justify-between text-xs">
              <span className="font-semibold">🎯 Thử thách tuần</span>
              <span className="tabular-nums text-muted-foreground">
                {me?.weekly_xp ?? 0} / {data.goal} XP
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-gradient-to-r from-amber-400 to-orange-500 transition-all"
                style={{ width: `${goalPct}%` }}
              />
            </div>
          </div>
        )}

        {/* Điều khiển tham gia (chỉ tài khoản đăng nhập) */}
        {isLoggedIn ? (
          <label className="flex items-center justify-between gap-3 rounded-xl border px-3 py-2.5">
            <span className="text-sm">
              <span className="font-medium">Tham gia bảng xếp hạng</span>
              <span className="block text-xs text-muted-foreground">
                Hiện tên đăng nhập + XP tuần của bạn cho người khác.
              </span>
            </span>
            <Switch
              checked={!!data?.opted_in}
              disabled={saving}
              onCheckedChange={(v) => setOptIn(v)}
            />
          </label>
        ) : (
          <p className="rounded-xl border border-dashed px-3 py-2.5 text-sm text-muted-foreground">
            Đăng nhập để tham gia bảng xếp hạng. Người dùng ẩn danh chỉ xem được.
          </p>
        )}

        {/* Danh sách hạng */}
        {loading && !data ? (
          <div className="flex justify-center py-8 text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : data && data.entries.length > 0 ? (
          <ol className="flex flex-col gap-1.5">
            {data.entries.map((e) => (
              <Row key={e.rank} e={e} />
            ))}
            {/* Nếu tôi ngoài top thì ghim hạng của tôi ở cuối */}
            {me && !data.entries.some((e) => e.is_me) && (
              <>
                <li className="py-1 text-center text-xs text-muted-foreground">···</li>
                <Row e={me} />
              </>
            )}
          </ol>
        ) : (
          <p className="py-6 text-center text-sm text-muted-foreground">
            Chưa có ai trên bảng tuần này. {isLoggedIn ? 'Bật tham gia và luyện tập để dẫn đầu!' : ''}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Row({ e }: { e: LeaderboardEntry }) {
  return (
    <li
      className={cn(
        'flex items-center gap-3 rounded-xl border px-3 py-2',
        e.is_me ? 'border-primary bg-primary/5' : 'border-border',
      )}
    >
      <span className="w-7 shrink-0 text-center text-sm font-bold tabular-nums">
        {e.rank <= 3 ? MEDAL[e.rank - 1] : e.rank}
      </span>
      <span className="min-w-0 flex-1 truncate text-sm font-medium">
        {e.username}
        {e.is_me && <span className="ml-1.5 text-xs text-primary">(bạn)</span>}
      </span>
      <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-xs font-semibold tabular-nums">
        Lv {e.level}
      </span>
      <span className="shrink-0 text-sm font-bold tabular-nums text-amber-600 dark:text-amber-400">
        {e.weekly_xp} XP
      </span>
    </li>
  );
}
