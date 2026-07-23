// Bảng game hóa của người dùng: cấp độ/XP, streak, nhiệm vụ ngày, xu, huy hiệu,
// cửa hàng cosmetic + xếp hạng tuần. Gom về trang Tài khoản (user info) thay vì gắn
// rải rác ở tab Khóa học / Từ đã lưu — cấp độ là thuộc tính của người dùng, không
// thuộc riêng một khóa học hay danh sách từ. Tự nạp snapshot XP khi mount (server là
// nguồn sự thật); ẩn hoàn toàn khi tắt cờ COURSE_XP_ENABLED (data == null).

import { useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useXp } from '@/store/xp';
import { useShop } from '@/store/shop';
import { useLeaderboard } from '@/store/leaderboard';
import XpBar from './XpBar';
import StreakFlame from './StreakFlame';
import BadgeGrid from './BadgeGrid';
import { DailyGoalRing, CoinChip } from './DailyGoal';
import ShopDialog from './ShopDialog';
import Leaderboard from './Leaderboard';

/** Nút pill (mở cửa hàng / xếp hạng) — kiểu giống nhau, chỉ khác icon + nhãn. */
function PillButton({ icon, label, title, onClick }: { icon: string; label: string; title: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1.5 rounded-full border border-border bg-muted/40 px-3 py-1.5 text-sm font-medium transition-colors hover:border-primary hover:bg-primary/5"
      title={title}
    >
      <span aria-hidden>{icon}</span>
      <span>{label}</span>
    </button>
  );
}

export default function GamifyPanel() {
  const data = useXp((s) => s.data);
  const fetchXp = useXp((s) => s.fetch);
  const openShop = useShop((s) => s.openShop);
  const openBoard = useLeaderboard((s) => s.openBoard);

  // Nạp XP mỗi lần vào trang Tài khoản (đây là nơi hiển thị chính sau khi tách khỏi tab).
  useEffect(() => {
    fetchXp();
  }, [fetchXp]);

  if (!data) return null;
  const badges = data.badges?.map((b) => b.id) ?? [];

  return (
    <>
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">🎮 Cấp độ & thành tích</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {/* Hàng 1 — tiến độ chính: thanh XP full, streak + nhiệm vụ ngày + xu cùng hàng khi đủ chỗ. */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
            <XpBar className="min-w-[220px] flex-1" />
            {data.streak && (
              <StreakFlame days={data.streak.streak_days} longest={data.streak.longest_streak} />
            )}
            <DailyGoalRing />
            <CoinChip />
          </div>
          {/* Huy hiệu — bản đầy đủ (có chỗ ở trang Tài khoản): hiện cả huy hiệu chưa mở. */}
          <div className="border-t border-border/60 pt-3">
            <span className="mb-2 block text-xs font-medium text-muted-foreground">Huy hiệu</span>
            <BadgeGrid earned={badges} />
          </div>
          {/* Hành động: cửa hàng cosmetic + xếp hạng tuần. */}
          <div className="flex flex-wrap items-center gap-2.5 border-t border-border/60 pt-3">
            <PillButton icon="🛍️" label="Cửa hàng" title="Cửa hàng cosmetic" onClick={openShop} />
            <PillButton icon="🏆" label="Xếp hạng" title="Bảng xếp hạng tuần" onClick={openBoard} />
          </div>
        </CardContent>
      </Card>
      {/* Dialog dùng chung (portal) — mount cùng bảng để nút mở được. */}
      <ShopDialog />
      <Leaderboard />
    </>
  );
}
