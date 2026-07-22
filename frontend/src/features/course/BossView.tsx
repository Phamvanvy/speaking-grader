// Màn Boss cuối chặng (/course/unit/:unitId/boss) — Phase 3A. Tải nội dung Boss
// (đoạn đọc-to tổng hợp), render BossSession, và khi hạ được thì ghi kết quả qua
// completeBoss (server gate + award XP/xu/huy hiệu MỘT LẦN, bonus-only) rồi ăn mừng.
// Chưa mở khóa (còn lesson chưa done) → server trả 403 → hiện thông báo + quay lại.

import { useNavigate, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { getUserId } from '@/lib/identity';
import { useXp } from '@/store/xp';
import { celebrateComplete } from '@/lib/celebrate';
import BossSession from './game/BossSession';
import { getBossContent, completeBoss } from './courseApi';

export default function BossView() {
  const { unitId = '' } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const userId = getUserId();

  const q = useQuery({
    queryKey: ['course', 'boss', unitId, userId],
    queryFn: () => getBossContent(unitId),
    enabled: !!unitId,
    retry: false, // 403 (chưa mở khóa) không nên retry
  });
  const boss = q.data;

  async function onDefeated(score: number) {
    try {
      const res = await completeBoss(unitId, score);
      qc.invalidateQueries({ queryKey: ['course'] });
      if (res.xp) useXp.getState().ingest(res.xp);
      if (res.done) {
        if (!res.xp?.leveled_up && !res.xp?.new_badges?.length) celebrateComplete();
        toast.success('Hạ Boss! 🎉 Chặng đã chinh phục.');
        navigate('/course');
      }
    } catch (e: any) {
      toast.error(`Lỗi ghi kết quả Boss: ${e.message || e}`);
    }
  }

  return (
    <div id="mode-course-boss">
      <div className="card">
        <div className="result-header">
          <button className="btn btn-secondary btn-inline" onClick={() => navigate('/course')}>
            ‹ Khóa học
          </button>
          {boss && <h2 className="course-lesson-head">{boss.title}</h2>}
        </div>
        {q.isLoading && <p className="history-empty">⏳ Đang triệu hồi Boss…</p>}
        {q.isError && (
          <p className="history-empty">
            🔒 Chưa mở khóa Boss — hãy hoàn thành tất cả bài trong chặng trước.
          </p>
        )}
        {boss && <BossSession boss={boss} onDefeated={onDefeated} />}
      </div>
    </div>
  );
}
