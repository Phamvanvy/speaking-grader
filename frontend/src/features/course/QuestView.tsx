// Màn Quest nhập vai (/course/quest/roleplay/:topic) — Phase 3B. Tải kịch bản (LLM
// sinh, cache server), render RolePlaySession, và khi xong hội thoại thì ghi kết quả
// qua completeQuest (server clamp + ngưỡng + award XP/xu/huy hiệu MỘT LẦN, bonus-only)
// rồi ăn mừng. Kịch bản null (LLM lỗi) → hiện thông báo + quay lại (fail-soft).

import { useNavigate, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { getUserId } from '@/lib/identity';
import { useUiStore } from '@/store/ui';
import { useCourseStore } from '@/store/course';
import { useXp } from '@/store/xp';
import { celebrateComplete } from '@/lib/celebrate';
import RolePlaySession from './game/RolePlaySession';
import { getRoleplayQuest, completeQuest } from './courseApi';

export default function QuestView() {
  const { topic = '' } = useParams();
  const exam = useCourseStore((s) => s.exam);
  const accent = useUiStore((s) => s.accent);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const userId = getUserId();

  const q = useQuery({
    queryKey: ['course', 'quest', 'roleplay', exam, topic, userId],
    queryFn: () => getRoleplayQuest(exam, topic),
    enabled: !!topic,
    retry: false,
  });
  const script = q.data;

  async function onCompleted(avgScore: number) {
    try {
      const res = await completeQuest(script!.quest_id, 'roleplay', avgScore);
      qc.invalidateQueries({ queryKey: ['course'] });
      if (res.xp) useXp.getState().ingest(res.xp);
      if (res.done) {
        if (!res.xp?.leveled_up && !res.xp?.new_badges?.length) celebrateComplete();
        toast.success('Hoàn thành nhiệm vụ nhập vai! 🎭');
      } else {
        toast('Đã xong hội thoại — luyện thêm để đạt mốc thưởng nhé.');
      }
      navigate('/course');
    } catch (e: any) {
      toast.error(`Lỗi ghi kết quả Quest: ${e.message || e}`);
    }
  }

  return (
    <div id="mode-course-quest">
      <div className="card">
        <div className="result-header">
          <button className="btn btn-secondary btn-inline" onClick={() => navigate('/course')}>
            ‹ Khóa học
          </button>
          <h2 className="course-lesson-head">🎭 Nhiệm vụ nhập vai</h2>
        </div>
        {q.isLoading && <p className="history-empty">⏳ Đang dựng kịch bản hội thoại…</p>}
        {(q.isError || (!q.isLoading && !script)) && (
          <p className="history-empty">
            😕 Chưa dựng được kịch bản lúc này — hãy thử lại sau.
          </p>
        )}
        {script && <RolePlaySession script={script} accent={accent} onCompleted={onCompleted} />}
      </div>
    </div>
  );
}
