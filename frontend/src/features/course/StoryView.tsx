// Màn Quest truyện đọc-to (/course/quest/story/:topic) — Phase 3C. Tải truyện (LLM
// sinh, cache server), render StorySession, và khi đọc hết thì ghi kết quả qua
// completeQuest(kind='story') (server clamp + ngưỡng + award XP/xu/huy hiệu MỘT LẦN,
// bonus-only) rồi ăn mừng. Truyện null (LLM lỗi) → thông báo + quay lại (fail-soft).

import { useNavigate, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { getUserId } from '@/lib/identity';
import { useUiStore } from '@/store/ui';
import { useCourseStore } from '@/store/course';
import { useXp } from '@/store/xp';
import { celebrateComplete } from '@/lib/celebrate';
import StorySession from './game/StorySession';
import { getStoryQuest, completeQuest } from './courseApi';

export default function StoryView() {
  const { topic = '' } = useParams();
  const exam = useCourseStore((s) => s.exam);
  const accent = useUiStore((s) => s.accent);
  const navigate = useNavigate();
  const qc = useQueryClient();
  const userId = getUserId();

  const q = useQuery({
    queryKey: ['course', 'quest', 'story', exam, topic, userId],
    queryFn: () => getStoryQuest(exam, topic),
    enabled: !!topic,
    retry: false,
  });
  const story = q.data;

  async function onCompleted(avgScore: number) {
    try {
      const res = await completeQuest(story!.quest_id, 'story', avgScore);
      qc.invalidateQueries({ queryKey: ['course'] });
      if (res.xp) useXp.getState().ingest(res.xp);
      if (res.done) {
        if (!res.xp?.leveled_up && !res.xp?.new_badges?.length) celebrateComplete();
        toast.success('Hoàn thành truyện đọc-to! 📖');
      } else {
        toast('Đã đọc hết truyện — luyện thêm để đạt mốc thưởng nhé.');
      }
      navigate('/course');
    } catch (e: any) {
      toast.error(`Lỗi ghi kết quả Quest: ${e.message || e}`);
    }
  }

  return (
    <div id="mode-course-story">
      <div className="card">
        <div className="result-header">
          <button className="btn btn-secondary btn-inline" onClick={() => navigate('/course')}>
            ‹ Khóa học
          </button>
          <h2 className="course-lesson-head">📖 Truyện đọc-to</h2>
        </div>
        {q.isLoading && <p className="history-empty">⏳ Đang dựng truyện…</p>}
        {(q.isError || (!q.isLoading && !story)) && (
          <p className="history-empty">
            😕 Chưa dựng được truyện lúc này — hãy thử lại sau.
          </p>
        )}
        {story && <StorySession story={story} accent={accent} onCompleted={onCompleted} />}
      </div>
    </div>
  );
}
