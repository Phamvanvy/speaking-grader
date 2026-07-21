// Tab "Khóa học" — giáo trình cá nhân hóa dạng LEARNING PATH (kiểu Duolingo/ELSA).
// Header game hóa: XP bar + streak + tiến độ tổng. Mỗi Unit là 1 chặng; lessons là
// node tròn trên đường zig-zag theo trạng thái (locked/available/in_progress/done)
// + badge "Nên học" (focus). Server state qua TanStack Query (courseApi); XP kèm sẵn
// trong payload /course/state → ingest vào useXp (không round-trip thêm).

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { RotateCw, Check, Lock, Play } from 'lucide-react';
import { NumberTicker } from '@/components/ui/number-ticker';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getUserId } from '@/lib/identity';
import { examConfig } from '@/lib/config';
import { useCourseStore, type CourseExam } from '@/store/course';
import { useXp } from '@/store/xp';
import XpBar from '@/features/gamify/XpBar';
import StreakFlame from '@/features/gamify/StreakFlame';
import BadgeGrid from '@/features/gamify/BadgeGrid';
import { getCourse, type CourseView, type LessonView, type LessonStatus } from './courseApi';

const DIM_ICON: Record<string, string> = {
  pronunciation: '🗣️',
  rubric: '📋',
  question_type: '🎯',
};

const EXAMS: CourseExam[] = ['toeic', 'ielts', 'topik'];

export default function CourseTab() {
  const exam = useCourseStore((s) => s.exam);
  const setExam = useCourseStore((s) => s.setExam);
  const ingestXp = useXp((s) => s.ingest);
  const navigate = useNavigate();
  const userId = getUserId();

  const q = useQuery({
    queryKey: ['course', exam, userId],
    queryFn: () => getCourse(exam),
  });
  const course = q.data;

  // Đồng bộ XP từ payload course-state (không leveled_up/new_badges → không ăn mừng).
  useEffect(() => {
    if (course?.xp) ingestXp(course.xp);
  }, [course?.xp, ingestXp]);

  const badges = course?.xp?.badges?.map((b) => b.id) ?? [];

  return (
    <div id="mode-course" className="flex flex-col gap-5">
      <Card className="overflow-hidden">
        <div className="flex items-start justify-between gap-3 border-b bg-gradient-to-r from-amber-50 to-orange-50 px-5 py-4 dark:from-amber-950/30 dark:to-orange-950/20">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-bold">🎓 Khóa học cá nhân hóa</h2>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Lộ trình sắp theo điểm yếu trong bài chấm của bạn. Càng luyện, khóa học càng bám sát —
              tiêu chí/dạng câu đã thành thạo sẽ <b>tự đánh dấu xong</b>.
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={() => q.refetch()} className="shrink-0 gap-1.5">
            <RotateCw className={`h-4 w-4 ${q.isFetching ? 'animate-spin' : ''}`} /> Cập nhật
          </Button>
        </div>

        {/* Chọn kỳ thi */}
        <div className="flex flex-wrap gap-2 border-b bg-muted/20 px-5 py-3">
          {EXAMS.map((e) => (
            <button
              key={e}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                e === exam
                  ? 'bg-primary text-primary-foreground shadow-sm'
                  : 'bg-background text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
              onClick={() => setExam(e)}
            >
              {examConfig(e).label}
            </button>
          ))}
        </div>

        <div className="p-5">
          {q.isLoading && <p className="text-sm text-muted-foreground">⏳ Đang dựng khóa học…</p>}
          {q.isError && <p className="text-sm text-muted-foreground">⚠️ Không tải được khóa học.</p>}
          {course && <CourseHeader course={course} badges={badges} />}
        </div>
      </Card>

      {course &&
        course.units.map((unit, ui) => (
          <Card className="overflow-hidden" key={unit.id}>
            <h3 className="flex items-center gap-2 border-b bg-muted/30 px-5 py-3 text-base font-semibold">
              <span className="text-lg" aria-hidden>
                {DIM_ICON[unit.dimension] || '•'}
              </span>
              <span className="text-xs font-bold text-muted-foreground">Chặng {ui + 1}</span>
              {unit.title}
            </h3>
            <LessonPath lessons={unit.lessons} onOpen={(id) => navigate(`/course/lesson/${id}`)} />
          </Card>
        ))}
    </div>
  );
}

function CourseHeader({ course, badges }: { course: CourseView; badges: string[] }) {
  const pct = Math.round(course.progress.pct * 100);
  return (
    <div className="flex flex-col gap-4">
      {/* XP + streak + tiến độ tổng */}
      <div className="flex flex-wrap items-center gap-4">
        <XpBar className="min-w-[220px] flex-1" />
        <StreakFlame days={course.streak.days} longest={course.streak.longest} />
        <div className="flex items-center gap-2 rounded-full border bg-muted/40 px-3 py-1.5">
          <span className="text-sm font-bold tabular-nums text-foreground">
            <NumberTicker value={pct} />%
          </span>
          <span className="text-xs text-muted-foreground">
            ({course.progress.done}/{course.progress.total} bài)
          </span>
        </div>
      </div>
      {/* Thanh tiến độ tổng */}
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-teal-500"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ type: 'spring', stiffness: 120, damping: 22 }}
        />
      </div>
      {badges.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">Huy hiệu:</span>
          <BadgeGrid earned={badges} compact />
        </div>
      )}
    </div>
  );
}

// ── Learning path: node zig-zag ──────────────────────────────────────────

function LessonPath({ lessons, onOpen }: { lessons: LessonView[]; onOpen: (id: string) => void }) {
  return (
    <div className="flex flex-col items-stretch gap-1 px-5 py-6">
      {lessons.map((ls, i) => (
        <div
          key={ls.id}
          className="flex items-center"
          style={{ justifyContent: i % 2 === 0 ? 'flex-start' : 'flex-end' }}
        >
          <LessonNode lesson={ls} index={i} onOpen={() => onOpen(ls.id)} />
        </div>
      ))}
    </div>
  );
}

const NODE_STYLE: Record<LessonStatus, string> = {
  done: 'border-emerald-400 bg-gradient-to-br from-emerald-400 to-teal-500 text-white shadow-md',
  in_progress: 'border-amber-400 bg-gradient-to-br from-amber-300 to-orange-400 text-white shadow-md',
  available: 'border-primary bg-gradient-to-br from-indigo-400 to-primary text-white shadow-md',
  locked: 'border-border bg-muted text-muted-foreground',
};

function LessonNode({ lesson, index, onOpen }: { lesson: LessonView; index: number; onOpen: () => void }) {
  const locked = lesson.status === 'locked';
  const available = lesson.status === 'available';
  const alignRight = index % 2 !== 0;
  return (
    <div
      className={`flex max-w-[70%] items-center gap-3 ${alignRight ? 'flex-row-reverse text-right' : ''}`}
    >
      <motion.button
        type="button"
        disabled={locked}
        onClick={onOpen}
        title={locked ? 'Hoàn thành bài trước để mở khóa' : lesson.title}
        whileHover={locked ? {} : { scale: 1.08 }}
        whileTap={locked ? {} : { scale: 0.94 }}
        animate={available ? { y: [0, -5, 0] } : {}}
        transition={available ? { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } : {}}
        className={`relative flex h-16 w-16 shrink-0 items-center justify-center rounded-full border-2 ${
          NODE_STYLE[lesson.status]
        } ${locked ? 'cursor-not-allowed' : 'cursor-pointer'} ${
          lesson.focus ? 'ring-4 ring-amber-300/70 ring-offset-2 ring-offset-background' : ''
        }`}
      >
        {lesson.status === 'done' ? (
          <Check className="h-7 w-7" strokeWidth={3} />
        ) : locked ? (
          <Lock className="h-6 w-6" />
        ) : lesson.status === 'in_progress' ? (
          <span className="text-xl font-bold">◐</span>
        ) : (
          <Play className="h-7 w-7 fill-current" />
        )}
        {lesson.best_score != null && (
          <span className="absolute -bottom-1 -right-1 rounded-full bg-background px-1 text-[0.6rem] font-bold tabular-nums text-foreground shadow ring-1 ring-border">
            {Math.round(lesson.best_score * 100)}%
          </span>
        )}
      </motion.button>

      <div className={`flex min-w-0 flex-col ${alignRight ? 'items-end' : ''}`}>
        {lesson.focus && (
          <Badge className="mb-1 w-fit gap-0.5 bg-amber-400 text-amber-950 hover:bg-amber-400">
            ⭐ Nên học
          </Badge>
        )}
        <span className={`truncate text-sm font-semibold ${locked ? 'text-muted-foreground' : 'text-foreground'}`}>
          {lesson.title}
        </span>
        {lesson.description && (
          <span className="line-clamp-2 text-xs text-muted-foreground">{lesson.description}</span>
        )}
      </div>
    </div>
  );
}
