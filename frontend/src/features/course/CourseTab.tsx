// Tab "Khóa học" — giáo trình cá nhân hóa dạng LEARNING PATH (kiểu Duolingo/ELSA).
// Header chỉ còn tiến độ tổng khóa học (cấp độ/XP/streak/huy hiệu đã chuyển sang trang
// Tài khoản — thuộc tính của người dùng). Mỗi Unit là 1 chặng; lessons là node tròn
// trên đường zig-zag theo trạng thái (locked/available/in_progress/done) + badge "Nên
// học" (focus). Server state qua TanStack Query (courseApi); XP vẫn kèm trong payload
// /course/state → ingest vào useXp (không round-trip thêm) để trang Tài khoản luôn mới.

import { Fragment, useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { motion } from 'motion/react';
import { RotateCw, Check, Lock, Play, ChevronDown, Drama, BookOpen, type LucideIcon } from 'lucide-react';
import { NumberTicker } from '@/components/ui/number-ticker';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { getUserId } from '@/lib/identity';
import { examConfig, COURSE_BOSS_ENABLED, COURSE_QUEST_ROLEPLAY, COURSE_QUEST_STORY } from '@/lib/config';
import { useCourseStore, type CourseExam } from '@/store/course';
import { useXp } from '@/store/xp';
import { getCourse, getQuests, type CourseView, type LessonView, type LessonStatus, type QuestListItem, type UnitView } from './courseApi';

const DIM_ICON: Record<string, string> = {
  pronunciation: '🗣️',
  rubric: '📋',
  question_type: '🎯',
};

const EXAMS: CourseExam[] = ['toeic', 'ielts', 'topik'];

// Tóm tắt trạng thái 1 Chặng để hiện badge trên header accordion (khỏi bung mới thấy tiến độ).
function unitSummary(unit: UnitView) {
  const total = unit.lessons.length;
  const done = unit.lessons.filter((l) => l.status === 'done').length;
  const allDone = total > 0 && done === total;
  const allLocked = total > 0 && unit.lessons.every((l) => l.status === 'locked');
  return { total, done, allDone, allLocked };
}

// Chặng "đang học" mở sẵn: unit đầu tiên có bài available/in_progress; fallback unit chưa xong đầu tiên.
function activeUnitId(units: UnitView[]): string | undefined {
  const inProgress = units.find((u) => u.lessons.some((l) => l.status === 'in_progress' || l.status === 'available'));
  if (inProgress) return inProgress.id;
  const notDone = units.find((u) => !unitSummary(u).allDone);
  return (notDone ?? units[0])?.id;
}

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

  // Accordion: chỉ Chặng đang học mở sẵn; các Chặng khác gập cho tab ngắn lại.
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  useEffect(() => {
    if (course?.units?.length) {
      const id = activeUnitId(course.units);
      setExpanded(id ? new Set([id]) : new Set());
    }
    // reset khi đổi kỳ thi (units mới) — bám vào chuỗi id đơn định.
  }, [course?.units.map((u) => u.id).join('|')]);

  const toggleUnit = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  // Đồng bộ XP từ payload course-state (không leveled_up/new_badges → không ăn mừng).
  useEffect(() => {
    if (course?.xp) ingestXp(course.xp);
  }, [course?.xp, ingestXp]);

  return (
    <div id="mode-course" className="flex flex-col gap-5">
      <Card className="overflow-hidden">
        <div className="flex items-start justify-between gap-3 border-b bg-gradient-to-r from-amber-50 to-orange-50 px-5 py-4 dark:from-amber-950/30 dark:to-orange-950/20">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-bold">🎓 Khóa học cá nhân hóa</h2>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Lộ trình sắp theo điểm yếu trong bài chấm của bạn — càng luyện càng bám sát.
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
          {course && <CourseHeader course={course} />}
        </div>
      </Card>

      {course &&
        course.units.map((unit, ui) => {
          const open = expanded.has(unit.id);
          const { done, total, allDone, allLocked } = unitSummary(unit);
          return (
            <Card className="overflow-hidden" key={unit.id}>
              <button
                type="button"
                onClick={() => toggleUnit(unit.id)}
                aria-expanded={open}
                className={`flex w-full items-center gap-2 px-5 py-3 text-left text-base font-semibold transition-colors hover:bg-muted/50 ${
                  open ? 'border-b bg-muted/30' : ''
                }`}
              >
                <ChevronDown
                  className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform ${open ? '' : '-rotate-90'}`}
                />
                <span className="text-lg" aria-hidden>
                  {DIM_ICON[unit.dimension] || '•'}
                </span>
                <span className="text-xs font-bold text-muted-foreground">Chặng {ui + 1}</span>
                <span className="min-w-0 truncate">{unit.title}</span>
                {/* Tóm tắt tiến độ ngay trên header (không cần bung) */}
                <span
                  className={`ml-auto shrink-0 rounded-full px-2 py-0.5 text-xs font-bold tabular-nums ${
                    allDone
                      ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                      : allLocked
                        ? 'bg-muted text-muted-foreground'
                        : 'bg-primary/15 text-primary'
                  }`}
                >
                  {allDone ? '✓ Xong' : allLocked ? '🔒' : `${done}/${total}`}
                </span>
              </button>
              {open && (
                <LessonPath
                  unit={unit}
                  onOpen={(id) => navigate(`/course/lesson/${id}`)}
                  onOpenBoss={() => navigate(`/course/unit/${unit.id}/boss`)}
                />
              )}
            </Card>
          );
        })}

      {course && (COURSE_QUEST_ROLEPLAY || COURSE_QUEST_STORY) && (
        <QuestSection
          exam={exam}
          userId={userId}
          onOpen={(kind, topic) => navigate(`/course/quest/${kind}/${topic}`)}
        />
      )}
    </div>
  );
}

// ── Nhiệm vụ nâng cao (Role-play/Story Quest, Phase 3B/3C) — BONUS, tự ẩn nếu rỗng ──
function QuestSection({
  exam,
  userId,
  onOpen,
}: {
  exam: string;
  userId: string;
  onOpen: (kind: 'roleplay' | 'story', topic: string) => void;
}) {
  const q = useQuery({
    queryKey: ['course', 'quests', exam, userId],
    queryFn: () => getQuests(exam),
  });
  // Lọc theo cờ từng loại (mỗi loại toggle riêng).
  const quests = (q.data?.quests ?? []).filter(
    (item) =>
      (item.kind === 'roleplay' && COURSE_QUEST_ROLEPLAY) ||
      (item.kind === 'story' && COURSE_QUEST_STORY),
  );
  if (!quests.length) return null; // kỳ thi chưa hỗ trợ / tắt cả hai cờ → ẩn hẳn

  return (
    <Card className="overflow-hidden">
      <h3 className="flex items-center gap-2 border-b bg-gradient-to-r from-fuchsia-50 to-rose-50 px-5 py-3 text-base font-semibold dark:from-fuchsia-950/30 dark:to-rose-950/20">
        <Drama className="h-5 w-5 text-fuchsia-500" aria-hidden />
        Nhiệm vụ nâng cao
        <span className="text-xs font-normal text-muted-foreground">· nhập vai &amp; truyện · thưởng XP</span>
      </h3>
      <div className="grid grid-cols-1 gap-3 p-5 sm:grid-cols-2">
        {quests.map((quest) => (
          <QuestCard key={quest.quest_id} quest={quest} onOpen={() => onOpen(quest.kind, quest.topic)} />
        ))}
      </div>
    </Card>
  );
}

const QUEST_META: Record<
  QuestListItem['kind'],
  { Icon: LucideIcon; subtitle: string; gradient: string; hover: string }
> = {
  roleplay: {
    Icon: Drama,
    subtitle: 'Hội thoại nhập vai — nhấn để bắt đầu',
    gradient: 'from-fuchsia-400 to-rose-500',
    hover: 'hover:border-fuchsia-400',
  },
  story: {
    Icon: BookOpen,
    subtitle: 'Truyện đọc-to — nhấn để bắt đầu',
    gradient: 'from-sky-400 to-indigo-500',
    hover: 'hover:border-sky-400',
  },
};

function QuestCard({ quest, onOpen }: { quest: QuestListItem; onOpen: () => void }) {
  const meta = QUEST_META[quest.kind];
  const Icon = meta.Icon;
  return (
    <button
      type="button"
      onClick={onOpen}
      className={`group flex items-center gap-3 rounded-xl border bg-card p-3 text-left transition-colors hover:bg-accent ${meta.hover}`}
    >
      <span
        className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br ${meta.gradient} text-white shadow-sm`}
      >
        {quest.cleared ? <Check className="h-6 w-6" strokeWidth={3} /> : <Icon className="h-6 w-6" />}
      </span>
      <span className="flex min-w-0 flex-col">
        <span className="truncate text-sm font-semibold text-foreground">{quest.title}</span>
        <span className="text-xs text-muted-foreground">
          {quest.cleared
            ? `Đã hoàn thành${quest.best_score != null ? ` · ${Math.round(quest.best_score * 100)}%` : ''}`
            : meta.subtitle}
        </span>
      </span>
    </button>
  );
}

function CourseHeader({ course }: { course: CourseView }) {
  const pct = Math.round(course.progress.pct * 100);
  return (
    <div className="flex flex-wrap items-center gap-4">
      {/* Tiến độ tổng khóa học — đặc thù tab, không phải game hóa (cấp độ nằm ở trang Tài khoản). */}
      <div className="flex items-center gap-2 rounded-full border bg-muted/40 px-3 py-1.5">
        <span className="text-sm font-bold tabular-nums text-foreground">
          <NumberTicker value={pct} />%
        </span>
        <span className="text-xs text-muted-foreground">
          ({course.progress.done}/{course.progress.total} bài)
        </span>
      </div>
    </div>
  );
}

// ── Learning path: hàng NGANG cuộn ngang (co chiều cao so với zig-zag dọc) ──
// Mỗi bài là 1 node tròn + nhãn ngắn bên dưới; các node nối bằng đoạn gạch ngang
// canh giữa tâm vòng tròn (mt-7 = ½ chiều cao h-14). Mô tả chuyển vào tooltip.

function LessonPath({
  unit,
  onOpen,
  onOpenBoss,
}: {
  unit: UnitView;
  onOpen: (id: string) => void;
  onOpenBoss: () => void;
}) {
  const lessons = unit.lessons;
  const showBoss = COURSE_BOSS_ENABLED && unit.boss;
  return (
    <div className="overflow-x-auto px-4 py-5">
      {/* w-full + connector flex-1 → path giãn đều lấp hết bề ngang (hết khoảng trống phải);
          mobile hẹp thì node shrink-0 tràn ra và cuộn ngang. */}
      <div className="flex w-full items-start">
        {lessons.map((ls, i) => (
          <Fragment key={ls.id}>
            {i > 0 && <PathConnector done={lessons[i - 1].status === 'done'} />}
            <LessonNode lesson={ls} onOpen={() => onOpen(ls.id)} />
          </Fragment>
        ))}
        {showBoss && (
          <>
            <PathConnector done={lessons.every((l) => l.status === 'done')} />
            <BossNodeView boss={unit.boss!} onOpen={onOpenBoss} />
          </>
        )}
      </div>
    </div>
  );
}

// Đoạn nối giữa 2 node, canh tâm vòng tròn (mt-7). Giãn đều (flex-1) để lấp bề ngang;
// có min-width để không biến mất khi cuộn. Xanh nếu bài trước đã xong.
function PathConnector({ done }: { done: boolean }) {
  return (
    <span
      aria-hidden
      className={`mt-7 h-0.5 min-w-4 flex-1 rounded-full sm:min-w-8 ${done ? 'bg-emerald-400' : 'bg-border'}`}
    />
  );
}

// Node Boss cuối chặng (👾) — style riêng nổi bật; locked đến khi mọi lesson done.
function BossNodeView({ boss, onOpen }: { boss: NonNullable<UnitView['boss']>; onOpen: () => void }) {
  const locked = boss.status === 'locked';
  const done = boss.status === 'done';
  return (
    <div className="flex w-20 shrink-0 flex-col items-center gap-1.5">
      <motion.button
        type="button"
        disabled={locked}
        onClick={onOpen}
        title={locked ? 'Hoàn thành tất cả bài trong chặng để mở Boss' : boss.title}
        whileHover={locked ? {} : { scale: 1.08 }}
        whileTap={locked ? {} : { scale: 0.94 }}
        animate={!locked && !done ? { scale: [1, 1.06, 1] } : {}}
        transition={!locked && !done ? { duration: 1.4, repeat: Infinity, ease: 'easeInOut' } : {}}
        className={`relative flex h-14 w-14 items-center justify-center rounded-2xl border-2 text-2xl ${
          done
            ? 'border-emerald-400 bg-gradient-to-br from-emerald-400 to-teal-500 text-white shadow-md'
            : locked
              ? 'cursor-not-allowed border-border bg-muted text-muted-foreground'
              : 'cursor-pointer border-rose-400 bg-gradient-to-br from-rose-400 to-fuchsia-600 text-white shadow-lg ring-4 ring-rose-300/50'
        }`}
      >
        {done ? <Check className="h-7 w-7" strokeWidth={3} /> : locked ? <Lock className="h-6 w-6" /> : '👾'}
        {boss.best_score != null && (
          <span className="absolute -bottom-1 -right-1 rounded-full bg-background px-1 text-[0.6rem] font-bold tabular-nums text-foreground shadow ring-1 ring-border">
            {Math.round(boss.best_score * 100)}%
          </span>
        )}
      </motion.button>
      <span className={`text-center text-xs font-bold ${locked ? 'text-muted-foreground' : 'text-rose-500'}`}>
        {done ? 'Đã hạ Boss' : locked ? 'Boss 🔒' : '👾 Boss'}
      </span>
    </div>
  );
}

const NODE_STYLE: Record<LessonStatus, string> = {
  done: 'border-emerald-400 bg-gradient-to-br from-emerald-400 to-teal-500 text-white shadow-md',
  in_progress: 'border-amber-400 bg-gradient-to-br from-amber-300 to-orange-400 text-white shadow-md',
  available: 'border-primary bg-gradient-to-br from-indigo-400 to-primary text-white shadow-md',
  locked: 'border-border bg-muted text-muted-foreground',
};

function LessonNode({ lesson, onOpen }: { lesson: LessonView; onOpen: () => void }) {
  const locked = lesson.status === 'locked';
  const available = lesson.status === 'available';
  // Tooltip gộp tiêu đề + mô tả (mô tả không còn hiển thị inline để co chiều cao).
  const tip = locked
    ? 'Hoàn thành bài trước để mở khóa'
    : lesson.description
      ? `${lesson.title} — ${lesson.description}`
      : lesson.title;
  return (
    <div className="flex w-20 shrink-0 flex-col items-center gap-1.5">
      <motion.button
        type="button"
        disabled={locked}
        onClick={onOpen}
        title={tip}
        whileHover={locked ? {} : { scale: 1.08 }}
        whileTap={locked ? {} : { scale: 0.94 }}
        animate={available ? { y: [0, -5, 0] } : {}}
        transition={available ? { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } : {}}
        className={`relative flex h-14 w-14 shrink-0 items-center justify-center rounded-full border-2 ${
          NODE_STYLE[lesson.status]
        } ${locked ? 'cursor-not-allowed' : 'cursor-pointer'} ${
          lesson.focus ? 'ring-4 ring-amber-300/70 ring-offset-2 ring-offset-background' : ''
        }`}
      >
        {lesson.status === 'done' ? (
          <Check className="h-6 w-6" strokeWidth={3} />
        ) : locked ? (
          <Lock className="h-5 w-5" />
        ) : lesson.status === 'in_progress' ? (
          <span className="text-lg font-bold">◐</span>
        ) : (
          <Play className="h-6 w-6 fill-current" />
        )}
        {lesson.best_score != null && (
          <span className="absolute -bottom-1 -right-1 rounded-full bg-background px-1 text-[0.6rem] font-bold tabular-nums text-foreground shadow ring-1 ring-border">
            {Math.round(lesson.best_score * 100)}%
          </span>
        )}
      </motion.button>

      {lesson.focus && (
        <Badge className="gap-0.5 px-1.5 py-0 text-[0.65rem] bg-amber-400 text-amber-950 hover:bg-amber-400">
          ⭐ Nên học
        </Badge>
      )}
      <span
        className={`line-clamp-2 text-center text-xs font-semibold leading-tight ${
          locked ? 'text-muted-foreground' : 'text-foreground'
        }`}
      >
        {lesson.title}
      </span>
    </div>
  );
}
