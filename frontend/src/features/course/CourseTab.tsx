// Tab "Khóa học" — giáo trình cá nhân hóa theo kết quả test. Unit → Lesson với
// trạng thái (locked/available/in_progress/done), badge "Nên học" (focus), thanh
// tiến độ tổng + streak. Server state qua TanStack Query (courseApi). Click lesson
// (nếu không khóa) → /course/lesson/:id.

import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { getUserId } from '@/lib/identity';
import { examConfig } from '@/lib/config';
import { useCourseStore, type CourseExam } from '@/store/course';
import { getCourse, type CourseView, type LessonView, type LessonStatus } from './courseApi';

const DIM_ICON: Record<string, string> = {
  pronunciation: '🗣️',
  rubric: '📋',
  question_type: '🎯',
};

const DIM_LABEL: Record<string, string> = {
  pronunciation: 'Phát âm',
  rubric: 'Tiêu chí',
  question_type: 'Dạng câu',
};

const STATUS_META: Record<LessonStatus, { label: string; cls: string }> = {
  locked: { label: '🔒 Khóa', cls: 'locked' },
  available: { label: '▶ Học ngay', cls: 'available' },
  in_progress: { label: '◐ Đang học', cls: 'in_progress' },
  done: { label: '✓ Đã xong', cls: 'done' },
};

const EXAMS: CourseExam[] = ['toeic', 'ielts', 'topik'];

export default function CourseTab() {
  const exam = useCourseStore((s) => s.exam);
  const setExam = useCourseStore((s) => s.setExam);
  const navigate = useNavigate();
  const userId = getUserId();

  const q = useQuery({
    queryKey: ['course', exam, userId],
    queryFn: () => getCourse(exam),
  });

  const course = q.data;

  return (
    <div id="mode-course">
      <div className="card">
        <div className="result-header">
          <h2>🎓 Khóa học cá nhân hóa</h2>
          <button className="btn btn-secondary btn-inline" onClick={() => q.refetch()}>
            ↻ Cập nhật
          </button>
        </div>
        <p className="course-intro">
          Lộ trình được sắp theo điểm yếu trong các bài chấm của bạn. Càng luyện & làm bài,
          khóa học càng bám sát chỗ cần cải thiện — tiêu chí/dạng câu bạn đã thành thạo trong
          bài thi sẽ <b>tự đánh dấu xong</b>.
        </p>

        <div className="course-exam-tabs">
          {EXAMS.map((e) => (
            <button
              key={e}
              className={'course-exam-tab' + (e === exam ? ' active' : '')}
              onClick={() => setExam(e)}
            >
              {examConfig(e).label}
            </button>
          ))}
        </div>

        {q.isLoading && <p className="history-empty">⏳ Đang dựng khóa học…</p>}
        {q.isError && <p className="history-empty">⚠️ Không tải được khóa học.</p>}
        {course && <CourseSummary course={course} />}
      </div>

      {course &&
        course.units.map((unit) => (
          <div className="card course-unit" key={unit.id}>
            <h3 className="course-unit__title">
              {DIM_ICON[unit.dimension] || '•'} {unit.title}
            </h3>
            <div className="course-lessons">
              {unit.lessons.map((ls) => (
                <LessonRow key={ls.id} lesson={ls} onOpen={() => navigate(`/course/lesson/${ls.id}`)} />
              ))}
            </div>
          </div>
        ))}
    </div>
  );
}

function CourseSummary({ course }: { course: CourseView }) {
  const pct = Math.round(course.progress.pct * 100);
  return (
    <div className="course-summary">
      <div className="course-progress">
        <div className="course-progress__bar">
          <div className="course-progress__fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="course-progress__label">
          {course.progress.done}/{course.progress.total} bài · {pct}%
        </div>
      </div>
      <div className="course-streak" title="Chuỗi ngày học liên tiếp">
        🔥 {course.streak.days} ngày
        {course.streak.longest > course.streak.days ? ` · kỷ lục ${course.streak.longest}` : ''}
      </div>
      <div className="course-dims">
        {['pronunciation', 'rubric', 'question_type'].map((d) => {
          const s = course.progress.by_dimension[d];
          if (!s) return null;
          return (
            <span className="course-dim" key={d}>
              {DIM_ICON[d]} {DIM_LABEL[d]} {s.done}/{s.total}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function LessonRow({ lesson, onOpen }: { lesson: LessonView; onOpen: () => void }) {
  const meta = STATUS_META[lesson.status];
  const locked = lesson.status === 'locked';
  return (
    <button
      type="button"
      className={'course-lesson' + (locked ? ' is-locked' : '') + (lesson.focus ? ' is-focus' : '')}
      disabled={locked}
      onClick={onOpen}
      title={locked ? 'Hoàn thành bài trước để mở khóa' : lesson.title}
    >
      <span className="course-lesson__main">
        <span className="course-lesson__title">{lesson.title}</span>
        {lesson.description && <span className="course-lesson__desc">{lesson.description}</span>}
      </span>
      <span className="course-lesson__side">
        {lesson.focus && <span className="course-badge focus">⭐ Nên học</span>}
        {lesson.best_score != null && (
          <span className="course-lesson__score">{Math.round(lesson.best_score * 100)}%</span>
        )}
        <span className={'course-badge ' + meta.cls}>{meta.label}</span>
      </span>
    </button>
  );
}
