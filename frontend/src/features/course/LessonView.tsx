// Màn hình 1 bài học (/course/lesson/:id). Render theo dimension:
//  • pronunciation: từ luyện (🔊 qua playback delegation, ⭐ mở popup luyện) + ghi âm
//    đọc các từ → /grade → tự hoàn thành theo % phoneme.
//  • rubric: tips + gợi ý/sửa lỗi từ chính bài chấm của user + đề luyện chấm thật.
//  • question_type: bài nói mẫu + thang điểm + hướng dẫn + đề luyện chấm thật.
// Hoàn thành:
//  - có lesson.practice (đề task-context) → PracticeRecorder ghi âm → POST
//    /course/lesson/:id/grade (server chấm + chuẩn hóa + tự hoàn thành + streak).
//  - không (dạng chỉ chấm bằng ảnh) → nút "Đã học xong" thủ công gọi
//    POST /course/lesson/:id/complete (score 1). Cả hai đều invalidate query khóa học.

import { useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { apiFetch } from '@/lib/api';
import { getUserId } from '@/lib/identity';
import { useUiStore } from '@/store/ui';
import { usePractice } from '@/store/practice';
import { useXp } from '@/store/xp';
import { celebrateComplete } from '@/lib/celebrate';
import { playSfx } from '@/lib/sfx';
import {
  getLesson,
  completeLesson,
  gradeLessonPractice,
  type LessonContent,
  type LessonGradeResult,
  type PracticeTask,
} from './courseApi';

// % chính xác: (ok + low-severity) / non-skipped — khớp practicePct của PracticeDialog.
function practicePct(phonemes: any[]): number | null {
  const scored = (phonemes || []).filter((p) => p.status !== 'skipped');
  if (!scored.length) return null;
  const pass = scored.filter((p) => p.status === 'ok' || p.severity === 'low').length;
  return Math.round((100 * pass) / scored.length);
}

export default function LessonView() {
  const { lessonId = '' } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const userId = getUserId();

  const q = useQuery({
    queryKey: ['course', 'lesson', lessonId, userId],
    queryFn: () => getLesson(lessonId),
    enabled: !!lessonId,
  });
  const lesson = q.data;

  async function onCompleted(score: number) {
    try {
      const res = await completeLesson(lessonId, score, lesson?.exam || 'toeic');
      qc.invalidateQueries({ queryKey: ['course'] });
      // XP/level/huy hiệu (chỉ có khi first-transition). ingest tự ăn mừng level-up/badge.
      if (res.xp) useXp.getState().ingest(res.xp);
      if (res.done) {
        // Tránh chồng hiệu ứng: chỉ confetti+SFX hoàn thành khi KHÔNG lên cấp/huy hiệu.
        if (!res.xp?.leveled_up && !res.xp?.new_badges?.length) celebrateComplete();
        toast.success(`Hoàn thành bài! 🎉 Chuỗi ${res.streak.streak_days} ngày.`);
        navigate('/course');
      } else {
        toast(`Đã lưu (${Math.round(score * 100)}%) — cần đạt ${Math.round((lesson?.done_threshold || 0.7) * 100)}% để xong.`);
      }
    } catch (e: any) {
      toast.error(`Lỗi lưu tiến độ: ${e.message || e}`);
    }
  }

  // Chấm THẬT (server đã lưu tiến độ + streak) — chỉ hiển thị kết quả & điều hướng.
  function onGraded(res: LessonGradeResult) {
    qc.invalidateQueries({ queryKey: ['course'] });
    if (res.score == null) {
      toast('Chưa chấm được tiêu chí này — hãy nói dài & rõ hơn rồi thử lại.');
      return;
    }
    const pct = Math.round(res.score * 100);
    const cr = res.progress;
    if (cr?.xp) useXp.getState().ingest(cr.xp);
    if (cr?.done) {
      if (!cr.xp?.leveled_up && !cr.xp?.new_badges?.length) celebrateComplete();
      toast.success(`Đạt ${pct}% — hoàn thành bài! 🎉 Chuỗi ${cr.streak.streak_days} ngày.`);
      navigate('/course');
    } else {
      toast(`Được ${pct}% — cần ${Math.round((lesson?.done_threshold || 0.67) * 100)}% để xong. Thử lại nhé.`);
    }
  }

  return (
    <div id="mode-course-lesson">
      <div className="card">
        <div className="result-header">
          <button className="btn btn-secondary btn-inline" onClick={() => navigate('/course')}>
            ‹ Khóa học
          </button>
          {lesson && <h2 className="course-lesson-head">{lesson.title}</h2>}
        </div>
        {q.isLoading && <p className="history-empty">⏳ Đang tải bài học…</p>}
        {q.isError && <p className="history-empty">⚠️ Không tải được bài học.</p>}
        {lesson && (
          <>
            {lesson.description && <p className="course-lesson-desc">{lesson.description}</p>}
            {lesson.dimension === 'pronunciation' && <PronBody lesson={lesson} onCompleted={onCompleted} />}
            {lesson.dimension === 'rubric' && <RubricBody lesson={lesson} onCompleted={onCompleted} onGraded={onGraded} />}
            {lesson.dimension === 'question_type' && <QtypeBody lesson={lesson} onCompleted={onCompleted} onGraded={onGraded} />}
          </>
        )}
      </div>
    </div>
  );
}

// ── Phát âm ────────────────────────────────────────────────────────────────

function PronBody({ lesson, onCompleted }: { lesson: LessonContent; onCompleted: (s: number) => void }) {
  const accent = useUiStore((s) => s.accent);
  const openPractice = usePractice((s) => s.openPractice);
  const words = lesson.words || [];
  const [recording, setRecording] = useState(false);
  const [grading, setGrading] = useState(false);
  const [status, setStatus] = useState('');
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const drill = words.map((w) => w.word).join(' ');

  function stopStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setRecording(false);
  }

  async function toggle() {
    if (grading) return;
    if (recording) {
      recRef.current?.stop();
      return;
    }
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setStatus('Không truy cập được micro — kiểm tra quyền trình duyệt.');
      return;
    }
    chunksRef.current = [];
    const rec = new MediaRecorder(streamRef.current);
    recRef.current = rec;
    rec.addEventListener('dataavailable', (e) => e.data?.size && chunksRef.current.push(e.data));
    rec.addEventListener('stop', () => {
      const mime = rec.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      stopStream();
      grade(blob, mime);
    });
    rec.start();
    setRecording(true);
    setStatus('Đang ghi âm… bấm lần nữa để dừng.');
  }

  async function grade(blob: Blob, mime: string) {
    if (!blob || blob.size < 1024) {
      setStatus('Ghi âm quá ngắn — bấm 🎙️, đọc rõ các từ, rồi bấm dừng.');
      return;
    }
    setGrading(true);
    setStatus('Đang chấm…');
    try {
      const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm';
      const fd = new FormData();
      fd.append('audio', new File([blob], `lesson-${lesson.id}.${ext}`, { type: mime }));
      fd.append('text', drill);
      fd.append('mode', 'mock_test');
      fd.append('no_ai', 'true');
      fd.append('strict', 'true');
      fd.append('accent', accent);
      if (lesson.exam === 'topik') fd.append('exam', 'topik'); // pipeline tiếng Hàn
      const res = await apiFetch('/grade', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      const ws = result?.phoneme?.score?.words || [];
      const merged = ws.flatMap((w: any) => w.phonemes || []);
      const pct = practicePct(merged);
      if (pct == null) {
        setStatus('Chưa nghe rõ — hãy đọc to, rõ từng từ rồi thử lại.');
        return;
      }
      setStatus(pct >= 80 ? `Tuyệt vời — ${pct}%! 🎉` : `Được ${pct}% — luyện thêm rồi thử lại nhé.`);
      // 1 SFX/đợt: chưa đạt ngưỡng → tiếng "sai"; đạt → confetti+SFX ở onCompleted.
      if (pct / 100 < lesson.done_threshold) playSfx('wrong');
      onCompleted(pct / 100);
    } catch (e: any) {
      setStatus(`Lỗi chấm: ${e.message || e}`);
    } finally {
      setGrading(false);
    }
  }

  return (
    <>
      <div className="course-section-label">Từ luyện tập (bấm 🔊 nghe mẫu, ⭐ để luyện riêng)</div>
      <div className="course-pron-words">
        {words.map((w) => (
          <div className="course-pron-word" key={w.word}>
            <button
              type="button"
              className="tts-play course-pron-word__play"
              data-word={w.word}
              data-ipa={w.ipa || undefined}
              title={`Nghe “${w.word}”`}
            >
              🔊
            </button>
            <span className="course-pron-word__text">
              <b>{w.word}</b> {w.ipa && <span className="course-pron-word__ipa">/{w.ipa}/</span>}
              {w.reason && <span className="course-pron-word__reason">{w.reason}</span>}
            </span>
            <button
              type="button"
              className="btn btn-secondary btn-inline"
              onClick={() => openPractice({ word: w.word, ipa: w.ipa })}
              title="Luyện & chấm riêng từ này"
            >
              ⭐ Luyện
            </button>
          </div>
        ))}
        {!words.length && <p className="history-empty">Chưa lấy được từ luyện — thử lại sau.</p>}
      </div>

      <div className="course-complete">
        <div className="course-complete__hint">
          Đọc to tất cả các từ trên, hoàn thành khi đạt {Math.round(lesson.done_threshold * 100)}% âm đúng.
        </div>
        <button
          type="button"
          className={'practice-mic' + (recording ? ' recording' : '')}
          onClick={toggle}
          disabled={grading || !words.length}
          title="Ghi âm đọc các từ"
        >
          🎙️
        </button>
        {status && <div className="course-complete__status">{status}</div>}
      </div>
    </>
  );
}

// ── Tiêu chí rubric ──────────────────────────────────────────────────────────

function RubricBody({
  lesson,
  onCompleted,
  onGraded,
}: {
  lesson: LessonContent;
  onCompleted: (s: number) => void;
  onGraded: (r: LessonGradeResult) => void;
}) {
  const tips = lesson.tips || [];
  const suggestions = lesson.learner_suggestions || [];
  const corrections = lesson.corrections || [];
  return (
    <>
      {tips.length > 0 && (
        <>
          <div className="course-section-label">Mẹo cải thiện</div>
          <ul className="course-tips">{tips.map((t, i) => <li key={i}>{t}</li>)}</ul>
        </>
      )}
      {suggestions.length > 0 && (
        <>
          <div className="course-section-label">Từ bài chấm của bạn — nên lưu ý</div>
          <ul className="course-tips course-tips--learner">{suggestions.map((t, i) => <li key={i}>{t}</li>)}</ul>
        </>
      )}
      {corrections.length > 0 && (
        <>
          <div className="course-section-label">Sửa lỗi gợi ý</div>
          <ul className="course-corrections">
            {corrections.map((c, i) => (
              <li key={i}>
                <s>{c.said}</s> → <b>{c.suggested}</b>
                {c.example && <div className="course-corrections__ex">{c.example}</div>}
              </li>
            ))}
          </ul>
        </>
      )}
      {!suggestions.length && !corrections.length && (
        <p className="course-note">
          Chưa có dữ liệu từ bài chấm cho tiêu chí này. Làm vài bài ở tab “Thi cả đề” để nhận gợi ý cá nhân hóa.
        </p>
      )}
      {lesson.practice ? (
        <PracticeRecorder lesson={lesson} practice={lesson.practice} onGraded={onGraded} />
      ) : (
        <MarkDoneButton lesson={lesson} onCompleted={onCompleted} />
      )}
    </>
  );
}

// ── Dạng câu ─────────────────────────────────────────────────────────────────

function QtypeBody({
  lesson,
  onCompleted,
  onGraded,
}: {
  lesson: LessonContent;
  onCompleted: (s: number) => void;
  onGraded: (r: LessonGradeResult) => void;
}) {
  const sample = lesson.sample_answer;
  return (
    <>
      {lesson.guidance && (
        <>
          <div className="course-section-label">Yêu cầu dạng câu</div>
          <p className="course-guidance">{lesson.guidance}</p>
        </>
      )}
      {sample ? (
        <>
          <div className="course-section-label">Bài nói mẫu {sample.target_band ? `(${sample.target_band})` : ''}</div>
          <p className="course-sample__answer">{sample.answer}</p>
          {sample.outline?.length > 0 && (
            <>
              <div className="course-section-label">Dàn ý</div>
              <ul className="course-tips">{sample.outline.map((o, i) => <li key={i}>{o}</li>)}</ul>
            </>
          )}
          {sample.highlights?.length > 0 && (
            <>
              <div className="course-section-label">Điểm nhấn nên học</div>
              <ul className="course-tips course-tips--learner">{sample.highlights.map((h, i) => <li key={i}>{h}</li>)}</ul>
            </>
          )}
        </>
      ) : (
        <p className="course-note">Chưa tạo được bài mẫu (dịch vụ AI tạm bận). Bạn vẫn có thể đọc phần yêu cầu trên.</p>
      )}
      {lesson.practice ? (
        <PracticeRecorder lesson={lesson} practice={lesson.practice} onGraded={onGraded} />
      ) : (
        <MarkDoneButton lesson={lesson} onCompleted={onCompleted} />
      )}
    </>
  );
}

// ── Đề luyện chấm thật (task-context) ────────────────────────────────────────
// Hiển thị đề luyện của lesson rubric/dạng câu (do server sinh) + ghi âm → chấm
// qua POST /course/lesson/:id/grade (server tự chuẩn hóa điểm + hoàn thành + streak).
function PracticeRecorder({
  lesson,
  practice,
  onGraded,
}: {
  lesson: LessonContent;
  practice: PracticeTask;
  onGraded: (r: LessonGradeResult) => void;
}) {
  const accent = useUiStore((s) => s.accent);
  const done = lesson.progress?.status === 'done';
  const [recording, setRecording] = useState(false);
  const [grading, setGrading] = useState(false);
  const [status, setStatus] = useState('');
  const recRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const isRead = practice.question_type === 'read_aloud';
  const imgSrc = practice.image_b64
    ? `data:${practice.image_media_type || 'image/jpeg'};base64,${practice.image_b64}`
    : null;

  function stopStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setRecording(false);
  }

  async function toggle() {
    if (grading) return;
    if (recording) {
      recRef.current?.stop();
      return;
    }
    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setStatus('Không truy cập được micro — kiểm tra quyền trình duyệt.');
      return;
    }
    chunksRef.current = [];
    const rec = new MediaRecorder(streamRef.current);
    recRef.current = rec;
    rec.addEventListener('dataavailable', (e) => e.data?.size && chunksRef.current.push(e.data));
    rec.addEventListener('stop', () => {
      const mime = rec.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      stopStream();
      grade(blob, mime);
    });
    rec.start();
    setRecording(true);
    setStatus('Đang ghi âm… bấm lần nữa để dừng.');
  }

  async function grade(blob: Blob, mime: string) {
    if (!blob || blob.size < 1024) {
      setStatus('Ghi âm quá ngắn — bấm 🎙️, trả lời rồi bấm dừng.');
      return;
    }
    setGrading(true);
    setStatus('Đang chấm…');
    try {
      const res = await gradeLessonPractice(lesson.id, blob, mime, accent);
      setStatus(res.score == null ? 'Chưa chấm được — thử nói dài & rõ hơn.' : `Điểm: ${Math.round(res.score * 100)}%.`);
      // Chưa đạt → tiếng "sai" (đạt → confetti+SFX ở onGraded).
      if (res.score != null && !res.progress?.done) playSfx('wrong');
      onGraded(res);
    } catch (e: any) {
      setStatus(`Lỗi chấm: ${e.message || e}`);
    } finally {
      setGrading(false);
    }
  }

  return (
    <div className="course-practice">
      <div className="course-section-label">
        {imgSrc
          ? 'Luyện & chấm — nhìn tranh và mô tả'
          : isRead
            ? 'Luyện & chấm — đọc to đoạn văn'
            : 'Luyện & chấm — trả lời đề dưới đây'}
      </div>
      {imgSrc && <img className="course-practice__image" src={imgSrc} alt="Ảnh đề tả tranh" />}
      {practice.provided_info && (
        <pre className="course-practice__info">{practice.provided_info}</pre>
      )}
      {isRead ? (
        <p className="course-practice__prompt course-practice__reference">{practice.reference}</p>
      ) : (
        practice.prompt && <p className="course-practice__prompt">{practice.prompt}</p>
      )}
      <div className="course-complete">
        <div className="course-complete__hint">
          {imgSrc
            ? 'Mô tả bối cảnh + người/vật + hoạt động (30-45s).'
            : isRead
              ? 'Đọc to, rõ cả đoạn.'
              : 'Trả lời bằng lời nói, đủ ý.'}{' '}
          Cần đạt {Math.round(lesson.done_threshold * 100)}% để hoàn thành.
          {done && ' ✓ Đã học xong — luyện thêm vẫn được.'}
        </div>
        <button
          type="button"
          className={'practice-mic' + (recording ? ' recording' : '')}
          onClick={toggle}
          disabled={grading}
          title="Ghi âm trả lời"
        >
          🎙️
        </button>
        {status && <div className="course-complete__status">{status}</div>}
      </div>
    </div>
  );
}

function MarkDoneButton({ lesson, onCompleted }: { lesson: LessonContent; onCompleted: (s: number) => void }) {
  const done = lesson.progress?.status === 'done';
  return (
    <div className="course-complete">
      <button
        type="button"
        className="btn btn-primary"
        onClick={() => onCompleted(1)}
        disabled={done}
      >
        {done ? '✓ Đã học xong' : 'Đánh dấu đã học xong'}
      </button>
    </div>
  );
}
