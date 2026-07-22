// API layer cho tab "Khóa học" — bám apiGet/apiPostForm (lib/api.ts) + getUserId.
// Backend: src/course/ (GET /course/state, GET /course/lesson/{id}/content, POST .../complete).
// Lưu ý: API KHÔNG dùng bare /course và /course/lesson/{id} — đó là route SPA.

import { apiFetch, apiGet, apiPostForm } from '@/lib/api';
import { getUserId } from '@/lib/identity';
import type { XpState } from '@/store/xp';

export type Dimension = 'pronunciation' | 'rubric' | 'question_type';
export type LessonStatus = 'locked' | 'available' | 'in_progress' | 'done';

export interface LessonView {
  id: string;
  title: string;
  dimension: Dimension;
  target: string;
  description: string;
  est_minutes: number;
  status: LessonStatus;
  priority: number;
  focus: boolean;
  best_score: number | null;
  attempts: number;
}

// Boss cuối chặng (Phase 3A) — node bonus sau các lesson của Unit.
export interface BossNode {
  id: string;
  title: string;
  status: LessonStatus; // locked (còn lesson chưa done) | available | done
  threshold: number;
  best_score: number | null;
}

export interface UnitView {
  id: string;
  title: string;
  dimension: Dimension;
  lessons: LessonView[];
  boss?: BossNode;
}

export interface CourseView {
  exam: string;
  progress: {
    done: number;
    total: number;
    pct: number;
    by_dimension: Record<string, { done: number; total: number }>;
  };
  streak: {
    days: number;
    longest: number;
    last_active_day: string | null;
    total_completed: number;
  };
  units: UnitView[];
  // Trạng thái XP/level/huy hiệu (gamification) — kèm sẵn để đỡ round-trip.
  // undefined/enabled=false khi tắt cờ COURSE_XP_ENABLED.
  xp?: XpState;
}

export interface PronWord {
  word: string;
  ipa: string;
  phoneme: string;
  reason: string | null;
}

export interface SampleAnswer {
  answer: string;
  outline: string[];
  highlights: string[];
  target_band: string;
}

export interface PracticeTask {
  question_type: string;
  prompt: string;
  reference: string;
  provided_info: string;
  target_criterion?: string; // chỉ lesson rubric
  image_b64?: string; // dạng tả tranh: ảnh đề (inline base64)
  image_media_type?: string;
}

export interface LessonContent {
  id: string;
  title: string;
  dimension: Dimension;
  target: string;
  exam: string;
  description: string;
  est_minutes: number;
  done_threshold: number;
  progress: { status: LessonStatus; best_score: number | null; attempts: number } | null;
  // pronunciation
  phonemes?: string[];
  words?: PronWord[];
  // rubric
  tips?: string[];
  learner_suggestions?: string[];
  corrections?: { said: string; suggested: string; example: string | null }[];
  // question_type
  sample_answer?: SampleAnswer | null;
  scale_description?: string;
  guidance?: string;
  // rubric + question_type: đề luyện chấm thật (null nếu dạng chỉ chấm bằng ảnh)
  practice?: PracticeTask | null;
}

export function getCourse(exam: string): Promise<CourseView> {
  const uid = getUserId();
  return apiGet<CourseView>(`/course/state?user_id=${encodeURIComponent(uid)}&exam=${encodeURIComponent(exam)}`);
}

export function getLesson(lessonId: string): Promise<LessonContent> {
  const uid = getUserId();
  return apiGet<LessonContent>(`/course/lesson/${encodeURIComponent(lessonId)}/content?user_id=${encodeURIComponent(uid)}`);
}

export interface CompleteResult {
  lesson_id: string;
  done: boolean;
  progress: { status: LessonStatus; best_score: number | null; attempts: number };
  streak: { streak_days: number; longest_streak: number; total_completed: number };
  // Chỉ có khi lesson CHUYỂN sang done lần đầu (first-transition award).
  xp?: XpState;
  new_badges?: string[];
}

/** score đã CHUẨN HÓA 0-1 (server so ngưỡng theo dimension). */
export function completeLesson(lessonId: string, score: number, exam: string): Promise<CompleteResult> {
  const uid = getUserId();
  const fd = new FormData();
  fd.append('user_id', uid);
  fd.append('score', String(score));
  fd.append('exam', exam);
  return apiPostForm<CompleteResult>(`/course/lesson/${encodeURIComponent(lessonId)}/complete`, fd);
}

// ── Boss cuối chặng (Phase 3A) ───────────────────────────────────────────

export interface BossContent {
  boss_id: string;
  unit_id: string;
  title: string;
  exam: string;
  dimension: Dimension;
  threshold: number;
  reference_text: string;
  words: PronWord[];
  best_score: number | null;
  done: boolean;
}

export interface BossCompleteResult {
  boss_id: string;
  done: boolean;
  score: number;
  best_score?: number | null;
  xp?: XpState; // chỉ khi lần đầu hạ Boss (first-transition award)
  new_badges?: string[];
}

export function getBossContent(unitId: string): Promise<BossContent> {
  const uid = getUserId();
  return apiGet<BossContent>(
    `/course/unit/${encodeURIComponent(unitId)}/boss/content?user_id=${encodeURIComponent(uid)}`,
  );
}

/** score đã CHUẨN HÓA 0-1 (client chấm qua /grade; server gate + clamp + ngưỡng). */
export function completeBoss(unitId: string, score: number): Promise<BossCompleteResult> {
  const uid = getUserId();
  const fd = new FormData();
  fd.append('user_id', uid);
  fd.append('score', String(score));
  return apiPostForm<BossCompleteResult>(
    `/course/unit/${encodeURIComponent(unitId)}/boss/complete`,
    fd,
  );
}

// ── Quest nhập vai (Role-play, Phase 3B) — bonus, LLM sinh kịch bản ─────────

// Một Quest trong danh sách (thẻ trên tab Khóa học).
export interface QuestListItem {
  quest_id: string;
  kind: 'roleplay' | 'story';
  topic: string;
  title: string;
  cleared: boolean;
  best_score: number | null;
}

export interface QuestList {
  exam: string;
  quests: QuestListItem[];
}

export interface RolePlayTurn {
  npc: string;
  expected_user: string;
  hint: string;
}

export interface RolePlayScript {
  quest_id: string;
  kind: 'roleplay';
  exam: string;
  topic: string;
  threshold: number;
  scenario: string;
  role_user: string;
  role_npc: string;
  turns: RolePlayTurn[];
  best_score: number | null;
  cleared: boolean;
}

export interface StorySegment {
  text: string;
}

export interface StoryQuest {
  quest_id: string;
  kind: 'story';
  exam: string;
  topic: string;
  threshold: number;
  title: string;
  segments: StorySegment[];
  best_score: number | null;
  cleared: boolean;
}

export interface QuestCompleteResult {
  quest_id: string;
  kind: string;
  done: boolean;
  score: number;
  best_score?: number | null;
  xp?: XpState; // chỉ khi lần đầu hoàn thành (first-transition award)
  new_badges?: string[];
}

export function getQuests(exam: string): Promise<QuestList> {
  const uid = getUserId();
  return apiGet<QuestList>(
    `/course/quests?user_id=${encodeURIComponent(uid)}&exam=${encodeURIComponent(exam)}`,
  );
}

/** Kịch bản Role-play; null khi backend không dựng được (LLM lỗi/chủ đề sai). */
export function getRoleplayQuest(exam: string, topic: string): Promise<RolePlayScript | null> {
  const uid = getUserId();
  return apiGet<RolePlayScript | null>(
    `/course/quest/roleplay?user_id=${encodeURIComponent(uid)}&exam=${encodeURIComponent(exam)}&topic=${encodeURIComponent(topic)}`,
  );
}

/** Truyện đọc-to; null khi backend không dựng được (LLM lỗi/chủ đề sai). */
export function getStoryQuest(exam: string, topic: string): Promise<StoryQuest | null> {
  const uid = getUserId();
  return apiGet<StoryQuest | null>(
    `/course/quest/story?user_id=${encodeURIComponent(uid)}&exam=${encodeURIComponent(exam)}&topic=${encodeURIComponent(topic)}`,
  );
}

/** score đã CHUẨN HÓA 0-1 (client chấm trung bình các lượt; server clamp + ngưỡng). */
export function completeQuest(
  questId: string,
  kind: string,
  score: number,
): Promise<QuestCompleteResult> {
  const uid = getUserId();
  const fd = new FormData();
  fd.append('user_id', uid);
  fd.append('quest_id', questId);
  fd.append('kind', kind);
  fd.append('score', String(score));
  return apiPostForm<QuestCompleteResult>('/course/quest/complete', fd);
}

export interface LessonGradeResult {
  // score: điểm lesson chuẩn hóa 0-1 (null nếu bài chấm thiếu tiêu chí đích).
  score: number | null;
  // progress: kết quả mark_lesson_complete (null nếu score null); done + streak.
  progress: CompleteResult | null;
  // result: output chấm đầy đủ (transcript/scores) — hiện chưa render chi tiết.
  result: any;
}

/** Chấm THẬT lesson rubric/dạng câu qua đề luyện task-context (server tự hoàn thành). */
export async function gradeLessonPractice(
  lessonId: string,
  blob: Blob,
  mime: string,
  accent: string,
): Promise<LessonGradeResult> {
  const uid = getUserId();
  const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm';
  const fd = new FormData();
  fd.append('audio', new File([blob], `lesson-${lessonId}.${ext}`, { type: mime }));
  fd.append('user_id', uid);
  fd.append('accent', accent);
  const res = await apiFetch(`/course/lesson/${encodeURIComponent(lessonId)}/grade`, {
    method: 'POST',
    body: fd,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = j.detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json() as Promise<LessonGradeResult>;
}
