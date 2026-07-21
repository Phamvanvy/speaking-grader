// API layer cho tab "Khóa học" — bám apiGet/apiPostForm (lib/api.ts) + getUserId.
// Backend: src/course/ (GET /course, GET /course/lesson/{id}, POST .../complete).

import { apiGet, apiPostForm } from '@/lib/api';
import { getUserId } from '@/lib/identity';

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

export interface UnitView {
  id: string;
  title: string;
  dimension: Dimension;
  lessons: LessonView[];
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
}

export function getCourse(exam: string): Promise<CourseView> {
  const uid = getUserId();
  return apiGet<CourseView>(`/course?user_id=${encodeURIComponent(uid)}&exam=${encodeURIComponent(exam)}`);
}

export function getLesson(lessonId: string): Promise<LessonContent> {
  const uid = getUserId();
  return apiGet<LessonContent>(`/course/lesson/${encodeURIComponent(lessonId)}?user_id=${encodeURIComponent(uid)}`);
}

export interface CompleteResult {
  lesson_id: string;
  done: boolean;
  progress: { status: LessonStatus; best_score: number | null; attempts: number };
  streak: { streak_days: number; longest_streak: number; total_completed: number };
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
