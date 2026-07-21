// Trạng thái nhẹ cho tab "Khóa học": kỳ thi đang chọn (nhớ giữa các lần vào tab).
// Server state (giáo trình/tiến độ) do TanStack Query sở hữu — store này chỉ giữ
// lựa chọn UI (mirror cách store/practice.ts giữ open/data).

import { create } from 'zustand';

export type CourseExam = 'toeic' | 'ielts';

const EXAM_KEY = 'course-exam';

function initialExam(): CourseExam {
  const v = localStorage.getItem(EXAM_KEY);
  return v === 'ielts' ? 'ielts' : 'toeic';
}

interface CourseState {
  exam: CourseExam;
  setExam: (exam: CourseExam) => void;
}

export const useCourseStore = create<CourseState>((set) => ({
  exam: initialExam(),
  setExam: (exam) => {
    localStorage.setItem(EXAM_KEY, exam);
    set({ exam });
  },
}));
