// Port từ web/js/state.js — cấu hình theo kỳ thi. Mọi khác biệt TOEIC/IELTS/TOPIK
// gom về một chỗ (tránh if/else rải rác). Dùng SỐ (overallMax/criterionMax) thay
// vì chuỗi '/200' để dễ thêm TOEFL/VSTEP sau này.

export type ExamId = 'toeic' | 'ielts' | 'topik';

export interface QuestionType {
  value: string;
  label: string;
  /** Ô nhập nào HIỆN cho dạng câu này (khớp display_inputs backend). */
  uses: Array<'reference' | 'image' | 'prompt'>;
  /** Chỉ cần MỘT trong các input này là coi như "có đề" (khớp required_inputs). */
  required?: Array<'reference' | 'image' | 'prompt'>;
}

export interface ExamConfig {
  label: string;
  scoreField: string;
  overallLabel: string;
  overallLabelVi: string;
  overallMax: number;
  criterionMax: number;
  lang: 'en' | 'ko';
  questionTypes: QuestionType[];
}

export const EXAM_CONFIG: Record<ExamId, ExamConfig> = {
  toeic: {
    label: 'TOEIC',
    scoreField: 'estimated_toeic_score',
    overallLabel: 'Estimated TOEIC Speaking Score',
    overallLabelVi: 'Điểm TOEIC Speaking (ước tính)',
    overallMax: 200,
    criterionMax: 3,
    lang: 'en',
    questionTypes: [
      { value: '', label: 'Auto-detect', uses: ['reference', 'image', 'prompt'] },
      { value: 'read_aloud', label: 'Read Aloud', uses: ['reference'], required: ['reference'] },
      { value: 'describe_picture', label: 'Describe Picture', uses: ['image'], required: ['image'] },
      { value: 'respond_questions', label: 'Respond to Questions', uses: ['prompt'], required: ['prompt'] },
      { value: 'respond_with_info', label: 'Respond with Info', uses: ['prompt', 'image'], required: ['prompt'] },
      { value: 'express_opinion', label: 'Express Opinion', uses: ['prompt'], required: ['prompt'] },
    ],
  },
  ielts: {
    label: 'IELTS',
    scoreField: 'estimated_ielts_band',
    overallLabel: 'Estimated IELTS Band',
    overallLabelVi: 'IELTS band (ước tính)',
    overallMax: 9,
    criterionMax: 9,
    lang: 'en',
    // Không có "Auto-detect": Part 1 vs Part 3 không phân biệt được → luôn gửi rõ.
    questionTypes: [
      { value: 'part1_interview', label: 'Part 1 — Interview', uses: ['prompt'], required: ['prompt'] },
      { value: 'part2_long_turn', label: 'Part 2 — Long turn (cue card)', uses: ['prompt'], required: ['prompt'] },
      { value: 'part3_discussion', label: 'Part 3 — Discussion', uses: ['prompt'], required: ['prompt'] },
    ],
  },
  topik: {
    label: 'TOPIK',
    scoreField: 'estimated_topik_score',
    overallLabel: 'Estimated TOPIK Speaking Score',
    overallLabelVi: 'Điểm TOPIK 말하기 (ước tính)',
    overallMax: 200,
    criterionMax: 5, // rubric NIIED: mỗi tiêu chí 0-5 (khác thang 3 của TOEIC)
    lang: 'ko',
    questionTypes: [
      { value: 'read_aloud', label: '낭독 — Đọc to (luyện tập)', uses: ['reference'], required: ['reference'] },
      { value: 'q1_answer_question', label: '문항 1 — Trả lời câu hỏi', uses: ['prompt'], required: ['prompt'] },
      { value: 'q2_role_play', label: '문항 2 — Nhìn tranh, thực hiện vai', uses: ['prompt', 'image'], required: ['prompt', 'image'] },
      { value: 'q3_picture_story', label: '문항 3 — Nhìn tranh, kể chuyện', uses: ['prompt', 'image'], required: ['prompt', 'image'] },
      { value: 'q4_complete_dialogue', label: '문항 4 — Hoàn thành hội thoại', uses: ['prompt'], required: ['prompt'] },
      { value: 'q5_interpret_data', label: '문항 5 — Diễn giải tư liệu', uses: ['prompt', 'image'], required: ['prompt', 'image'] },
      { value: 'q6_present_opinion', label: '문항 6 — Trình bày ý kiến', uses: ['prompt'], required: ['prompt'] },
    ],
  },
};

export function examConfig(exam: string): ExamConfig {
  return EXAM_CONFIG[exam as ExamId] || EXAM_CONFIG.toeic;
}

export type Accent = 'default' | 'gb' | 'us';
export const VALID_ACCENTS: Accent[] = ['default', 'gb', 'us'];

/** TOPIK chấm tiếng Hàn: nhiều chỗ render/TTS rẽ nhánh theo "từ này là Hangul?". */
export function hasHangul(s: string | null | undefined): boolean {
  return /[가-힣]/.test(s || '');
}
