// Trạng thái popup luyện từ (ELSA-style). Mở được từ MỌI tab: click .practice-open trên
// DOM do legacy/render.ts inject (grading/exam/history), hàng gợi ý, bảng Từ đã lưu, hoặc
// review-toast. Delegated handler (savedInterop.ts) set store → <PracticeDialog> mở.
// Logic ghi âm/chấm nằm trong component (dùng hook); store chỉ giữ open + data.

import { create } from 'zustand';

export interface PracticeData {
  word: string;
  ipa?: string | null;
  accuracy?: number | null;
  skip_reason?: string | null;
  phonemes?: any[];
  /** _w: chỉ số từ trong cụm (gộp phoneme nhiều từ) — do gradePracticeAttempt gắn. */
}

interface PracticeState {
  open: boolean;
  data: PracticeData | null;
  openPractice: (data: PracticeData) => void;
  setData: (data: PracticeData) => void;
  close: () => void;
}

export const usePractice = create<PracticeState>((set) => ({
  open: false,
  data: null,
  openPractice: (data) => {
    if (!data || !data.word) return;
    set({ open: true, data });
  },
  setData: (data) => set({ data }),
  close: () => set({ open: false, data: null }),
}));
