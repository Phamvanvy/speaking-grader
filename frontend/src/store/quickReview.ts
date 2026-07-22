// Trạng thái phiên "Luyện nhanh" — luyện liên tiếp N từ trong 1 vòng chơi có nhịp.
// Mirror cấu trúc practice.ts (Zustand, store chỉ giữ state; logic thu âm/chấm ở
// component qua useWordGrader). Kết quả lưu THEO VỊ TRÍ trong queue để combo/tổng
// kết tính thuần từ mảng (tất định, không trạng thái ẩn).

import { create } from 'zustand';
import type { SavedWord } from './savedWords';

/** Ngưỡng "đúng" cho combo — khớp ngưỡng ăn mừng của popup luyện (pct>=80). */
export const PASS_PCT = 80;

export interface ReviewResult {
  pct: number;
  prevScore: number | null; // last_score trước phiên (để tính sao tăng thêm)
}

interface QuickReviewState {
  open: boolean;
  queue: SavedWord[];
  index: number; // từ đang luyện
  results: (ReviewResult | null)[]; // theo vị trí trong queue
  phase: 'play' | 'summary';
  start: (queue: SavedWord[]) => void;
  /** Ghi kết quả cho từ ĐANG luyện (overwrite nếu chấm lại). */
  record: (pct: number) => void;
  /** Sang từ kế; hết queue → chuyển sang màn tổng kết. */
  advance: () => void;
  close: () => void;
}

export const useQuickReview = create<QuickReviewState>((set, get) => ({
  open: false,
  queue: [],
  index: 0,
  results: [],
  phase: 'play',

  start: (queue) => {
    if (!queue.length) return;
    set({ open: true, queue, index: 0, results: queue.map(() => null), phase: 'play' });
  },

  record: (pct) => {
    const { queue, index, results } = get();
    const cur = queue[index];
    if (!cur) return;
    const next = results.slice();
    next[index] = { pct, prevScore: cur.last_score ?? null };
    set({ results: next });
  },

  advance: () => {
    const { index, queue } = get();
    if (index + 1 < queue.length) set({ index: index + 1 });
    else set({ phase: 'summary' });
  },

  close: () => set({ open: false, queue: [], index: 0, results: [], phase: 'play' }),
}));

// ── Helper thuần cho combo + tổng kết (tính từ results) ──────────────────

/** Combo hiện tại = chuỗi "đúng" liên tiếp kết thúc ở vị trí `upto` (bao gồm). */
export function comboAt(results: (ReviewResult | null)[], upto: number): number {
  let c = 0;
  for (let i = upto; i >= 0; i--) {
    const r = results[i];
    if (r && r.pct >= PASS_PCT) c++;
    else break;
  }
  return c;
}

/** Combo cao nhất đạt được trong cả phiên. */
export function maxCombo(results: (ReviewResult | null)[]): number {
  let best = 0;
  let run = 0;
  for (const r of results) {
    if (r && r.pct >= PASS_PCT) run++;
    else run = 0;
    if (run > best) best = run;
  }
  return best;
}

/** Số từ đạt "đúng" (pct>=80). */
export function correctCount(results: (ReviewResult | null)[]): number {
  return results.filter((r) => r && r.pct >= PASS_PCT).length;
}
