// Metadata hiển thị huy hiệu (icon + nhãn + mô tả). Nguồn sự thật của việc ĐÃ mở
// khóa là backend (user_badges); file này chỉ để render. `id` khớp _BADGE_RULES
// trong src/course/xp.py.

export interface BadgeMeta {
  id: string;
  icon: string;
  label: string;
  desc: string;
}

// Thứ tự hiển thị chuẩn trong lưới huy hiệu.
export const BADGE_ORDER = [
  'first_lesson',
  'streak_3',
  'streak_7',
  'streak_30',
  'words_10',
  'words_50',
  'words_100',
  'perfect_10',
  'level_5',
  'level_10',
  'boss_1',
  'boss_5',
  'quest_1',
  'quest_5',
] as const;

const META: Record<string, BadgeMeta> = {
  first_lesson: { id: 'first_lesson', icon: '🎓', label: 'Nhập môn', desc: 'Hoàn thành bài học đầu tiên' },
  streak_3: { id: 'streak_3', icon: '🔥', label: 'Chăm 3 ngày', desc: 'Chuỗi học 3 ngày liên tiếp' },
  streak_7: { id: 'streak_7', icon: '🔥', label: 'Chăm 1 tuần', desc: 'Chuỗi học 7 ngày liên tiếp' },
  streak_30: { id: 'streak_30', icon: '🏆', label: 'Bền bỉ 30 ngày', desc: 'Chuỗi học 30 ngày liên tiếp' },
  words_10: { id: 'words_10', icon: '📖', label: '10 từ vững', desc: '10 từ khác nhau đạt độ chuẩn' },
  words_50: { id: 'words_50', icon: '📚', label: '50 từ vững', desc: '50 từ khác nhau đạt độ chuẩn' },
  words_100: { id: 'words_100', icon: '🧠', label: '100 từ vững', desc: '100 từ khác nhau đạt độ chuẩn' },
  perfect_10: { id: 'perfect_10', icon: '💯', label: 'Hoàn hảo x10', desc: '10 từ từng đạt 100%' },
  level_5: { id: 'level_5', icon: '⭐', label: 'Cấp 5', desc: 'Đạt cấp độ 5' },
  level_10: { id: 'level_10', icon: '🌟', label: 'Cấp 10', desc: 'Đạt cấp độ 10' },
  boss_1: { id: 'boss_1', icon: '👾', label: 'Hạ Boss', desc: 'Hạ Boss cuối chặng đầu tiên' },
  boss_5: { id: 'boss_5', icon: '⚔️', label: 'Thợ săn Boss', desc: 'Hạ 5 Boss cuối chặng' },
  quest_1: { id: 'quest_1', icon: '🎭', label: 'Nhiệm vụ đầu', desc: 'Hoàn thành nhiệm vụ nâng cao đầu tiên (nhập vai/truyện)' },
  quest_5: { id: 'quest_5', icon: '🎬', label: 'Người kể chuyện', desc: 'Hoàn thành 5 nhiệm vụ nâng cao' },
};

export function badgeMeta(id: string): BadgeMeta {
  return META[id] ?? { id, icon: '🏅', label: id, desc: '' };
}
