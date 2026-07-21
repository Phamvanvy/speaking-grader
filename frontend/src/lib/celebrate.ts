// Hiệu ứng ăn mừng (confetti) + SFX cho các mốc game hóa. canvas-confetti bundle
// gọn (offline OK). Tôn trọng prefers-reduced-motion (confetti tự tắt), SFX vẫn
// theo cờ tắt tiếng của sfx.ts.

import confetti from 'canvas-confetti';
import { playSfx } from './sfx';

const reduce = () =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Nổ confetti nhỏ tại giữa-dưới màn hình (đạt điểm tốt / hoàn thành nhẹ). */
export function burstConfetti(): void {
  if (reduce()) return;
  confetti({
    particleCount: 70,
    spread: 65,
    startVelocity: 42,
    origin: { y: 0.72 },
    disableForReducedMotion: true,
  });
}

/** Ăn mừng lớn: vài đợt bắn từ hai bên (level-up / 100%). */
export function bigCelebrate(): void {
  if (reduce()) return;
  const end = Date.now() + 700;
  const fire = () => {
    confetti({ particleCount: 40, angle: 60, spread: 55, origin: { x: 0, y: 0.7 }, disableForReducedMotion: true });
    confetti({ particleCount: 40, angle: 120, spread: 55, origin: { x: 1, y: 0.7 }, disableForReducedMotion: true });
    if (Date.now() < end) requestAnimationFrame(fire);
  };
  fire();
}

// ── Combo confetti + SFX (dùng ở call site cho gọn) ──
export function celebrateGood(): void {
  playSfx('correct');
  burstConfetti();
}
export function celebratePerfect(): void {
  playSfx('complete');
  bigCelebrate();
}
export function celebrateComplete(): void {
  playSfx('complete');
  burstConfetti();
}
export function celebrateLevelUp(): void {
  playSfx('levelup');
  bigCelebrate();
}
export function celebrateBadge(): void {
  playSfx('badge');
  burstConfetti();
}
