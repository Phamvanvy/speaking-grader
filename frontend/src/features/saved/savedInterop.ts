// Cầu nối delegated: DOM do legacy/render.ts inject bằng chuỗi HTML (bảng lỗi grading/
// exam/history) phát ra click .word-bookmark / .practice-open → bơm vào store React.
// Gắn 1 lần ở document (giống installPlaybackHandlers). Port practice.js phần delegated.

import { usePractice } from '@/store/practice';
import { useSavedWords, syncBookmarkButtons } from '@/store/savedWords';

let installed = false;

export function installSavedInterop() {
  if (installed) return;
  installed = true;

  document.addEventListener('click', (e) => {
    const t = e.target;
    if (!(t instanceof Element)) return;

    // ☆/★ trên bảng lỗi → toggle lưu, KHÔNG mở popup.
    const star = t.closest<HTMLElement>('.word-bookmark');
    if (star && star.dataset.practice) {
      e.preventDefault();
      e.stopPropagation();
      let data: any;
      try {
        data = JSON.parse(star.dataset.practice);
      } catch {
        return;
      }
      const sw = useSavedWords.getState();
      const done = sw.has(data.word)
        ? sw.remove(data.word)
        : sw.add({ word: data.word, ipa: data.ipa, phonemes: data.phonemes, accuracy: data.accuracy });
      done.then(() => syncBookmarkButtons(data.word)).catch((err) => alert(`Lỗi lưu từ: ${err.message || err}`));
      return;
    }

    // Click vào từ (.practice-open) → mở popup luyện. Bỏ qua nút audio lồng bên trong.
    const opener = t.closest<HTMLElement>('.practice-open');
    if (opener && opener.dataset.practice) {
      if (t.closest('.tts-play') || t.closest('.phoneme-play')) return;
      e.preventDefault();
      try {
        usePractice.getState().openPractice(JSON.parse(opener.dataset.practice));
      } catch {
        /* attr hỏng */
      }
    }
  });
}
