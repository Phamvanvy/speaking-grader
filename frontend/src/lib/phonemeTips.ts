// Port web/js/phoneme-tips.js — tip cấu âm tiếng Việt (kiểu ELSA) + từ ví dụ để phát
// mẫu TTS trong popup luyện từ. Key = symbol chuẩn hoá backend (US, có ː); phonemeTip()
// tự thử biến thể bỏ/thêm ː. Nội dung giữ NGUYÊN VĂN từ legacy (không đổi câu chữ).

export interface PhonemeTip {
  tip: string;
  example: string;
}

const PHONEME_TIPS: Record<string, PhonemeTip> = {
  // ── Phụ âm ──
  p: { tip: 'Mím chặt hai môi rồi bật hơi mạnh ra. Không rung dây thanh.', example: 'pen' },
  b: { tip: 'Mím chặt hai môi rồi bật ra, CÓ rung dây thanh (như "b" tiếng Việt nhưng bật mạnh hơn).', example: 'bad' },
  t: { tip: 'Đặt đầu lưỡi vào lợi trên (sau răng cửa), chặn hơi rồi bật ra. Không rung dây thanh.', example: 'tea' },
  d: { tip: 'Đặt đầu lưỡi vào lợi trên, chặn hơi rồi bật ra, CÓ rung dây thanh.', example: 'did' },
  k: { tip: 'Nâng cuống lưỡi chạm vòm mềm, chặn hơi rồi bật ra. Không rung dây thanh.', example: 'cat' },
  ɡ: { tip: 'Nâng cuống lưỡi chạm vòm mềm, chặn hơi rồi bật ra, CÓ rung dây thanh.', example: 'go' },
  tʃ: { tip: 'Đầu tiên chặn hơi bằng lưỡi như /t/, rồi thả ra thành âm xát /ʃ/ ("ch" nhưng mạnh và tròn môi hơn).', example: 'chair' },
  dʒ: { tip: 'Nhấn mặt trước của lưỡi vào vòm miệng để ngăn luồng hơi, tương tự như âm /d/. Sau đó, giải phóng không khí và để âm phát ra ("j" trong "jump").', example: 'job' },
  f: { tip: 'Đặt răng trên chạm nhẹ môi dưới, thổi hơi qua khe. Không rung dây thanh (như "ph" tiếng Việt).', example: 'fan' },
  v: { tip: 'Đặt răng trên chạm nhẹ môi dưới, thổi hơi qua khe, CÓ rung dây thanh (khác /f/ ở độ rung).', example: 'van' },
  θ: { tip: 'Đưa đầu lưỡi ra GIỮA hai hàm răng, thổi hơi qua khe lưỡi–răng. Không rung dây thanh ("th" trong "think").', example: 'think' },
  ð: { tip: 'Đưa đầu lưỡi ra giữa hai hàm răng, thổi hơi và CÓ rung dây thanh ("th" trong "this").', example: 'this' },
  s: { tip: 'Đầu lưỡi gần lợi trên, đẩy hơi qua khe hẹp tạo tiếng xì. Không rung dây thanh.', example: 'see' },
  z: { tip: 'Như /s/ nhưng CÓ rung dây thanh — đặt tay lên cổ họng phải thấy rung.', example: 'zoo' },
  ʃ: { tip: 'Cong nhẹ lưỡi về sau, môi hơi tròn, đẩy hơi tạo âm "sh" (như suỵt im lặng).', example: 'she' },
  ʒ: { tip: 'Như /ʃ/ nhưng CÓ rung dây thanh (âm giữa trong "vision", "measure").', example: 'vision' },
  h: { tip: 'Thở hơi nhẹ từ cổ họng ra, không chặn ở đâu cả (như "h" tiếng Việt nhưng nhẹ hơn).', example: 'hat' },
  m: { tip: 'Mím hai môi, đẩy hơi qua MŨI, rung dây thanh.', example: 'man' },
  n: { tip: 'Đầu lưỡi chạm lợi trên, đẩy hơi qua MŨI, rung dây thanh.', example: 'no' },
  ŋ: { tip: 'Cuống lưỡi chạm vòm mềm, hơi thoát qua MŨI ("ng" tiếng Việt). KHÔNG thêm /ɡ/ hay /k/ ở cuối.', example: 'sing' },
  l: { tip: 'Đầu lưỡi chạm lợi trên, hơi thoát HAI BÊN lưỡi. Cuối từ phải giữ lưỡi chạm lợi (dark L).', example: 'let' },
  r: { tip: 'Cong đầu lưỡi về sau KHÔNG chạm vòm miệng, môi hơi tròn. Không rung lưỡi như "r" tiếng Việt.', example: 'red' },
  ɹ: { tip: 'Cong đầu lưỡi về sau KHÔNG chạm vòm miệng, môi hơi tròn. Không rung lưỡi như "r" tiếng Việt.', example: 'red' },
  w: { tip: 'Tròn môi như /u/ rồi lướt nhanh sang nguyên âm sau. Không để thành /v/ (răng không chạm môi).', example: 'we' },
  j: { tip: 'Mặt lưỡi nâng gần vòm cứng như /i/ rồi lướt nhanh sang nguyên âm sau ("y" trong "yes").', example: 'yes' },
  ɾ: { tip: 'Âm vỗ kiểu Mỹ: đầu lưỡi chạm NHANH lợi trên một cái rồi nhả ("t/d" giữa từ như "water").', example: 'water' },

  // ── Nguyên âm đơn ──
  iː: { tip: 'Kéo môi sang hai bên như cười, lưỡi nâng cao phía trước, giữ ÂM DÀI ("ee" trong "see").', example: 'see' },
  ɪ: { tip: 'Ngắn và lỏng hơn /iː/: miệng hơi mở, lưỡi thấp hơn một chút. KHÔNG kéo dài.', example: 'sit' },
  e: { tip: 'Miệng mở vừa, lưỡi ở giữa phía trước (như "e" tiếng Việt nhưng ngắn).', example: 'bed' },
  æ: { tip: 'Mở miệng RỘNG, lưỡi thấp phía trước — giữa "a" và "e" tiếng Việt ("a" trong "cat").', example: 'cat' },
  ɑː: { tip: 'Mở miệng rộng, lưỡi thấp phía sau, âm DÀI như khi bác sĩ bảo "aaa".', example: 'far' },
  ɒ: { tip: 'Môi hơi tròn, lưỡi thấp phía sau, âm NGẮN ("o" kiểu Anh trong "hot").', example: 'hot' },
  ɔː: { tip: 'Tròn môi, lưỡi lùi về sau, âm DÀI ("aw" trong "saw").', example: 'saw' },
  ʌ: { tip: 'Miệng mở vừa, lưỡi ở giữa, âm NGẮN dứt khoát ("â/ơ" ngắn — "u" trong "cup").', example: 'cup' },
  ʊ: { tip: 'Môi hơi tròn, lưỡi lùi sau, âm NGẮN và lỏng ("oo" trong "book" — không phải /uː/ dài).', example: 'book' },
  uː: { tip: 'Tròn môi chặt, lưỡi lùi sâu về sau, âm DÀI ("oo" trong "food").', example: 'food' },
  ə: { tip: 'Âm lơi (schwa): mọi cơ thả lỏng, miệng he hé, phát rất NGẮN và NHẸ — chỉ ở âm tiết không nhấn.', example: 'about' },
  ɜː: { tip: 'Lưỡi ở giữa, môi không tròn, âm DÀI ("ir" trong "bird"). Giọng Mỹ cong nhẹ lưỡi (âm r-hóa).', example: 'bird' },
  ɝ: { tip: 'Âm /ɜː/ r-hóa kiểu Mỹ: lưỡi giữa, cong nhẹ đầu lưỡi về sau suốt âm ("ir" trong "bird").', example: 'bird' },
  ɚ: { tip: 'Schwa r-hóa kiểu Mỹ, ở âm tiết KHÔNG nhấn: thả lỏng + cong nhẹ đầu lưỡi ("er" trong "teacher").', example: 'teacher' },

  // ── Nguyên âm đôi ──
  eɪ: { tip: 'Lướt từ /e/ sang /ɪ/: bắt đầu mở vừa rồi khép dần, môi kéo ngang ("ay" trong "say").', example: 'say' },
  aɪ: { tip: 'Lướt từ /a/ mở rộng sang /ɪ/: mở to rồi khép dần ("i" trong "my").', example: 'my' },
  ɔɪ: { tip: 'Lướt từ /ɔ/ (tròn môi) sang /ɪ/ (kéo ngang) — "oy" trong "boy".', example: 'boy' },
  oʊ: { tip: 'Lướt từ /o/ sang /ʊ/: môi tròn dần lại ("o" trong "go" kiểu Mỹ). Đừng phát thành "ô" phẳng.', example: 'go' },
  əʊ: { tip: 'Kiểu Anh: bắt đầu từ schwa /ə/ rồi tròn môi dần sang /ʊ/ ("o" trong "go").', example: 'go' },
  aʊ: { tip: 'Lướt từ /a/ mở rộng sang /ʊ/ tròn môi ("ow" trong "now").', example: 'now' },
  ɪə: { tip: 'Kiểu Anh: lướt từ /ɪ/ về schwa /ə/ ("ear" trong "near").', example: 'near' },
  eə: { tip: 'Kiểu Anh: lướt từ /e/ về schwa /ə/ ("air" trong "hair").', example: 'hair' },
  ʊə: { tip: 'Kiểu Anh: lướt từ /ʊ/ về schwa /ə/ ("ure" trong "sure").', example: 'sure' },
};

/** Tra tip cho 1 symbol: nguyên bản → thêm ː → bỏ ː. null = symbol lạ (không có nút 🔊). */
export function phonemeTip(symbol: string | null | undefined): PhonemeTip | null {
  if (!symbol) return null;
  const s = String(symbol);
  return PHONEME_TIPS[s] || PHONEME_TIPS[s + 'ː'] || PHONEME_TIPS[s.replace(/ː$/, '')] || null;
}
