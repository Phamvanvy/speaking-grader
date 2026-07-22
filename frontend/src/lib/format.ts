// Helper hiển thị dùng chung. JSX tự escape nên escapeHtml chỉ cần cho print/PDF
// (dựng chuỗi HTML thủ công — M2). Port từ web/js/render.js.

export function escapeHtml(s: unknown): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  };
  return String(s ?? '').replace(/[&<>"']/g, (c) => map[c]);
}

export function pct(x: number | null | undefined): string {
  return ((x || 0) * 100).toFixed(1) + '%';
}

// Khối "📝 Transcript" dùng chung cho kết quả exam + lịch sử. Kèm nút 🔊 "nghe lại":
// đọc CẢ transcript bằng Web Speech API (delegated .transcript-tts ở playback.ts) —
// KHÔNG dùng /tts vì transcript là cả đoạn, vượt trần độ dài của Piper. Nút chỉ hiện
// khi có transcript. Style inline để không phụ thuộc CSS parent-scoped.
export function transcriptSectionHtml(transcript: unknown): string {
  const text = String(transcript ?? '');
  const btn = text.trim()
    ? `<button type="button" class="transcript-tts" data-text="${escapeHtml(text)}"` +
      ` title="Nghe lại (máy đọc — tham khảo)" aria-label="Nghe lại transcript"` +
      ` style="margin-left:0.4rem;border:none;background:transparent;cursor:pointer;font-size:1rem;` +
      `line-height:1;vertical-align:middle;padding:0.1rem 0.3rem;border-radius:6px;">🔊</button>`
    : '';
  return `<div class="result-section"><h4>📝 Transcript${btn}</h4><p>${escapeHtml(text)}</p></div>`;
}
