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
