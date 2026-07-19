// Port từ web/js/state.js — danh tính user (ẩn danh ↔ tài khoản) + opt-in lịch sử.
// Thuần localStorage, không phụ thuộc React → dùng được cả trong apiClient lẫn store.
//
// Hai lớp danh tính:
//  1. UUID ẩn danh (USER_ID_KEY): sinh 1 lần/trình duyệt, cách ly mềm lịch sử theo
//     máy — KHÔNG auth. Giữ để tương thích ngược khi CHƯA đăng nhập.
//  2. Tài khoản (AUTH_KEY = {token, user_id, username}): đăng nhập để lấy user_id
//     CỐ ĐỊNH gắn tài khoản → đồng bộ lịch sử qua nhiều thiết bị.

export const API_URL_KEY = 'toeic-grader-api-url';
export const USER_ID_KEY = 'speaking-grader-user-id';
export const AUTH_KEY = 'speaking-grader-auth';
export const HISTORY_OPT_KEY = 'speaking-grader-save-history';
export const ACCENT_KEY = 'pron_accent';

export interface AuthInfo {
  token: string;
  user_id: string;
  username?: string;
}

function newUuid(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  // Fallback uuid-shaped khi thiếu crypto.randomUUID (context không secure) —
  // server validate [A-Za-z0-9_-]{1,64} nên format phải chuẩn.
  return ('' + 1e7 + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, (c) =>
    ((+c) ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> ((+c) / 4)))).toString(16),
  );
}

/** UUID ẩn danh của trình duyệt này (dùng khi CHƯA đăng nhập, và làm nguồn "claim"). */
export function getAnonUserId(): string {
  let id = localStorage.getItem(USER_ID_KEY);
  if (!id) {
    id = newUuid();
    localStorage.setItem(USER_ID_KEY, id);
  }
  return id;
}

/** Sau /auth/claim, sinh UUID ẩn danh MỚI để id cũ (đã nhập tài khoản) không bị gộp lại. */
export function regenerateAnonUserId(): string {
  const id = newUuid();
  localStorage.setItem(USER_ID_KEY, id);
  return id;
}

export function authState(): AuthInfo | null {
  try {
    return JSON.parse(localStorage.getItem(AUTH_KEY) || 'null');
  } catch {
    return null;
  }
}
export function isLoggedIn(): boolean {
  const a = authState();
  return !!(a && a.token);
}
export function authToken(): string | null {
  const a = authState();
  return a ? a.token : null;
}
export function setAuth(obj: AuthInfo): void {
  localStorage.setItem(AUTH_KEY, JSON.stringify(obj));
}
export function clearAuth(): void {
  localStorage.removeItem(AUTH_KEY);
}

/** user_id "đang hoạt động": tài khoản nếu đã đăng nhập, ngược lại ẩn danh. */
export function getUserId(): string {
  const a = authState();
  return a && a.user_id ? a.user_id : getAnonUserId();
}

// Opt-out privacy: tắt → không gửi user_id → server không lưu gì. Default BẬT.
export function historySaveEnabled(): boolean {
  return localStorage.getItem(HISTORY_OPT_KEY) !== 'false';
}
export function setHistorySaveEnabled(on: boolean): void {
  localStorage.setItem(HISTORY_OPT_KEY, on ? 'true' : 'false');
}
