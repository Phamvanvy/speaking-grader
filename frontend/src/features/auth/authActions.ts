// Hành động auth dùng chung giữa AuthWidget và AuthDialog — port web/js/auth.js
// (completeLogin / maybeClaimHistory / doLogout / verifyAuthOnLoad).
//
// Danh tính đổi ⇒ MỌI server state đổi theo: query key có user_id nên TanStack tự
// tách cache, nhưng vẫn clear để không giữ dữ liệu của danh tính cũ trong bộ nhớ,
// và refresh savedWords (Zustand, không thuộc query cache).

import type { QueryClient } from '@tanstack/react-query';
import { apiFetch, apiUrl } from '../../lib/api';
import { getAnonUserId, isLoggedIn, regenerateAnonUserId } from '../../lib/identity';
import { useAuthStore } from '../../store/auth';
import { useSavedWords } from '../../store/savedWords';

export interface AuthResponse {
  token: string;
  user_id: string;
  username?: string;
}

/** Sau khi đổi trạng thái đăng nhập: bỏ cache server state cũ + nạp lại từ đã lưu. */
export function refreshAfterAuthChange(qc: QueryClient): void {
  qc.clear();
  useSavedWords.getState().refresh().catch(() => {
    /* server tắt / lỗi mạng → giữ nguyên, tab Saved tự báo lỗi */
  });
}

/**
 * Gộp lịch sử ẩn danh vào tài khoản vừa đăng nhập (chỉ hỏi khi UUID ẩn danh thật
 * sự có dữ liệu). Probe /history/list phải đi KHÔNG kèm Bearer — backend
 * (_resolve_read_user_id) ưu tiên session token và sẽ bỏ qua user_id ẩn danh.
 */
export async function maybeClaimHistory(anonId: string): Promise<void> {
  if (!anonId || !isLoggedIn()) return;

  let hasData = false;
  try {
    const res = await fetch(apiUrl(`/history/list?user_id=${encodeURIComponent(anonId)}&limit=1`));
    if (res.ok) hasData = ((await res.json()).total || 0) > 0;
  } catch {
    return; // không probe được → không chặn đăng nhập
  }
  if (!hasData) return;

  const ok = confirm(
    'Máy này đang có lịch sử chấm bài chưa gắn tài khoản.\n\n' +
      'Chuyển toàn bộ lịch sử đó vào tài khoản vừa đăng nhập?',
  );
  if (!ok) return;

  try {
    const res = await apiFetch('/auth/claim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ anon_user_id: anonId }),
      noRetry: true,
    });
    if (!res.ok) return;
    const d = await res.json();
    // Đã gộp xong → UUID ẩn danh mới để id cũ không bị gộp lại lần sau.
    regenerateAnonUserId();
    useAuthStore.getState().refresh();
    if ((d.records || 0) + (d.words || 0) > 0) {
      alert(
        `Đã chuyển ${d.records} bản ghi lịch sử` +
          (d.words ? ` và ${d.words} từ đã lưu` : '') +
          ' vào tài khoản.',
      );
    }
  } catch {
    /* bỏ qua — đăng nhập vẫn thành công, gộp thử lại lần sau */
  }
}

/** Hoàn tất đăng nhập (dùng chung password + Google). */
export async function completeLogin(data: AuthResponse, qc: QueryClient): Promise<void> {
  // Ghi nhớ UUID ẩn danh TRƯỚC khi lưu auth (getUserId sẽ đổi sang id tài khoản).
  const anonId = getAnonUserId();
  useAuthStore.getState().login(data);
  await maybeClaimHistory(anonId);
  refreshAfterAuthChange(qc);
}

export async function doLogout(qc: QueryClient): Promise<void> {
  try {
    await apiFetch('/auth/logout', { method: 'POST', noRetry: true });
  } catch {
    /* vẫn xoá phía client dù server lỗi */
  }
  useAuthStore.getState().logout();
  refreshAfterAuthChange(qc);
}

/** Khôi phục phiên khi mở lại trang: token hết hạn/bị thu hồi → dọn auth cục bộ. */
export async function verifyAuthOnLoad(qc: QueryClient): Promise<void> {
  if (!isLoggedIn()) return;
  try {
    const res = await apiFetch('/auth/me', { noRetry: true });
    if (res.status === 401) {
      useAuthStore.getState().logout();
      refreshAfterAuthChange(qc);
    }
  } catch {
    /* offline — giữ nguyên phiên, thử lại lần mở sau */
  }
}
