// API layer — thay cho monkey-patch window.fetch (state.js) + fetchWithRetry (http.js).
//
// MỌI call tới backend đi qua apiClient: tự gắn Authorization: Bearer (khi đăng nhập),
// timeout từng lần + retry 429/502/503/504 (admission control, xem src/admission.py).
// TanStack Query gọi qua apiClient này — KHÔNG patch window.fetch nữa.
//
// Media/resource gán vào DOM (<audio src>, link tải, ảnh, print/PDF) KHÔNG set được
// header → dùng authedResourceUrl để đính token qua ?token= (quy tắc thống nhất, xem plan).

import { authToken } from './identity';

// Dev: VITE_API_BASE=http://127.0.0.1:8000 (CORS đã bật ở backend). Prod: rỗng
// (same-origin, FastAPI serve build). Bỏ dấu '/' cuối để nối path an toàn.
const RAW_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export function apiBase(): string {
  return RAW_BASE || window.location.origin.replace(/\/$/, '');
}

/** URL đầy đủ cho một path API ('/health' → 'http://.../health'). */
export function apiUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return apiBase() + (path.startsWith('/') ? path : '/' + path);
}

/**
 * URL cho tài nguyên gán vào DOM (src/href) hoặc mở context mới — đính token qua
 * query param vì các ngữ cảnh này không mang được header Authorization.
 */
export function authedResourceUrl(path: string): string {
  const url = apiUrl(path);
  const token = authToken();
  if (!token) return url;
  return url + (url.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(token);
}

const RETRY = {
  attemptTimeoutMs: 240000, // 1 lần gửi tối đa 4 phút (mock_test + hàng đợi dài là hợp lệ)
  totalTimeoutMs: 300000, // tổng (kể cả chờ retry) tối đa 5 phút
  maxBackoffMs: 30000,
};

export interface ApiOptions extends RequestInit {
  /** Tắt retry (mặc định BẬT cho GET/POST idempotent-an-toàn phía server). */
  noRetry?: boolean;
}

function withAuth(headers: Headers): Headers {
  const token = authToken();
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', 'Bearer ' + token);
  }
  return headers;
}

/**
 * fetch có timeout + retry khi server quá tải. Trả về Response thô (caller tự
 * .json()/.blob()). Ném lỗi chỉ khi mạng chết hẳn/hết tổng thời gian.
 */
export async function apiFetch(path: string, options: ApiOptions = {}): Promise<Response> {
  const url = apiUrl(path);
  const { noRetry, headers: initHeaders, ...rest } = options;
  const headers = withAuth(new Headers(initHeaders || {}));
  const started = Date.now();
  let attempt = 0;
  for (;;) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), RETRY.attemptTimeoutMs);
    let res: Response | null = null;
    let err: unknown = null;
    try {
      res = await fetch(url, { ...rest, headers, signal: controller.signal });
    } catch (e) {
      err = e; // lỗi mạng hoặc abort do timeout
    } finally {
      clearTimeout(timer);
    }
    const retryable = res
      ? res.status === 429 || res.status === 502 || res.status === 503 || res.status === 504
      : true;
    if (noRetry || !retryable) {
      if (res) return res;
      throw err || new Error('Server không phản hồi — thử lại sau.');
    }

    attempt++;
    // Ưu tiên Retry-After của server; không có → exponential backoff + jitter.
    let delayMs = Math.min(2 ** attempt * 2000, RETRY.maxBackoffMs);
    const ra = res && res.headers.get('Retry-After');
    if (ra && !isNaN(parseFloat(ra))) delayMs = parseFloat(ra) * 1000;
    delayMs += Math.random() * 1000;

    if (Date.now() - started + delayMs > RETRY.totalTimeoutMs) {
      if (res) return res; // hết kiên nhẫn → trả response lỗi cuối cùng
      throw err || new Error('Server không phản hồi — thử lại sau.');
    }
    await new Promise((r) => setTimeout(r, delayMs));
  }
}

/** GET JSON tiện dụng cho TanStack Query. Ném lỗi khi !res.ok. */
export async function apiGet<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const res = await apiFetch(path, { ...options, method: 'GET' });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

/** POST JSON body → JSON response. */
export async function apiPostJson<T>(path: string, body: unknown, options: ApiOptions = {}): Promise<T> {
  const headers = new Headers(options.headers || {});
  headers.set('Content-Type', 'application/json');
  const res = await apiFetch(path, { ...options, method: 'POST', body: JSON.stringify(body), headers });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

/** POST FormData (upload audio/ảnh) → JSON. Không set Content-Type để browser tự thêm boundary. */
export async function apiPostForm<T>(path: string, form: FormData, options: ApiOptions = {}): Promise<T> {
  const res = await apiFetch(path, { ...options, method: 'POST', body: form });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}
