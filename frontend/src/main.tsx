import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

// CSS gốc dùng chung với legacy — import theo ĐÚNG thứ tự cascade của index.html cũ
// (base → components → theme-dark → phoneme → exam → history → practice → auth).
// Single source of truth: sửa CSS ở web/css/ áp cho cả legacy lẫn app React.
import '../../web/css/base.css';
import '../../web/css/components.css';
import '../../web/css/theme-dark.css';
import '../../web/css/phoneme.css';
import '../../web/css/exam.css';
import '../../web/css/history.css';
import '../../web/css/practice.css';
import '../../web/css/auth.css';

// Tailwind + shadcn tokens SAU cùng — layer utility/component nằm trên cascade legacy.
// preflight tắt (tailwind.config.js) nên KHÔNG reset element M1–M3.
import './styles/tailwind.css';

// TanStack Query = chủ sở hữu duy nhất mọi server state. Giữ dữ liệu "tươi" hợp lý:
// history/saved cần refetch khi quay lại tab; kết quả chấm giữ lâu (đọc lại cho print).
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false, // retry mạng đã do apiFetch xử lý; không nhân đôi
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
});

// Delegated handlers (1 lần): accent-select (đổi giọng → re-render) + ▶/🔊 phoneme.
import { installAccentDelegation } from './store/ui';
import { installPlaybackHandlers } from './features/grading/playback';
installAccentDelegation();
installPlaybackHandlers();

// M4 — Từ đã lưu / luyện tập:
//  • savedWords store (import → bơm {has} vào renderer legacy cho sao ☆ trên bảng lỗi)
//    + nạp cache 1 lần lúc mở app (sao hiện đúng trạng thái ngay).
//  • savedInterop: bắc cầu click .word-bookmark / .practice-open trên DOM legacy → store.
//  • reviewToast: mồi lịch nhắc ôn (sonner) + sync cài đặt từ server.
import { useSavedWords } from './store/savedWords';
import { installSavedInterop } from './features/saved/savedInterop';
import { installReviewToast } from './features/saved/reviewToast';
installSavedInterop();
useSavedWords.getState().refresh().catch(() => { /* server tắt → coi như chưa lưu gì */ });
installReviewToast();

// M5 cutover — dọn cache của SW legacy (web/sw.js, CACHE_NAME "sg-shell-v*"). SW mới
// (Workbox, cùng URL /sw.js) đã thay thế SW cũ, nhưng cleanupOutdatedCaches chỉ dọn
// precache của chính Workbox nên cache legacy còn lại sẽ chiếm chỗ vô ích.
if ('caches' in window) {
  caches
    .keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith('sg-shell-')).map((k) => caches.delete(k))))
    .catch(() => { /* storage bị chặn → bỏ qua */ });
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, '')}>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
