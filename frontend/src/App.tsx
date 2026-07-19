import { NavLink, Routes, Route, Navigate } from 'react-router-dom';
import ThemeToggle from './components/ThemeToggle';
import AuthWidget from './components/AuthWidget';
import GradingTab from './features/grading/GradingTab';
import ExamTab from './features/exam/ExamTab';
import HistoryTab from './features/history/HistoryTab';
import SavedTab from './features/saved/SavedTab';

// 4 tab top-level (khớp switchMode legacy) map sang route thật (History API) — giữ
// đúng path cũ để link/reload hoạt động qua catch-all SPA của FastAPI.
const TABS = [
  { to: '/', label: '📝 Chấm bài lẻ / cả lớp', end: true },
  { to: '/exam', label: '📄 Thi cả đề (cá nhân)', end: false },
  { to: '/history', label: '🕘 Lịch sử', end: false },
  { to: '/saved', label: '📚 Từ đã lưu', end: false },
];

export default function App() {
  return (
    <>
      <ThemeToggle />
      <a
        className="feedback-btn"
        href="https://www.facebook.com/myengbuddy/"
        target="_blank"
        rel="noopener"
        title="Góp ý, báo lỗi hoặc đề xuất cải tiến"
        aria-label="Góp ý, báo lỗi hoặc đề xuất cải tiến"
      >
        💬
      </a>
      <button className="addwords-btn" id="addwords-btn" title="Thêm từ vựng để luyện tập" aria-label="Thêm từ vựng để luyện tập">
        📚
      </button>
      <AuthWidget />

      <div className="container">
        <h1>🎤 Speaking Grader</h1>

        <nav className="mode-tabs">
          {TABS.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              end={t.end}
              className={({ isActive }) => 'mode-tab' + (isActive ? ' active' : '')}
            >
              {t.label}
            </NavLink>
          ))}
        </nav>

        <Routes>
          <Route path="/" element={<GradingTab />} />
          {/* Router legacy dùng path lồng như /exam/toeic/set2/q/3 — bắt hết về ExamTab. */}
          <Route path="/exam/*" element={<ExamTab />} />
          <Route path="/history/*" element={<HistoryTab />} />
          <Route path="/saved/*" element={<SavedTab />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </>
  );
}
