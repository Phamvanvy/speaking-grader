import { NavLink, Routes, Route, Navigate } from 'react-router-dom';
import ThemeToggle from './components/ThemeToggle';
import SoundToggle from './components/SoundToggle';
import AuthWidget from './components/AuthWidget';
import GradingTab from './features/grading/GradingTab';
import HomePage from './features/home/HomePage';
import ExamTab from './features/exam/ExamTab';
import HistoryTab from './features/history/HistoryTab';
import SavedTab from './features/saved/SavedTab';
import CourseTab from './features/course/CourseTab';
import LessonView from './features/course/LessonView';
import BossView from './features/course/BossView';
import AccountPage from './features/account/AccountPage';
import PracticeDialog from './features/saved/PracticeDialog';
import AddWordsDialog, { useAddWords } from './features/saved/AddWordsDialog';
import AuthDialog from './features/auth/AuthDialog';
import { Toaster } from './components/ui/sonner';
import { TooltipProvider } from './components/ui/tooltip';

// 4 tab top-level (khớp switchMode legacy) map sang route thật (History API) — giữ
// đúng path cũ để link/reload hoạt động qua catch-all SPA của FastAPI.
const TABS = [
  { to: '/exam', label: '📄 Thi cả đề (cá nhân)', end: false },
  { to: '/course', label: '🎓 Khóa học', end: false },
  { to: '/history', label: '🕘 Lịch sử', end: false },
  { to: '/saved', label: '📚 Từ đã lưu', end: false },
  { to: '/grade', label: '📝 Chấm bài lẻ / cả lớp', end: false },
];

export default function App() {
  const openAddWords = useAddWords((s) => s.setOpen);
  return (
    <TooltipProvider delayDuration={200}>
      <ThemeToggle />
      <SoundToggle />
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
      <button
        className="addwords-btn"
        id="addwords-btn"
        title="Thêm từ vựng để luyện tập"
        aria-label="Thêm từ vựng để luyện tập"
        onClick={() => openAddWords(true)}
      >
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
          {/* '/' = "Thi cả đề" (luồng chính). Grading chuyển sang path riêng /grade;
              '/' cũ trỏ Grading nên redirect để bookmark/PWA start_url không gãy. */}
          <Route path="/" element={<Navigate to="/exam" replace />} />
          <Route path="/grade" element={<GradingTab />} />
          <Route path="/home" element={<HomePage />} />
          {/* Router legacy dùng path lồng như /exam/toeic/set2/q/3 — bắt hết về ExamTab. */}
          <Route path="/exam/*" element={<ExamTab />} />
          <Route path="/history/*" element={<HistoryTab />} />
          <Route path="/saved/*" element={<SavedTab />} />
          {/* Khóa học: danh sách + màn hình 1 bài (:lessonId). */}
          <Route path="/course" element={<CourseTab />} />
          <Route path="/course/lesson/:lessonId" element={<LessonView />} />
          <Route path="/course/unit/:unitId/boss" element={<BossView />} />
          {/* Ngoài 4 tab — vào từ widget danh tính góc trên. */}
          <Route path="/account" element={<AccountPage />} />
          <Route path="*" element={<Navigate to="/exam" replace />} />
        </Routes>
      </div>

      {/* Popup dùng chung (mở từ mọi tab) */}
      <PracticeDialog />
      <AddWordsDialog />
      <AuthDialog />

      {/* Toast dùng chung (sonner) — review-toast + thông báo lưu từ, chấm điểm… */}
      <Toaster position="bottom-right" richColors closeButton />
    </TooltipProvider>
  );
}
