import { useUiStore } from '../store/ui';

// Nút góc phải bật/tắt dark mode (giữ nguyên hành vi form.js: class body.dark).
export default function ThemeToggle() {
  const theme = useUiStore((s) => s.theme);
  const toggle = useUiStore((s) => s.toggleTheme);
  return (
    <button
      className="theme-toggle"
      onClick={toggle}
      title="Toggle dark mode"
      aria-label="Toggle dark mode"
    >
      {theme === 'dark' ? '☀️' : '🌙'}
    </button>
  );
}
