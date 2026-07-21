import { useSound } from '../lib/sfx';

// Nút bật/tắt tiếng SFX game hóa (đặt cạnh ThemeToggle). Dùng class .theme-toggle
// để tái dùng style nút góc sẵn có; đẩy vị trí qua .sound-toggle.
export default function SoundToggle() {
  const muted = useSound((s) => s.muted);
  const toggle = useSound((s) => s.toggle);
  return (
    <button
      className="theme-toggle sound-toggle"
      onClick={toggle}
      title={muted ? 'Bật âm thanh' : 'Tắt âm thanh'}
      aria-label={muted ? 'Bật âm thanh' : 'Tắt âm thanh'}
    >
      {muted ? '🔇' : '🔊'}
    </button>
  );
}
