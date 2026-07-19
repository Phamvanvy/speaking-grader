import { Link } from 'react-router-dom';
import { AuroraText } from '../../components/ui/aurora-text';
import { InteractiveHoverButton } from '../../components/ui/interactive-hover-button';

/**
 * Landing/hero — dựng ở /home, KHÔNG thay route '/' (vẫn là GradingTab).
 *
 * Lý do tách route: '/' là màn Grading đang trong giai đoạn cutover React↔legacy.
 * Đổi '/' lúc này sẽ kéo theo bookmark, start_url của PWA và nav tab. Sau khi
 * cutover xong chỉ cần đổi một dòng trong App.tsx để flip '/' về đây.
 *
 * Trang này CHỈ điều hướng — không gọi API, không hiển thị số liệu. Cố ý: chưa có
 * endpoint thống kê nào, nên mọi con số ở đây sẽ là số bịa.
 *
 * Style bằng utility Tailwind thay vì class legacy: đây là trang mới hoàn toàn,
 * không cần coexist với CSS cũ. Lưu ý `preflight: false` → không có reset toàn cục,
 * nên mọi khoảng cách/cỡ chữ ở đây đều đặt tường minh.
 */

const ENTRIES = [
  {
    to: '/',
    icon: '📝',
    title: 'Chấm bài lẻ / cả lớp',
    desc: 'Nộp một file ghi âm hoặc cả thư mục của lớp, nhận điểm theo từng tiêu chí kèm phân tích phát âm.',
  },
  {
    to: '/exam',
    icon: '📄',
    title: 'Thi cả đề',
    desc: 'Làm trọn một đề TOEIC Speaking hoặc TOPIK nói, tính điểm tổng theo trọng số từng phần.',
  },
  {
    to: '/history',
    icon: '🕘',
    title: 'Lịch sử',
    desc: 'Nghe lại bài đã nộp, đối chiếu điểm và xem tiến bộ qua từng lần luyện.',
  },
  {
    to: '/saved',
    icon: '📚',
    title: 'Từ đã lưu',
    desc: 'Những từ bạn hay phát âm lệch được gom lại để luyện riêng tới khi chuẩn.',
  },
];

export default function HomePage() {
  return (
    <div className="py-6">
      <section className="mx-auto max-w-2xl px-2 text-center">
        <p className="m-0 text-3xl font-bold leading-tight sm:text-4xl">
          Luyện nói cho tới khi <AuroraText>phát âm chuẩn</AuroraText>
        </p>
        <p className="mx-auto mt-4 max-w-xl text-base leading-relaxed text-muted-foreground">
          Chấm điểm nói TOEIC và TOPIK tự động: nhận diện lời nói, đối chiếu tới từng âm vị, và chỉ
          đúng chỗ bạn đọc lệch.
        </p>
        <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
          <Link to="/">
            <InteractiveHoverButton>Chấm bài ngay</InteractiveHoverButton>
          </Link>
          <Link
            to="/exam"
            className="rounded-lg border border-border px-5 py-2.5 text-sm font-medium no-underline transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            Thi thử cả đề
          </Link>
        </div>
      </section>

      <section className="mt-12 grid gap-4 sm:grid-cols-2">
        {ENTRIES.map((e) => (
          <Link
            key={e.to}
            to={e.to}
            className="group relative overflow-hidden rounded-xl border border-border bg-card p-5 no-underline transition-shadow hover:shadow-md"
          >
            <span className="text-2xl leading-none" aria-hidden="true">
              {e.icon}
            </span>
            <span className="mt-3 block text-base font-semibold text-card-foreground">
              {e.title}
            </span>
            <span className="mt-1.5 block text-sm leading-relaxed text-muted-foreground">
              {e.desc}
            </span>
          </Link>
        ))}
      </section>
    </div>
  );
}
