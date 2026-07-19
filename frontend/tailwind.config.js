import animate from 'tailwindcss-animate';

/**
 * Tailwind cấu hình để COEXIST với ~2.068 dòng CSS kế thừa từ bản vanilla
 * (src/styles/legacy/*), không viết lại.
 *
 * - `preflight: false` + `container: false`: KHÔNG reset element toàn cục và KHÔNG sinh
 *   `.container` (legacy đã có `.container` riêng) → tránh phá M1–M3 đã port. box-sizing
 *   border-box đã set global ở base.css nên component shadcn vẫn đúng.
 * - `darkMode: ['selector', 'body.dark']`: theme legacy toggle `document.body.classList('dark')`
 *   (store/ui.ts) → biến `dark:` utility + token shadcn (định nghĩa dưới `body.dark`) khớp cùng cơ chế.
 * - Prefix KHÔNG dùng: shadcn component sinh sẵn class không prefix; utility Tailwind mới
 *   không đụng tên class legacy (legacy dùng semantic class như .btn/.card, không dùng utility).
 *
 * @type {import('tailwindcss').Config}
 */
export default {
  darkMode: ['selector', 'body.dark'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  corePlugins: {
    preflight: false,
    container: false,
  },
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        // Magic UI: registry của họ phát Tailwind v4 (`@theme inline` trong CSS),
        // v3 không hiểu → keyframes bị drop âm thầm. Port tay sang đây khi add
        // component Magic UI nào có animation. Xem src/components/ui/marquee.tsx.
        marquee: {
          from: { transform: 'translateX(0)' },
          to: { transform: 'translateX(calc(-100% - var(--gap)))' },
        },
        'marquee-vertical': {
          from: { transform: 'translateY(0)' },
          to: { transform: 'translateY(calc(-100% - var(--gap)))' },
        },
        aurora: {
          '0%': { backgroundPosition: '0% 50%', transform: 'rotate(-5deg) scale(0.9)' },
          '25%': { backgroundPosition: '50% 100%', transform: 'rotate(5deg) scale(1.1)' },
          '50%': { backgroundPosition: '100% 50%', transform: 'rotate(-3deg) scale(0.95)' },
          '75%': { backgroundPosition: '50% 0%', transform: 'rotate(3deg) scale(1.05)' },
          '100%': { backgroundPosition: '0% 50%', transform: 'rotate(-5deg) scale(0.9)' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        marquee: 'marquee var(--duration) infinite linear',
        'marquee-vertical': 'marquee-vertical var(--duration) linear infinite',
        aurora: 'aurora 8s ease-in-out infinite alternate',
      },
    },
  },
  plugins: [animate],
};
