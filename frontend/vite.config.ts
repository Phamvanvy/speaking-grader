import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import { fileURLToPath, URL } from 'node:url';

// Build ra web/dist (FastAPI serve từ đó — xem src/api.py). base '/' vì app chạy
// ở gốc domain sau cutover; giai đoạn dogfood serve qua /beta (route riêng ở FastAPI).
//
// PWA: SW BỊ TẮT trong suốt migration (injectRegister:false) để không tranh chấp
// cache với web/sw.js legacy đang chạy ở scope '/'. Khi cutover (M5) mới bật lại
// (injectRegister:'auto', registerType:'autoUpdate'). Xem plan "Service Worker
// trong giai đoạn coexistence".
// base: '/' sau cutover; trong migration build dogfood với VITE_BASE=/beta/ để asset
// + router chạy dưới /beta (FastAPI serve web/dist ở đó, legacy vẫn ở '/').
const BASE = process.env.VITE_BASE ?? '/';

export default defineConfig({
  base: BASE,
  plugins: [
    react(),
    VitePWA({
      injectRegister: false, // KHÔNG tự chèn <script> đăng ký SW (migration)
      selfDestroying: false,
      registerType: 'autoUpdate',
      devOptions: { enabled: false }, // dev: không SW
      manifest: {
        name: 'Speaking Grader',
        short_name: 'Speaking Grader',
        start_url: '/',
        scope: '/',
        display: 'standalone',
        background_color: '#ffffff',
        theme_color: '#4f46e5',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: '/icons/icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        navigateFallback: '/index.html',
        // Không cache các route API (giữ đúng hành vi legacy sw.js).
        navigateFallbackDenylist: [
          /^\/(grade|grade-batch|exam|auth|history|words|word-info|tts|settings|suggest|health|docs|openapi)/,
        ],
      },
    }),
  ],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    // Cho phép import CSS gốc ở ../web/css (single source of truth trong migration).
    fs: { allow: ['..'] },
  },
  build: {
    outDir: '../web/dist',
    emptyOutDir: true,
  },
});
