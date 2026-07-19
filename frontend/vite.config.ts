import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import { fileURLToPath, URL } from 'node:url';

// Build ra web/dist — FastAPI serve thẳng thư mục đó ở '/' (xem _ROOT_DIR trong
// src/api.py). App chạy ở gốc domain nên base '/'; VITE_BASE chỉ cần khi muốn
// serve thử dưới một path con.
//
// PWA: SW Workbox build ra ĐÚNG /sw.js — cùng URL với sw.js của bản vanilla trước
// đây — nên trình duyệt của người dùng cũ coi là bản cập nhật và thay thế tại chỗ:
// không có 2 SW tranh scope '/'. Cache cũ ("sg-shell-*") do app xoá lúc khởi động
// (xem main.tsx), vì cleanupOutdatedCaches chỉ dọn precache của chính Workbox.
const BASE = process.env.VITE_BASE ?? '/';

export default defineConfig({
  base: BASE,
  plugins: [
    react(),
    VitePWA({
      injectRegister: 'auto', // chèn đăng ký SW vào index.html (cutover)
      selfDestroying: false,
      registerType: 'autoUpdate',
      devOptions: { enabled: false }, // dev: vẫn không SW (tránh cache bản dev)
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
        cleanupOutdatedCaches: true,
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
  // (không cần server.fs.allow: mọi source — kể cả CSS kế thừa — đã nằm trong frontend/)
  build: {
    outDir: '../web/dist',
    emptyOutDir: true,
  },
});
