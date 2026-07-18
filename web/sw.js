/* Service worker — app shell network-first.
 *
 * Phạm vi CHỦ ĐÍCH hẹp: chỉ đụng static shell (HTML/CSS/JS/vendor/icon/manifest).
 * Mọi request API (/grade, /history/*, /words*, /tts, /auth/*, /exam/*,
 * /word-info, /health, non-GET, upload audio) KHÔNG bị intercept — browser xử lý
 * mặc định nên Bearer header (fetch monkey-patch trong state.js) và audio
 * ?token= đi qua nguyên vẹn.
 *
 * Network-first để không bao giờ dính stale khi sửa code (repo không có build
 * step, cache-bust thủ công bằng ?v=): online luôn lấy bản mới nhất, offline
 * rơi về cache.
 *
 * BẢO TRÌ: thêm/bớt file CSS/JS trong index.html thì phải cập nhật
 * PRECACHE_URLS bên dưới VÀ bump CACHE_NAME.
 */

const CACHE_NAME = "sg-shell-v4";

// Path KHÔNG kèm ?v= — lúc fetch runtime sẽ match lại bằng ignoreSearch.
const PRECACHE_URLS = [
  "/",
  "/index.html",
  "/manifest.json",
  "/favicon.ico",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/icon-maskable-512.png",
  "/icons/apple-touch-icon.png",
  "/icons/favicon-32.png",
  "/css/base.css",
  "/css/components.css",
  "/css/theme-dark.css",
  "/css/phoneme.css",
  "/css/exam.css",
  "/css/history.css",
  "/css/practice.css",
  "/css/auth.css",
  "/js/http.js",
  "/js/state.js",
  "/js/auth.js",
  "/js/render.js",
  "/js/playback.js",
  "/js/recording.js",
  "/js/form.js",
  "/js/grade.js",
  "/js/suggest.js",
  "/js/report.js",
  "/js/history.js",
  "/js/phoneme-tips.js",
  "/js/saved.js",
  "/js/practice.js",
  "/js/review-toast.js",
  "/js/exam.js",
  "/js/router.js",
  "/vendor/alpine.min.js",
];

// Prefix static được phép cache runtime (kèm file lẻ ở root).
const STATIC_PREFIXES = ["/css/", "/js/", "/vendor/", "/icons/"];
const STATIC_FILES = ["/manifest.json", "/favicon.ico"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
      )
      .then(() => self.clients.claim())
  );
});

function isStaticAsset(url) {
  return (
    STATIC_FILES.includes(url.pathname) ||
    STATIC_PREFIXES.some((p) => url.pathname.startsWith(p))
  );
}

async function networkFirstNavigate(request) {
  try {
    return await fetch(request);
  } catch (err) {
    const shell =
      (await caches.match("/")) || (await caches.match("/index.html"));
    if (shell) return shell;
    throw err;
  }
}

async function networkFirstStatic(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (err) {
    // Thử match đúng URL (đã cache runtime kèm ?v=) rồi mới bỏ query để
    // khớp entry precache không có ?v=.
    const cached =
      (await caches.match(request)) ||
      (await caches.match(request, { ignoreSearch: true }));
    if (cached) return cached;
    throw err;
  }
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirstNavigate(request));
    return;
  }

  const url = new URL(request.url);
  if (url.origin === self.location.origin && isStaticAsset(url)) {
    event.respondWith(networkFirstStatic(request));
  }
  // Mọi request khác: không respondWith → browser tự xử lý (API đi thẳng mạng).
});
