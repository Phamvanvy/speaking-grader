// fetchWithRetry — fetch có timeout từng lần gửi + tự retry khi server quá tải.
//
// Server trả 429 + Retry-After khi hàng đợi chấm đầy (admission control, xem
// src/admission.py). Học viên không cần làm gì: hàm này tự chờ rồi gửi lại —
// UI chỉ thấy "Đang chờ chấm…" lâu hơn. Cũng retry 502/503/504 (proxy/backend
// chớp nhoáng) và lỗi mạng. KHÔNG retry 500 hay 4xx khác — đó là lỗi chấm thật,
// gửi lại chỉ tốn thêm một lượt chấm.
//
// FormData dựng từ Blob tái dùng được giữa các lần fetch → caller build 1 lần
// rồi truyền vào, không cần dựng lại mỗi attempt.

const HTTP_RETRY = {
    // 1 lần gửi tối đa 4 phút: mock_test (whisperx large-v3) + chờ hàng đợi dài
    // là hợp lệ — timeout này chỉ để giết socket chết, không phải request chậm.
    attemptTimeoutMs: 240000,
    // Tổng thời gian (kể cả các lần chờ retry) tối đa 5 phút rồi trả lỗi cuối.
    totalTimeoutMs: 300000,
    maxBackoffMs: 30000,
};

async function fetchWithRetry(url, options = {}) {
    const started = Date.now();
    let attempt = 0;
    for (;;) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), HTTP_RETRY.attemptTimeoutMs);
        let res = null, err = null;
        try {
            res = await fetch(url, { ...options, signal: controller.signal });
        } catch (e) {
            err = e;   // lỗi mạng hoặc abort do timeout
        } finally {
            clearTimeout(timer);
        }
        const retryable = res
            ? (res.status === 429 || res.status === 502 || res.status === 503 || res.status === 504)
            : true;
        if (!retryable) return res;   // OK hoặc lỗi thật (500/4xx) — caller xử lý

        attempt++;
        // Ưu tiên Retry-After của server; không có → exponential backoff. Luôn
        // cộng jitter để 50 học viên không dội lại cùng một nhịp.
        let delayMs = Math.min(2 ** attempt * 2000, HTTP_RETRY.maxBackoffMs);
        const ra = res && res.headers.get('Retry-After');
        if (ra && !isNaN(parseFloat(ra))) delayMs = parseFloat(ra) * 1000;
        delayMs += Math.random() * 1000;

        if (Date.now() - started + delayMs > HTTP_RETRY.totalTimeoutMs) {
            if (res) return res;   // hết kiên nhẫn → trả response lỗi cuối cùng
            throw err || new Error('Server không phản hồi — thử lại sau.');
        }
        await new Promise(r => setTimeout(r, delayMs));
    }
}
