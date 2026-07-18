'use strict';

// Toast "Ôn lại từ đã lưu" — định kỳ hiện ~N từ trong danh sách đã lưu (ưu tiên
// từ yếu / lâu chưa ôn) để user liếc qua hoặc bấm luyện lại. Hoàn toàn client-side:
// đọc cache window.SavedWords (saved.js), tái dùng delegated handler .practice-open
// (practice.js) và .tts-play (playback.js) nên không tự viết click wiring.

// ── Settings (localStorage, convention speaking-grader-*) ────────────────
const REVIEW_TOAST_KEY = 'speaking-grader-review-toast';
const RT_DEFAULTS = { enabled: true, count: 5, hideSec: 15, intervalMin: 10 };

// Delay lịch trình.
const RT_FIRST_DELAY_MS = 1200;   // lần đầu: ngay sau khi mở app (DOM ổn + refresh xong)
const RT_LOAD_RETRY_MS = 800;     // cache chưa load kịp → thử lại nhanh (không lỡ lần đầu)
const RT_BUSY_RETRY_MS = 60000;   // đang bận (modal/ghi âm/toast mở) → thử lại sau

function reviewToastSettings() {
    let s = {};
    try { s = JSON.parse(localStorage.getItem(REVIEW_TOAST_KEY) || '{}') || {}; } catch (e) { s = {}; }
    const clamp = (v, lo, hi, dflt) => {
        v = parseInt(v, 10);
        return Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : dflt;
    };
    return {
        enabled: s.enabled !== false,   // default ON (giống historySaveEnabled)
        count: clamp(s.count, 1, 10, RT_DEFAULTS.count),
        hideSec: clamp(s.hideSec, 5, 120, RT_DEFAULTS.hideSec),
        intervalMin: clamp(s.intervalMin, 1, 120, RT_DEFAULTS.intervalMin),
        noMobile: s.noMobile === true,   // default false — không nhắc trên màn nhỏ
    };
}

function setReviewToastSettings(patch) {
    const next = Object.assign(reviewToastSettings(), patch);
    localStorage.setItem(REVIEW_TOAST_KEY, JSON.stringify(next));
    applyReviewToastSettings();       // re-arm / stop timer ngay (đọc bản mới)
    saveReviewSettingsToServer();     // đồng bộ DB per-user (fire-and-forget)
}

// Màn nhỏ (điện thoại) — khớp breakpoint CSS .review-toast mobile.
function isMobileViewport() {
    return typeof window.matchMedia === 'function'
        && window.matchMedia('(max-width: 640px)').matches;
}

// ── Đồng bộ cài đặt với DB (per-user, đa thiết bị) ───────────────────────
// localStorage là nguồn ĐỒNG BỘ (scheduler đọc tức thì + fallback offline);
// server là kho BỀN, sync nền: load lúc mở app, ghi khi user đổi.
const REVIEW_SETTING_SERVER_KEY = 'review_toast';

async function loadReviewSettingsFromServer() {
    if (typeof apiBase !== 'function' || typeof getUserId !== 'function') return;
    try {
        const res = await fetch(
            `${apiBase()}/settings?key=${REVIEW_SETTING_SERVER_KEY}&user_id=${encodeURIComponent(getUserId())}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data && data.value) {
            localStorage.setItem(REVIEW_TOAST_KEY, data.value);   // server thắng khi có bản ghi
            fillReviewSettingsUi();
            reconcileReviewSchedule();   // KHÔNG re-arm full — giữ lần hiện đầu sau khi mở app
        }
    } catch (e) { /* offline/khách vãng lai → dùng localStorage */ }
}

function saveReviewSettingsToServer() {
    if (typeof apiBase !== 'function' || typeof getUserId !== 'function') return;
    const fd = new FormData();
    fd.append('user_id', getUserId());
    fd.append('key', REVIEW_SETTING_SERVER_KEY);
    fd.append('value', localStorage.getItem(REVIEW_TOAST_KEY) || '{}');
    fetch(`${apiBase()}/settings`, { method: 'POST', body: fd })
        .catch(() => { /* lưu localStorage vẫn giữ, thử lại lần đổi sau */ });
}

// ── Danh sách từ TẮT nhắc (per-word, client-side) ────────────────────────
const REVIEW_MUTED_KEY = 'speaking-grader-review-muted';
const _mkey = w => (w || '').trim().toLowerCase();

function reviewMutedSet() {
    try { return new Set(JSON.parse(localStorage.getItem(REVIEW_MUTED_KEY) || '[]')); }
    catch (e) { return new Set(); }
}
function isWordMuted(word) { return reviewMutedSet().has(_mkey(word)); }
function setWordMuted(word, muted) {
    const s = reviewMutedSet();
    if (muted) s.add(_mkey(word)); else s.delete(_mkey(word));
    localStorage.setItem(REVIEW_MUTED_KEY, JSON.stringify([...s]));
}
// API cho saved.js (savedRowHtml đọc trạng thái lúc render).
window.ReviewToast = {
    isMuted: isWordMuted,
    setMuted: setWordMuted,
    toggleMuted(word) { const m = !isWordMuted(word); setWordMuted(word, m); return m; },
};

// ── Picker: chọn N từ ưu tiên yếu / lâu chưa ôn ──────────────────────────
let _lastShownWords = new Set();   // chống lặp trong phiên

function pickReviewWords(n) {
    if (!window.SavedWords || !SavedWords._loaded) return [];
    const muted = reviewMutedSet();
    let pool = SavedWords.list().filter(w => !muted.has(_mkey(w.word)));   // bỏ từ đã tắt nhắc
    if (!pool.length) return [];
    // Loại từ của toast trước — chỉ khi còn đủ ≥ n từ khác để hiện.
    const fresh = pool.filter(w => !_lastShownWords.has(w.word));
    if (fresh.length >= n) pool = fresh;

    const now = Date.now();
    const scored = pool.map(w => {
        const s = (w.last_score != null) ? w.last_score
            : (w.accuracy != null) ? w.accuracy : 0;   // chưa luyện → yếu nhất
        const ref = w.last_practiced_at || w.saved_at;
        const days = ref ? (now - Date.parse(ref)) / 86400000 : 999;
        const staleness = Math.min(Math.max(days, 0) / 14, 1);   // 14 ngày = cũ hẳn
        // random*0.08 ≪ trọng số yếu(0.6)+cũ(0.4): chỉ phá thế hòa, không lấn ưu tiên.
        const need = (1 - s) * 0.6 + staleness * 0.4 + Math.random() * 0.08;
        return { w, need };
    });
    scored.sort((a, b) => b.need - a.need);
    const picked = scored.slice(0, n).map(x => x.w);
    _lastShownWords = new Set(picked.map(w => w.word));
    return picked;
}

// ── DOM toast (singleton) ────────────────────────────────────────────────
let _reviewToastEl = null;
let _rtHideTimer = null;

function ensureReviewToast() {
    if (_reviewToastEl) return _reviewToastEl;
    const el = document.createElement('div');
    el.id = 'review-toast';
    el.className = 'review-toast hidden';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.innerHTML = `
        <div class="review-toast__head">
            <span class="review-toast__title">📖 Ôn lại từ đã lưu</span>
            <button type="button" class="review-toast__close" aria-label="Đóng">✕</button>
        </div>
        <div class="review-toast__list"></div>`;
    document.body.appendChild(el);

    el.querySelector('.review-toast__close').addEventListener('click', hideReviewToast);
    // Click vào từ (.practice-open) mở popup luyện (handler practice.js) → đóng toast.
    el.addEventListener('click', e => {
        if (e.target.closest('.practice-open') && !e.target.closest('.tts-play')) hideReviewToast();
    });
    // Hover: dừng tự ẩn để không biến mất giữa lúc đang đọc.
    el.addEventListener('mouseenter', () => { clearTimeout(_rtHideTimer); });
    el.addEventListener('mouseleave', () => {
        const { hideSec } = reviewToastSettings();
        _rtHideTimer = setTimeout(hideReviewToast, hideSec * 1000);
    });
    _reviewToastEl = el;
    return el;
}

function reviewRowHtml(w) {
    // data-practice cùng format savedRowHtml → practice.js mở đúng popup.
    const payload = escapeHtml(JSON.stringify({
        word: w.word, ipa: w.ipa || null, accuracy: w.accuracy,
        phonemes: w.phonemes || [],
    }));
    const ipaStr = (typeof ipaStressString === 'function' && (w.phonemes || []).length)
        ? ipaStressString(w.phonemes) : (w.ipa || '');
    return `<div class="review-toast__row">
        <span class="review-toast__word practice-open" data-practice="${payload}"
            title="Bấm để luyện từ này">${escapeHtml(w.word)}</span>
        <span class="review-toast__ipa">${ipaStr ? `/${escapeHtml(ipaStr)}/` : ''}</span>
        <button type="button" class="tts-play" data-word="${escapeHtml(w.word)}"
            title="Nghe phát âm chuẩn">🔊</button>
        <button type="button" class="review-toast__mute review-mute-toggle" data-word="${escapeHtml(w.word)}"
            title="Dừng nhắc ôn từ này">🔕</button>
    </div>`;
}

function showReviewToast(words) {
    const el = ensureReviewToast();
    el.querySelector('.review-toast__list').innerHTML = words.map(reviewRowHtml).join('');
    el.classList.remove('hidden');
    // Replay animation vào mỗi lần hiện (hidden = display:none nên keyframe chạy lại).
    el.style.animation = 'none';
    void el.offsetWidth;   // reflow
    el.style.animation = '';
    clearTimeout(_rtHideTimer);
    const { hideSec } = reviewToastSettings();
    _rtHideTimer = setTimeout(hideReviewToast, hideSec * 1000);
}

function hideReviewToast() {
    clearTimeout(_rtHideTimer);
    if (_reviewToastEl) _reviewToastEl.classList.add('hidden');
}

function isReviewToastVisible() {
    return _reviewToastEl && !_reviewToastEl.classList.contains('hidden');
}

// ── Scheduler / vòng đời ─────────────────────────────────────────────────
let _rtTimer = null;
let _rtPendingHidden = false;

function scheduleReviewToast(delayMs) {
    clearTimeout(_rtTimer);
    _rtTimer = null;
    if (!reviewToastSettings().enabled) return;
    _rtTimer = setTimeout(maybeShowReviewToast, delayMs);
}

// Đối chiếu nhẹ sau khi nạp cài đặt từ server: bật/tắt hoặc mồi lịch nếu cần,
// nhưng KHÔNG re-arm full interval (tránh huỷ lần hiện đầu ~1.2s sau khi mở app).
function reconcileReviewSchedule() {
    const cfg = reviewToastSettings();
    if (!cfg.enabled) { clearTimeout(_rtTimer); _rtTimer = null; hideReviewToast(); return; }
    if (!_rtTimer) scheduleReviewToast(cfg.intervalMin * 60000);
}

function maybeShowReviewToast() {
    const cfg = reviewToastSettings();
    if (!cfg.enabled) return;
    const intervalMs = cfg.intervalMin * 60000;

    if (document.hidden) { _rtPendingHidden = true; return; }   // hiện lại khi tab quay lại

    // Không nhắc trên điện thoại (màn nhỏ) — theo tuỳ chọn user.
    if (cfg.noMobile && isMobileViewport()) { scheduleReviewToast(intervalMs); return; }

    // Cache chưa load kịp (initial refresh chạy song song) → thử lại nhanh.
    if (!window.SavedWords || !SavedWords._loaded) {
        scheduleReviewToast(RT_LOAD_RETRY_MS);
        return;
    }
    // Đang bận: có modal mở, đang ghi âm chính, hoặc toast đang hiện → chờ.
    const modalOpen = document.querySelector('.practice-overlay:not(.hidden)');
    const recording = typeof mediaRecorder !== 'undefined' && mediaRecorder
        && mediaRecorder.state === 'recording';
    if (modalOpen || recording || isReviewToastVisible()) {
        scheduleReviewToast(RT_BUSY_RETRY_MS);
        return;
    }
    const words = pickReviewWords(cfg.count);
    if (!words.length) { scheduleReviewToast(intervalMs); return; }   // chưa có từ nào
    showReviewToast(words);
    scheduleReviewToast(intervalMs);
}

function applyReviewToastSettings() {
    const cfg = reviewToastSettings();
    if (!cfg.enabled) {
        clearTimeout(_rtTimer);
        hideReviewToast();
        return;
    }
    scheduleReviewToast(cfg.intervalMin * 60000);   // arm lại full interval từ bây giờ
}

document.addEventListener('visibilitychange', () => {
    if (!document.hidden && _rtPendingHidden) {
        _rtPendingHidden = false;
        scheduleReviewToast(3000);   // hiện 3s sau khi quay lại (không pop giữa lúc chuyển tab)
    }
});

// ── Nút tắt/bật nhắc (delegated — dùng chung cho toast và tab Từ đã lưu) ──
function updateMuteButtons(word, muted) {
    document.querySelectorAll(`.review-mute-toggle[data-word="${CSS.escape(word)}"]`).forEach(b => {
        // Trong tab Từ đã lưu: đổi icon/nhãn tại chỗ (không re-fetch).
        if (b.classList.contains('review-toast__mute')) return;   // nút trên toast bị gỡ cùng row
        b.classList.toggle('muted', muted);
        b.textContent = muted ? '🔕' : '🔔';
        b.title = muted ? 'Đã tắt nhắc ôn — bấm để bật lại' : 'Đang nhắc ôn — bấm để tắt nhắc từ này';
    });
}

document.addEventListener('click', e => {
    const btn = e.target instanceof Element ? e.target.closest('.review-mute-toggle') : null;
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const word = btn.dataset.word || '';
    const onToast = !!btn.closest('.review-toast');
    // Trên toast: nút chỉ có nghĩa "dừng nhắc" (mute). Trong tab: toggle.
    const muted = onToast ? (setWordMuted(word, true), true) : window.ReviewToast.toggleMuted(word);
    if (onToast) {
        const row = btn.closest('.review-toast__row');
        if (row) row.remove();
        if (_reviewToastEl && !_reviewToastEl.querySelector('.review-toast__row')) hideReviewToast();
    }
    updateMuteButtons(word, muted);   // đồng bộ nút bên tab Từ đã lưu (nếu đang mở)
    // Báo saved.js xếp lại danh sách (từ tắt nhắc dồn xuống cuối).
    document.dispatchEvent(new CustomEvent('reviewmute:changed'));
});

// ── Settings UI trong tab Từ đã lưu (#review-toast-settings, markup tĩnh) ──
function _rtSettingsEls() {
    return {
        enabled: document.getElementById('rt-enabled'),
        count: document.getElementById('rt-count'),
        hide: document.getElementById('rt-hide'),
        interval: document.getElementById('rt-interval'),
        noMobile: document.getElementById('rt-no-mobile'),
    };
}

function fillReviewSettingsUi() {
    const el = _rtSettingsEls();
    if (!el.enabled) return;   // markup chưa có (chưa DOM-ready)
    const cfg = reviewToastSettings();
    el.enabled.checked = cfg.enabled;
    el.count.value = cfg.count;
    el.hide.value = cfg.hideSec;
    el.interval.value = cfg.intervalMin;
    if (el.noMobile) el.noMobile.checked = cfg.noMobile;
}

function wireReviewSettingsUi() {
    const box = document.getElementById('review-toast-settings');
    if (!box) return;
    fillReviewSettingsUi();
    box.addEventListener('change', () => {
        const el = _rtSettingsEls();
        setReviewToastSettings({
            enabled: el.enabled.checked,
            count: el.count.value,
            hideSec: el.hide.value,
            intervalMin: el.interval.value,
            noMobile: el.noMobile ? el.noMobile.checked : false,
        });
        fillReviewSettingsUi();   // ghi lại giá trị đã clamp để nhập ngoài range tự sửa
    });
}

document.addEventListener('DOMContentLoaded', () => {
    wireReviewSettingsUi();
    scheduleReviewToast(RT_FIRST_DELAY_MS);
    loadReviewSettingsFromServer();   // đồng bộ DB (nền) — không chặn lần hiện đầu
});
