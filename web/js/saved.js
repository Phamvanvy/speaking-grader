'use strict';

// Từ đã lưu để luyện tập — client của API /words (server-side theo user_id ẩn
// danh, cùng cơ chế tab Lịch sử) + renderer tab "Từ đã lưu" (#mode-saved).
// SavedWords giữ cache Map để render.js/practice.js hỏi has(word) ĐỒNG BỘ.

const SavedWords = {
    _cache: new Map(),   // word (lowercase) → entry từ server
    _loaded: false,

    _key(word) { return (word || '').trim().toLowerCase(); },

    has(word) { return this._cache.has(this._key(word)); },
    get(word) { return this._cache.get(this._key(word)) || null; },
    list() { return [...this._cache.values()]; },

    async refresh() {
        const res = await fetch(`${apiBase()}/words?user_id=${encodeURIComponent(getUserId())}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        this._cache = new Map((data.words || []).map(w => [this._key(w.word), w]));
        this._loaded = true;
        return this.list();
    },

    // Upsert: lưu mới hoặc cập nhật (server COALESCE — field không gửi giữ nguyên).
    async add(entry) {
        const fd = new FormData();
        fd.append('user_id', getUserId());
        fd.append('word', this._key(entry.word));
        if (entry.ipa) fd.append('ipa', entry.ipa);
        if (entry.phonemes) fd.append('phonemes', JSON.stringify(entry.phonemes));
        if (entry.accuracy != null) fd.append('accuracy', entry.accuracy);
        if (entry.last_score != null) fd.append('last_score', entry.last_score);
        const res = await fetch(`${apiBase()}/words`, { method: 'POST', body: fd });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const saved = await res.json();
        this._cache.set(this._key(saved.word), saved);
        return saved;
    },

    async remove(word) {
        const res = await fetch(
            `${apiBase()}/words/${encodeURIComponent(this._key(word))}?user_id=${encodeURIComponent(getUserId())}`,
            { method: 'DELETE' },
        );
        if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
        this._cache.delete(this._key(word));
    },
};
window.SavedWords = SavedWords;

// Nạp cache 1 lần khi mở trang để sao ☆/★ trên bảng lỗi hiện đúng trạng thái.
document.addEventListener('DOMContentLoaded', () => {
    SavedWords.refresh().catch(() => { /* server tắt/offline → coi như chưa lưu gì */ });
});

// ── Renderer tab "Từ đã lưu" ────────────────────────────────────────────
function savedRowHtml(w) {
    const pctText = w.last_score != null ? `${Math.round(w.last_score * 100)}%`
        : (w.accuracy != null ? `${Math.round(w.accuracy * 100)}%` : '—');
    const when = w.saved_at ? new Date(w.saved_at).toLocaleDateString('vi-VN') : '';
    // data-practice cùng format render.js → nút "Luyện tập" mở lại đúng popup.
    const payload = escapeHtml(JSON.stringify({
        word: w.word, ipa: w.ipa || null, accuracy: w.accuracy,
        phonemes: w.phonemes || [],
    }));
    // IPA kèm trọng âm: dựng từ phonemes đã lưu (trùng Pronunciation detail); từ lưu
    // từ payload cũ (không phonemes) → fallback chuỗi w.ipa (nay backend cũng kèm nhấn).
    const ipaStr = (typeof ipaStressString === 'function' && (w.phonemes || []).length)
        ? ipaStressString(w.phonemes) : (w.ipa || '');
    // Nút tắt/bật nhắc ôn (per-word) — trạng thái do review-toast.js quản lý; để CUỐI hàng.
    const muted = !!(window.ReviewToast && ReviewToast.isMuted(w.word));
    const muteBtn = `<button type="button" class="btn btn-secondary btn-inline review-mute-toggle${muted ? ' muted' : ''}"
        data-word="${escapeHtml(w.word)}"
        title="${muted ? 'Đã tắt nhắc ôn — bấm để bật lại' : 'Đang nhắc ôn — bấm để tắt nhắc từ này'}">${muted ? '🔕' : '🔔'}</button>`;
    return `<div class="saved-row">
        <span class="saved-row__word">${escapeHtml(w.word)}</span>
        <span class="saved-row__ipa">${ipaStr ? `/${escapeHtml(ipaStr)}/` : ''}</span>
        <button type="button" class="tts-play" data-word="${escapeHtml(w.word)}" title="Nghe phát âm chuẩn">🔊</button>
        <span class="saved-row__score" title="Điểm luyện gần nhất">${pctText}</span>
        <span class="saved-row__meta">${when}</span>
        <button type="button" class="btn btn-secondary btn-inline practice-open" data-practice="${payload}">🎙️ Luyện tập</button>
        <button type="button" class="btn btn-secondary btn-inline saved-delete" data-word="${escapeHtml(w.word)}" title="Bỏ lưu">🗑</button>
        ${muteBtn}
    </div>`;
}

// ── Phân trang danh sách (mặc định 10 từ mới nhất) ──────────────────────
const SAVED_PAGESIZE_KEY = 'speaking-grader-saved-pagesize';
const SAVED_PAGE_OPTIONS = [10, 20, 50, 0];   // 0 = tất cả
let _savedItems = [];   // toàn bộ đã sort mới→cũ (giữ để prev/next re-slice không fetch lại)
let _savedPage = 0;

function savedPageSize() {
    const v = parseInt(localStorage.getItem(SAVED_PAGESIZE_KEY), 10);
    return SAVED_PAGE_OPTIONS.includes(v) ? v : 10;
}

function savedPagerHtml(total, pageCount, size) {
    const opts = SAVED_PAGE_OPTIONS
        .map(o => `<option value="${o}"${o === size ? ' selected' : ''}>${o === 0 ? 'Tất cả' : o}</option>`)
        .join('');
    const nav = pageCount > 1
        ? `<span class="saved-pager__nav">
             <button type="button" class="saved-pager__btn" data-page="prev"${_savedPage <= 0 ? ' disabled' : ''} aria-label="Trang trước">‹</button>
             <span class="saved-pager__pos">Trang ${_savedPage + 1}/${pageCount}</span>
             <button type="button" class="saved-pager__btn" data-page="next"${_savedPage >= pageCount - 1 ? ' disabled' : ''} aria-label="Trang sau">›</button>
           </span>` : '';
    return `<div class="saved-pager">
        <label class="saved-pager__size">Hiện <select class="saved-pager__select">${opts}</select> từ mới nhất</label>
        ${nav}
        <span class="saved-pager__total">${total} từ</span>
    </div>`;
}

// Nhận danh sách đầy đủ → sort mới→cũ, reset về trang đầu (từ mới nhất), rồi render.
function renderSavedList(items) {
    _savedItems = (items || []).slice().sort((a, b) => {
        const ta = a.saved_at ? Date.parse(a.saved_at) : 0;
        const tb = b.saved_at ? Date.parse(b.saved_at) : 0;
        return tb - ta;   // mới nhất trước
    });
    _savedPage = 0;
    renderSavedPage();
}

function renderSavedPage() {
    const box = document.getElementById('saved-list');
    if (!box) return;
    if (!_savedItems.length) {
        box.innerHTML = `<div class="saved-empty">Chưa có từ nào. Gõ từ vào ô "Thêm từ" ở trên, hoặc khi xem
           kết quả chấm bấm ☆ trên bảng lỗi để lưu từ vào đây luyện tập.</div>`;
        return;
    }
    const size = savedPageSize();
    const total = _savedItems.length;
    const pageCount = size ? Math.ceil(total / size) : 1;
    _savedPage = Math.min(Math.max(_savedPage, 0), pageCount - 1);
    const slice = size ? _savedItems.slice(_savedPage * size, _savedPage * size + size) : _savedItems;
    box.innerHTML = slice.map(savedRowHtml).join('') + savedPagerHtml(total, pageCount, size);
}

// Điều khiển phân trang (delegated — footer dựng lại mỗi lần render).
document.addEventListener('click', e => {
    const btn = e.target instanceof Element ? e.target.closest('.saved-pager__btn') : null;
    if (!btn || btn.disabled) return;
    _savedPage += btn.dataset.page === 'next' ? 1 : -1;
    renderSavedPage();
});
document.addEventListener('change', e => {
    const sel = e.target instanceof Element ? e.target.closest('.saved-pager__select') : null;
    if (!sel) return;
    localStorage.setItem(SAVED_PAGESIZE_KEY, sel.value);
    _savedPage = 0;
    renderSavedPage();
});

async function loadSavedWords() {
    const box = document.getElementById('saved-list');
    if (!box) return;
    box.innerHTML = '<div class="saved-empty">Đang tải…</div>';
    try {
        renderSavedList(await SavedWords.refresh());
    } catch (err) {
        box.innerHTML = `<div class="saved-empty">Lỗi tải danh sách: ${escapeHtml(String(err.message || err))}</div>`;
    }
    // Gợi ý luyện âm nạp kèm mỗi lần mở tab — chỉ fetch 1 lần/phiên (nút ↻ ép mới).
    loadWordSuggestions(false);
}
window.loadSavedWords = loadSavedWords;

// ☆ toggle ở bất kỳ đâu (bảng lỗi, hàng gợi ý, popup luyện tập — practice.js
// dispatch sau khi server xong) → nếu tab Từ đã lưu đang mở thì dựng lại danh
// sách từ cache (không fetch lại).
document.addEventListener('savedwords:changed', () => {
    const tab = document.getElementById('mode-saved');
    // Refresh từ server (GET nhẹ) thay vì render cache: giữ đúng thứ tự
    // "mới lưu trước" của server (cache Map append từ mới vào CUỐI).
    if (tab && !tab.classList.contains('hidden')) {
        SavedWords.refresh().then(renderSavedList).catch(() => { /* giữ list cũ */ });
    }
});

// ── Thêm từ ngoài (form trên danh sách) ─────────────────────────────────
// Cho phép lưu từ CHƯA từng xuất hiện trong bài chấm: chỉ gửi word, server tự
// tra IPA (CMUdict). Validate client mirror server (_WORD_RE của src/words.py)
// để lỗi hiện ngay không tốn round-trip.
const _ADD_WORD_RE = /^[A-Za-z][A-Za-z' -]{0,39}$/;

async function addSavedWordSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('saved-add-input');
    const msg = document.getElementById('saved-add-msg');
    const word = (input.value || '').trim().toLowerCase().replace(/\s+/g, ' ');
    msg.className = 'saved-add__msg';
    if (!_ADD_WORD_RE.test(word) || word.split(' ').length > 4) {
        msg.classList.add('err');
        msg.textContent = 'Từ/cụm chỉ gồm chữ cái tiếng Anh (nháy đơn, gạch nối; cụm tối đa 4 từ).';
        return;
    }
    if (SavedWords.has(word)) {
        msg.textContent = `"${word}" đã có trong danh sách.`;
        return;
    }
    try {
        await SavedWords.add({ word });
        input.value = '';
        msg.textContent = `Đã lưu "${word}".`;
        renderSavedList(await SavedWords.refresh());
    } catch (err) {
        msg.classList.add('err');
        msg.textContent = `Lỗi lưu từ: ${err.message || err}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('saved-add-form');
    if (form) form.addEventListener('submit', addSavedWordSubmit);
});

// ── Popup thêm nhanh từ vựng (nút 📚 góc phải trên — index.html) ────────
// Nhiều dòng, mỗi dòng 1 ô nhập từ/cụm tiếng Anh; "Lưu tất cả" → POST /words
// từng từ (server tự tra IPA như form "Thêm từ"). Dòng lưu xong/trùng thì bỏ
// khỏi form, dòng không hợp lệ giữ lại + viền đỏ để user sửa.

function addWordsRowHtml() {
    return `<div class="addwords-row">
        <input type="text" class="saved-add__input addwords-input" placeholder="ví dụ: bookstore / gain valuable insights"
            maxlength="40" autocomplete="off" spellcheck="false">
        <button type="button" class="addwords-row__del" title="Xoá dòng" aria-label="Xoá dòng">✕</button>
    </div>`;
}

function ensureAddWordsModal() {
    let overlay = document.getElementById('addwords-modal');
    if (overlay) return overlay;
    overlay = document.createElement('div');
    overlay.id = 'addwords-modal';
    overlay.className = 'practice-overlay hidden';
    overlay.innerHTML = `
        <div class="practice-modal addwords-box" role="dialog" aria-modal="true" aria-label="Thêm từ vựng">
            <div class="practice-head">
                <h3 class="addwords-title">📚 Thêm từ vựng</h3>
                <span class="practice-head__spacer"></span>
                <button type="button" class="practice-close" id="addwords-close" title="Đóng" aria-label="Đóng">✕</button>
            </div>
            <div class="addwords-hint">Nhập từ hoặc cụm tiếng Anh muốn lưu để luyện tập
                (cụm tối đa 4 từ). Bấm ＋ để thêm nhiều từ cùng lúc.</div>
            <form id="addwords-form">
                <div id="addwords-rows"></div>
                <button type="button" class="btn btn-secondary btn-inline" id="addwords-addrow">＋ Thêm dòng</button>
                <div class="saved-add__msg addwords-msg" id="addwords-msg"></div>
                <button type="submit" class="btn addwords-save" id="addwords-save">Lưu tất cả</button>
            </form>
        </div>`;
    document.body.appendChild(overlay);
    overlay.addEventListener('click', e => { if (e.target === overlay) closeAddWordsPopup(); });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && !overlay.classList.contains('hidden')) closeAddWordsPopup();
    });
    overlay.querySelector('#addwords-close').addEventListener('click', closeAddWordsPopup);
    overlay.querySelector('#addwords-form').addEventListener('submit', addWordsSubmit);
    overlay.querySelector('#addwords-addrow').addEventListener('click', () => {
        const rows = overlay.querySelector('#addwords-rows');
        rows.insertAdjacentHTML('beforeend', addWordsRowHtml());
        rows.lastElementChild.querySelector('.addwords-input').focus();
    });
    // Xoá dòng (delegated — dòng thêm/xoá động); luôn giữ lại ít nhất 1 dòng.
    overlay.querySelector('#addwords-rows').addEventListener('click', e => {
        const del = e.target instanceof Element ? e.target.closest('.addwords-row__del') : null;
        if (!del) return;
        const row = del.closest('.addwords-row');
        const rows = overlay.querySelector('#addwords-rows');
        if (rows.children.length > 1) row.remove();
        else { row.classList.remove('err'); row.querySelector('.addwords-input').value = ''; }
    });
    return overlay;
}

function openAddWordsPopup() {
    const overlay = ensureAddWordsModal();
    const rows = overlay.querySelector('#addwords-rows');
    if (!rows.children.length) rows.insertAdjacentHTML('beforeend', addWordsRowHtml());
    const msg = document.getElementById('addwords-msg');
    msg.className = 'saved-add__msg addwords-msg';
    msg.textContent = '';
    overlay.classList.remove('hidden');
    rows.querySelector('.addwords-input').focus();
}

function closeAddWordsPopup() {
    const overlay = document.getElementById('addwords-modal');
    if (overlay) overlay.classList.add('hidden');
}

async function addWordsSubmit(e) {
    e.preventDefault();
    const overlay = document.getElementById('addwords-modal');
    const msg = document.getElementById('addwords-msg');
    const saveBtn = document.getElementById('addwords-save');
    msg.className = 'saved-add__msg addwords-msg';
    msg.textContent = 'Đang lưu…';
    let saved = 0, dup = 0, bad = 0, failed = 0;
    saveBtn.disabled = true;
    try {
        for (const row of [...overlay.querySelectorAll('.addwords-row')]) {
            const input = row.querySelector('.addwords-input');
            const word = (input.value || '').trim().toLowerCase().replace(/\s+/g, ' ');
            row.classList.remove('err');
            if (!word) continue;   // dòng trống: bỏ qua, không tính lỗi
            if (!_ADD_WORD_RE.test(word) || word.split(' ').length > 4) {
                row.classList.add('err');
                bad++;
                continue;
            }
            // Trùng danh sách đã lưu HOẶC trùng dòng phía trên vừa lưu xong
            // (SavedWords.add cập nhật cache ngay) — bỏ dòng, không gọi server.
            if (SavedWords.has(word)) { dup++; row.remove(); continue; }
            try {
                await SavedWords.add({ word });
                saved++;
                row.remove();
            } catch (err) {
                row.classList.add('err');
                failed++;
            }
        }
    } finally {
        saveBtn.disabled = false;
    }
    const rowsBox = overlay.querySelector('#addwords-rows');
    if (!rowsBox.children.length) rowsBox.insertAdjacentHTML('beforeend', addWordsRowHtml());
    const parts = [];
    if (saved) parts.push(`đã lưu ${saved} từ`);
    if (dup) parts.push(`${dup} từ đã có sẵn`);
    if (bad) parts.push(`${bad} dòng không hợp lệ (chỉ chữ cái tiếng Anh, nháy đơn, gạch nối; cụm ≤4 từ)`);
    if (failed) parts.push(`${failed} dòng lưu lỗi — thử lại`);
    const text = parts.length ? parts.join(' · ') : 'Chưa có từ nào để lưu — nhập từ vào ô trên.';
    msg.textContent = text.charAt(0).toUpperCase() + text.slice(1);
    if (bad || failed) msg.classList.add('err');
    // Tab "Từ đã lưu" (nếu đang mở) + các sao ☆ đang hiện tự cập nhật.
    if (saved) document.dispatchEvent(new CustomEvent('savedwords:changed'));
}

document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('addwords-btn');
    if (btn) btn.addEventListener('click', openAddWordsPopup);
});

// ── Gợi ý luyện âm (API /words/suggestions) ─────────────────────────────
let suggestLoaded = false;

function suggestWeakChipHtml(w) {
    const info = typeof phonemeTip === 'function' ? phonemeTip(w.symbol) : null;
    const pct = w.error_rate != null ? ` · sai ${Math.round(w.error_rate * 100)}%` : '';
    const sym = `/${escapeHtml(w.symbol)}/${pct}`;
    const title = info ? escapeHtml(info.tip) : '';
    // Có từ ví dụ → chip kiêm nút nghe (delegated .tts-play của playback.js).
    return info && info.example
        ? `<button type="button" class="practice-chip bad tts-play" data-word="${escapeHtml(info.example)}" title="${title} — nghe trong từ “${escapeHtml(info.example)}”">${sym}</button>`
        : `<span class="practice-chip bad" title="${title}">${sym}</span>`;
}

function suggestRowHtml(s) {
    // data-practice cùng format savedRowHtml — popup render ok với phonemes rỗng,
    // chấm điểm sẽ điền chip; ☆ trong popup cho phép lưu từ.
    const payload = escapeHtml(JSON.stringify({
        word: s.word, ipa: s.ipa || null, accuracy: null, phonemes: [],
    }));
    const targets = (s.target_phonemes || [])
        .map(p => `<span class="practice-chip bad suggest-row__target">/${escapeHtml(p)}/</span>`)
        .join('');
    const reason = s.reason
        ? `<div class="suggest-row__reason">${escapeHtml(s.reason)}</div>` : '';
    // ☆/★ lưu từ ngay từ hàng gợi ý — cùng class .word-bookmark + data-practice
    // như bảng lỗi (render.js) nên click do listener delegated của practice.js
    // xử lý, và mọi nút cùng data-word tự đồng bộ trạng thái.
    const saved = window.SavedWords && SavedWords.has(s.word);
    const star = `<button type="button" class="word-bookmark${saved ? ' saved' : ''}"
        data-word="${escapeHtml(s.word)}" data-practice="${payload}"
        title="${saved ? 'Bỏ lưu từ này' : 'Lưu từ để luyện tập'}">${saved ? '★' : '☆'}</button>`;
    return `<div class="saved-row suggest-row">
        <div class="suggest-row__main">
            <span class="saved-row__word">${escapeHtml(s.word)}</span>
            ${star}
            <span class="saved-row__ipa">${s.ipa ? `/${escapeHtml(s.ipa)}/` : ''}</span>
            <button type="button" class="tts-play" data-word="${escapeHtml(s.word)}" title="Nghe phát âm chuẩn">🔊</button>
            <span class="suggest-row__targets">${targets}</span>
        </div>
        ${reason}
        <button type="button" class="btn btn-secondary btn-inline practice-open" data-practice="${payload}">🎙️ Luyện tập</button>
    </div>`;
}

async function loadWordSuggestions(force) {
    const weakBox = document.getElementById('suggest-weak');
    const listBox = document.getElementById('suggest-list');
    if (!weakBox || !listBox) return;
    if (suggestLoaded && !force) return;
    weakBox.innerHTML = '';
    listBox.innerHTML = `<div class="saved-empty">Đang tải gợi ý…
        (lần đầu có thể mất vài giây — AI chọn từ cho từng âm)</div>`;
    try {
        const res = await fetch(`${apiBase()}/words/suggestions?user_id=${encodeURIComponent(getUserId())}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        suggestLoaded = true;
        const note = data.source === 'fallback'
            ? `<div class="suggest-note">Chưa đủ dữ liệu chấm điểm — gợi ý theo các âm
               người Việt thường gặp khó. Chấm thêm bài để gợi ý bám sát bạn hơn.</div>`
            : `<div class="suggest-note">Các âm bạn hay sai (tính từ lịch sử chấm) —
               bấm chip để nghe âm mẫu:</div>`;
        weakBox.innerHTML = note + (data.weak_phonemes || []).map(suggestWeakChipHtml).join(' ');
        listBox.innerHTML = (data.suggestions || []).length
            ? data.suggestions.map(suggestRowHtml).join('')
            : '<div class="saved-empty">Chưa có gợi ý — thử lại sau.</div>';
    } catch (err) {
        listBox.innerHTML = `<div class="saved-empty">Không tải được gợi ý: ${escapeHtml(String(err.message || err))}</div>`;
    }
}
window.loadWordSuggestions = loadWordSuggestions;

// Xoá từ trong tab (delegated — list dựng lại mỗi lần load).
document.addEventListener('click', e => {
    const btn = e.target instanceof Element ? e.target.closest('.saved-delete') : null;
    if (!btn) return;
    e.preventDefault();
    const word = btn.dataset.word || '';
    if (!confirm(`Bỏ lưu từ "${word}"?`)) return;
    SavedWords.remove(word)
        .then(() => loadSavedWords())
        .catch(err => alert(`Lỗi xoá: ${err.message || err}`));
});
