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
    const wordEsc = escapeHtml(w.word);
    return `<tr class="saved-trow">
        <td class="saved-td saved-td--word">${wordEsc}</td>
        <td class="saved-td saved-td--ipa">
            <span class="saved-row__ipa">${ipaStr ? `/${escapeHtml(ipaStr)}/` : ''}</span>
            <button type="button" class="tts-play" data-word="${wordEsc}" title="Nghe phát âm chuẩn">🔊</button>
        </td>
        <td class="saved-td saved-td--score" title="Điểm luyện gần nhất">${pctText}</td>
        <td class="saved-td saved-td--date">${when}</td>
        <td class="saved-td saved-td--remind">${muteBtn}</td>
        <td class="saved-td saved-td--actions">
            <button type="button" class="btn btn-secondary btn-inline practice-open" data-practice="${payload}" title="Luyện tập từ này">🎙️</button>
            <button type="button" class="btn btn-secondary btn-inline saved-delete" data-word="${wordEsc}" title="Bỏ lưu">🗑</button>
        </td>
    </tr>`;
}

// Hàng tiêu đề bảng — các cột có data-sort là bấm được để đổi khoá sắp xếp.
// Mũi tên ▲/▼ hiện ở cột đang active; dựng lại mỗi lần render nên luôn khớp state.
function savedHeadHtml() {
    const cols = [
        { key: 'word', label: 'Từ' },
        { key: null, label: 'Phát âm' },
        { key: 'score', label: 'Điểm' },
        { key: 'date', label: 'Ngày lưu' },
        { key: 'remind', label: 'Nhắc' },
        { key: null, label: '' },
    ];
    const cells = cols.map(c => {
        if (!c.key) return `<th class="saved-th">${c.label}</th>`;
        const active = _savedSort.key === c.key;
        // Cột chưa chọn vẫn hiện ↕ (mờ) để lộ rõ là bấm sắp xếp được;
        // cột đang chọn hiện ▲/▼ đậm theo chiều.
        const arrow = active ? (_savedSort.dir === 'asc' ? '▲' : '▼') : '↕';
        const aria = active ? ` aria-sort="${_savedSort.dir === 'asc' ? 'ascending' : 'descending'}"` : '';
        return `<th class="saved-th saved-th--sortable${active ? ' active' : ''}" data-sort="${c.key}"${aria}
            title="Bấm để sắp xếp theo ${c.label}">${c.label}<span class="saved-th__arrow">${arrow}</span></th>`;
    }).join('');
    return `<thead><tr>${cells}</tr></thead>`;
}

// ── Phân trang danh sách (mặc định 10 từ mới nhất) ──────────────────────
const SAVED_PAGESIZE_KEY = 'speaking-grader-saved-pagesize';
const SAVED_SORT_KEY = 'speaking-grader-saved-sort';
const SAVED_PAGE_OPTIONS = [10, 20, 50, 0];   // 0 = tất cả
let _savedItems = [];   // toàn bộ đã sort (giữ để prev/next re-slice không fetch lại)
let _savedPage = 0;
let _savedPageCount = 1;   // số trang của lần render gần nhất (cho nút Trang cuối)
let _savedFilter = '';  // lọc theo từ khoá tìm kiếm (lowercase)

// Sắp xếp theo cột: key ∈ {remind, date, word, score}, dir ∈ {asc, desc}.
// Mặc định 'remind'/'desc' = từ đang nhắc trước (mới→cũ) — hành vi cũ.
const SAVED_SORT_KEYS = ['remind', 'date', 'word', 'score'];
const SAVED_SORT_DEFAULT_DIR = { remind: 'desc', date: 'desc', score: 'desc', word: 'asc' };
let _savedSort = _loadSavedSort();

function _loadSavedSort() {
    try {
        const s = JSON.parse(localStorage.getItem(SAVED_SORT_KEY) || '');
        if (s && SAVED_SORT_KEYS.includes(s.key) && (s.dir === 'asc' || s.dir === 'desc')) return s;
    } catch { /* chưa lưu / hỏng → mặc định */ }
    return { key: 'remind', dir: 'desc' };
}

function savedPageSize() {
    const v = parseInt(localStorage.getItem(SAVED_PAGESIZE_KEY), 10);
    return SAVED_PAGE_OPTIONS.includes(v) ? v : 10;
}

function savedPagerHtml(total, pageCount, size) {
    const opts = SAVED_PAGE_OPTIONS
        .map(o => `<option value="${o}"${o === size ? ' selected' : ''}>${o === 0 ? 'Tất cả' : o}</option>`)
        .join('');
    const atFirst = _savedPage <= 0;
    const atLast = _savedPage >= pageCount - 1;
    const nav = pageCount > 1
        ? `<span class="saved-pager__nav">
             <button type="button" class="saved-pager__btn" data-page="first"${atFirst ? ' disabled' : ''} aria-label="Trang đầu" title="Trang đầu">«</button>
             <button type="button" class="saved-pager__btn" data-page="prev"${atFirst ? ' disabled' : ''} aria-label="Trang trước" title="Trang trước">‹</button>
             <span class="saved-pager__pos">Trang ${_savedPage + 1}/${pageCount}</span>
             <button type="button" class="saved-pager__btn" data-page="next"${atLast ? ' disabled' : ''} aria-label="Trang sau" title="Trang sau">›</button>
             <button type="button" class="saved-pager__btn" data-page="last"${atLast ? ' disabled' : ''} aria-label="Trang cuối" title="Trang cuối">»</button>
           </span>` : '';
    return `<div class="saved-pager">
        <label class="saved-pager__size">Hiện <select class="saved-pager__select">${opts}</select> từ mới nhất</label>
        ${nav}
        <span class="saved-pager__total">${total} từ</span>
    </div>`;
}

// Sắp xếp theo cột đang chọn (_savedSort). Mọi khoá tie-break mới→cũ rồi A→Z
// để thứ tự luôn tất định.
function _sortSavedItems() {
    const muted = w => !!(window.ReviewToast && ReviewToast.isMuted(w));
    const dateVal = w => (w.saved_at ? Date.parse(w.saved_at) : 0);
    const scoreVal = w => (w.last_score != null ? w.last_score
        : (w.accuracy != null ? w.accuracy : -1));   // chưa có điểm → dồn cuối
    const { key, dir } = _savedSort;
    _savedItems.sort((a, b) => {
        if (key === 'remind') {
            const ma = muted(a.word), mb = muted(b.word);
            // desc: đang nhắc trước; asc: đã tắt nhắc trước. Tie-break mới→cũ.
            if (ma !== mb) return (ma ? 1 : -1) * (dir === 'asc' ? -1 : 1);
            return dateVal(b) - dateVal(a);
        }
        let base;
        if (key === 'word') base = a.word.localeCompare(b.word);      // A→Z
        else if (key === 'score') base = scoreVal(a) - scoreVal(b);   // thấp→cao
        else base = dateVal(a) - dateVal(b);                          // 'date' cũ→mới
        if (base === 0) base = dateVal(b) - dateVal(a);               // tie: mới→cũ
        if (base === 0) base = a.word.localeCompare(b.word);          // tie: A→Z
        return dir === 'asc' ? base : -base;
    });
}

// Đặt khoá sắp xếp: cùng cột → đảo chiều; cột khác → chiều mặc định của cột đó.
// Header dựng lại trong renderSavedPage nên tự cập nhật mũi tên/active.
function setSavedSort(key) {
    if (!SAVED_SORT_KEYS.includes(key)) return;
    _savedSort = _savedSort.key === key
        ? { key, dir: _savedSort.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: SAVED_SORT_DEFAULT_DIR[key] || 'desc' };
    localStorage.setItem(SAVED_SORT_KEY, JSON.stringify(_savedSort));
    _sortSavedItems();
    _savedPage = 0;
    renderSavedPage();
}

// Đưa danh sách về đầu tầm nhìn (đổi số/trang → xem hàng đầu ngay).
function _scrollSavedTop() {
    const box = document.getElementById('saved-list');
    if (box) box.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Nhận danh sách đầy đủ → sort, reset về trang đầu, rồi render.
function renderSavedList(items) {
    _savedItems = (items || []).slice();
    _sortSavedItems();
    _savedPage = 0;
    renderSavedPage();
}

// Toggle tắt/bật nhắc (review-toast.js phát sự kiện) → xếp lại (mute xuống cuối),
// giữ trang hiện tại. Chỉ khi tab Từ đã lưu đang mở.
document.addEventListener('reviewmute:changed', () => {
    const tab = document.getElementById('mode-saved');
    if (tab && !tab.classList.contains('hidden') && _savedItems.length) {
        _sortSavedItems();
        renderSavedPage();
    }
});

function renderSavedPage() {
    const box = document.getElementById('saved-list');
    if (!box) return;
    if (!_savedItems.length) {
        box.innerHTML = `<div class="saved-empty">Chưa có từ nào. Gõ từ vào ô "Thêm từ" ở trên, hoặc khi xem
           kết quả chấm bấm ☆ trên bảng lỗi để lưu từ vào đây luyện tập.</div>`;
        return;
    }
    // Lọc theo từ khoá (trên từ; sort/thứ tự giữ nguyên) rồi mới phân trang.
    const items = _savedFilter
        ? _savedItems.filter(w => (w.word || '').toLowerCase().includes(_savedFilter))
        : _savedItems;
    if (!items.length) {
        box.innerHTML = `<div class="saved-empty">Không tìm thấy từ nào khớp “${escapeHtml(_savedFilter)}”.</div>`;
        return;
    }
    const size = savedPageSize();
    const total = items.length;
    const pageCount = size ? Math.ceil(total / size) : 1;
    _savedPageCount = pageCount;
    _savedPage = Math.min(Math.max(_savedPage, 0), pageCount - 1);
    const slice = size ? items.slice(_savedPage * size, _savedPage * size + size) : items;
    box.innerHTML = `<div class="saved-table-wrap">
        <table class="saved-table">
            ${savedHeadHtml()}
            <tbody>${slice.map(savedRowHtml).join('')}</tbody>
        </table>
    </div>` + savedPagerHtml(total, pageCount, size);
}

// Điều khiển phân trang (delegated — footer dựng lại mỗi lần render).
document.addEventListener('click', e => {
    const btn = e.target instanceof Element ? e.target.closest('.saved-pager__btn') : null;
    if (!btn || btn.disabled) return;
    switch (btn.dataset.page) {
        case 'first': _savedPage = 0; break;
        case 'prev':  _savedPage -= 1; break;
        case 'next':  _savedPage += 1; break;
        case 'last':  _savedPage = _savedPageCount - 1; break;
    }
    renderSavedPage();
    _scrollSavedTop();
});
document.addEventListener('change', e => {
    const sel = e.target instanceof Element ? e.target.closest('.saved-pager__select') : null;
    if (!sel) return;
    localStorage.setItem(SAVED_PAGESIZE_KEY, sel.value);
    _savedPage = 0;
    renderSavedPage();
    _scrollSavedTop();
});

// Ô tìm từ đã lưu (markup tĩnh — lọc client trên danh sách đã nạp).
document.addEventListener('DOMContentLoaded', () => {
    const s = document.getElementById('saved-search');
    if (s) s.addEventListener('input', () => {
        _savedFilter = (s.value || '').trim().toLowerCase();
        _savedPage = 0;
        renderSavedPage();
    });
});

// Sắp xếp khi bấm tiêu đề cột (delegated — bảng dựng lại mỗi lần render).
document.addEventListener('click', e => {
    const th = e.target instanceof Element ? e.target.closest('.saved-table .saved-th--sortable') : null;
    if (th) setSavedSort(th.dataset.sort);
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
