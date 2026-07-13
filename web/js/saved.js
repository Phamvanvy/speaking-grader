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
    return `<div class="saved-row">
        <span class="saved-row__word">${escapeHtml(w.word)}</span>
        <span class="saved-row__ipa">${w.ipa ? `/${escapeHtml(w.ipa)}/` : ''}</span>
        <button type="button" class="tts-play" data-word="${escapeHtml(w.word)}" title="Nghe phát âm chuẩn">🔊</button>
        <span class="saved-row__score" title="Điểm luyện gần nhất">${pctText}</span>
        <span class="saved-row__meta">${when}</span>
        <button type="button" class="btn btn-secondary btn-inline practice-open" data-practice="${payload}">🎙️ Luyện tập</button>
        <button type="button" class="btn btn-secondary btn-inline saved-delete" data-word="${escapeHtml(w.word)}" title="Bỏ lưu">🗑</button>
    </div>`;
}

async function loadSavedWords() {
    const box = document.getElementById('saved-list');
    if (!box) return;
    box.innerHTML = '<div class="saved-empty">Đang tải…</div>';
    try {
        const items = await SavedWords.refresh();
        box.innerHTML = items.length
            ? items.map(savedRowHtml).join('')
            : `<div class="saved-empty">Chưa có từ nào. Khi xem kết quả chấm, bấm vào từ sai
               rồi bấm ☆ (hoặc bấm ☆ ngay trên bảng lỗi) để lưu từ vào đây luyện tập.</div>`;
    } catch (err) {
        box.innerHTML = `<div class="saved-empty">Lỗi tải danh sách: ${escapeHtml(String(err.message || err))}</div>`;
    }
    // Gợi ý luyện âm nạp kèm mỗi lần mở tab — chỉ fetch 1 lần/phiên (nút ↻ ép mới).
    loadWordSuggestions(false);
}
window.loadSavedWords = loadSavedWords;

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
        ? `<span class="suggest-row__reason">${escapeHtml(s.reason)}</span>` : '';
    return `<div class="saved-row">
        <span class="saved-row__word">${escapeHtml(s.word)}</span>
        <span class="saved-row__ipa">${s.ipa ? `/${escapeHtml(s.ipa)}/` : ''}</span>
        <button type="button" class="tts-play" data-word="${escapeHtml(s.word)}" title="Nghe phát âm chuẩn">🔊</button>
        <span class="saved-row__meta">${targets}${reason}</span>
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
