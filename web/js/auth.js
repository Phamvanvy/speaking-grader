'use strict';

// Đăng nhập bằng tài khoản (tuỳ chọn) — để đồng bộ lịch sử chấm bài qua nhiều
// thiết bị thay vì phụ thuộc UUID trong localStorage của 1 trình duyệt.
//
// State/danh tính nằm ở state.js (authState/getUserId/authToken/...). File này chỉ
// lo UI: widget góc màn hình, modal đăng nhập/đăng ký, và popup "gộp lịch sử" sau
// khi đăng nhập lần đầu trên máy đang có dữ liệu ẩn danh.
//
// Bảo mật per-user do server lo (user_id tài khoản là "khoá", cần Bearer token);
// fetch tự đính token qua wrapper ở state.js.

function authApi(path) { return `${apiBase()}${path}`; }

// ── Render widget danh tính (góc trên) ────────────────────────────────
function renderAuthWidget() {
    const el = document.getElementById('auth-widget');
    if (!el) return;
    const a = authState();
    if (a && a.username) {
        el.innerHTML =
            `<span class="auth-user" title="Đã đăng nhập — lịch sử đồng bộ đa thiết bị">`
            + `👤 ${escapeAuth(a.username)}</span>`
            + `<button class="auth-link" id="auth-logout-btn">Đăng xuất</button>`;
        document.getElementById('auth-logout-btn').onclick = doLogout;
    } else {
        el.innerHTML =
            `<button class="auth-link auth-login-cta" id="auth-login-btn">🔐 Đăng nhập</button>`;
        document.getElementById('auth-login-btn').onclick = () => openAuthModal('login');
    }
}

function escapeAuth(s) {
    return String(s).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ── Modal đăng nhập / đăng ký ─────────────────────────────────────────
function openAuthModal(tab) {
    let modal = document.getElementById('auth-modal');
    if (!modal) { modal = buildAuthModal(); document.body.appendChild(modal); }
    setAuthTab(tab || 'login');
    setAuthError('');
    modal.classList.remove('hidden');
    const first = modal.querySelector('input');
    if (first) first.focus();
}
function closeAuthModal() {
    const modal = document.getElementById('auth-modal');
    if (modal) modal.classList.add('hidden');
}

function setAuthTab(tab) {
    const modal = document.getElementById('auth-modal');
    if (!modal) return;
    modal.dataset.tab = tab;
    modal.querySelector('#auth-tab-login').classList.toggle('active', tab === 'login');
    modal.querySelector('#auth-tab-register').classList.toggle('active', tab === 'register');
    modal.querySelector('#auth-submit').textContent =
        tab === 'login' ? 'Đăng nhập' : 'Tạo tài khoản';
    modal.querySelector('#auth-confirm-group').style.display =
        tab === 'register' ? '' : 'none';
    setAuthError('');
}

function setAuthError(msg) {
    const el = document.getElementById('auth-error');
    if (el) { el.textContent = msg || ''; el.style.display = msg ? '' : 'none'; }
}

function buildAuthModal() {
    const modal = document.createElement('div');
    modal.id = 'auth-modal';
    modal.className = 'auth-modal-overlay hidden';
    modal.innerHTML = `
      <div class="auth-modal">
        <button class="auth-modal-close" id="auth-close" aria-label="Đóng">✕</button>
        <div class="auth-tabs">
          <button id="auth-tab-login" class="auth-tab">Đăng nhập</button>
          <button id="auth-tab-register" class="auth-tab">Đăng ký</button>
        </div>
        <p class="auth-hint">Đăng nhập để lịch sử chấm bài đồng bộ trên mọi thiết bị.
           Không đăng nhập vẫn dùng được (lưu theo trình duyệt).</p>
        <form id="auth-form" autocomplete="on">
          <div class="form-group">
            <label for="auth-username">Tên đăng nhập</label>
            <input type="text" id="auth-username" name="username" autocomplete="username"
                   placeholder="3–32 ký tự: chữ, số, . _ -">
          </div>
          <div class="form-group">
            <label for="auth-password">Mật khẩu</label>
            <input type="password" id="auth-password" name="password"
                   autocomplete="current-password" placeholder="ít nhất 8 ký tự">
          </div>
          <div class="form-group" id="auth-confirm-group">
            <label for="auth-confirm">Nhập lại mật khẩu</label>
            <input type="password" id="auth-confirm" name="confirm"
                   autocomplete="new-password">
          </div>
          <div class="auth-error" id="auth-error" style="display:none"></div>
          <button type="submit" class="btn btn-primary" id="auth-submit">Đăng nhập</button>
        </form>
      </div>`;
    modal.querySelector('#auth-close').onclick = closeAuthModal;
    modal.querySelector('#auth-tab-login').onclick = () => setAuthTab('login');
    modal.querySelector('#auth-tab-register').onclick = () => setAuthTab('register');
    modal.addEventListener('click', e => { if (e.target === modal) closeAuthModal(); });
    modal.querySelector('#auth-form').addEventListener('submit', onAuthSubmit);
    return modal;
}

async function onAuthSubmit(e) {
    e.preventDefault();
    const modal = document.getElementById('auth-modal');
    const tab = modal.dataset.tab || 'login';
    const username = modal.querySelector('#auth-username').value.trim();
    const password = modal.querySelector('#auth-password').value;
    const submitBtn = modal.querySelector('#auth-submit');
    setAuthError('');

    if (!username || !password) { setAuthError('Nhập đủ tên đăng nhập và mật khẩu.'); return; }
    if (tab === 'register') {
        const confirm = modal.querySelector('#auth-confirm').value;
        if (password.length < 8) { setAuthError('Mật khẩu phải ít nhất 8 ký tự.'); return; }
        if (password !== confirm) { setAuthError('Mật khẩu nhập lại không khớp.'); return; }
    }

    submitBtn.disabled = true;
    const prevText = submitBtn.textContent;
    submitBtn.textContent = tab === 'login' ? 'Đang đăng nhập…' : 'Đang tạo…';
    try {
        const path = tab === 'login' ? '/auth/login' : '/auth/register';
        const res = await fetch(authApi(path), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) { setAuthError(data.detail || 'Có lỗi xảy ra. Thử lại.'); return; }

        // Ghi nhớ UUID ẩn danh TRƯỚC khi lưu auth (getUserId sẽ đổi sang tài khoản).
        const anonId = getAnonUserId();
        setAuth({ token: data.token, user_id: data.user_id, username: data.username });
        closeAuthModal();
        renderAuthWidget();
        await maybeClaimHistory(anonId);
        refreshAfterAuthChange();
    } catch (err) {
        setAuthError('Không kết nối được máy chủ.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = prevText;
    }
}

// ── Gộp lịch sử ẩn danh vào tài khoản (đăng nhập lần đầu trên máy có dữ liệu) ──
async function maybeClaimHistory(anonId) {
    if (!anonId || !isLoggedIn()) return;
    // Chỉ hỏi khi UUID ẩn danh thật sự có dữ liệu trên server.
    let hasData = false;
    try {
        const res = await fetch(
            `${apiBase()}/history/list?user_id=${encodeURIComponent(anonId)}&limit=1`,
            { headers: {} });
        if (res.ok) { const d = await res.json(); hasData = (d.total || 0) > 0; }
    } catch (e) { /* bỏ qua — không chặn đăng nhập */ }

    if (!hasData) return;
    const ok = confirm(
        'Máy này đang có lịch sử chấm bài chưa gắn tài khoản.\n\n'
        + 'Chuyển toàn bộ lịch sử đó vào tài khoản vừa đăng nhập?');
    if (!ok) return;
    try {
        const res = await fetch(authApi('/auth/claim'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ anon_user_id: anonId }),
        });
        if (res.ok) {
            const d = await res.json();
            // Đã gộp xong → sinh UUID ẩn danh mới để id cũ không tái dùng.
            regenerateAnonUserId();
            if ((d.records || 0) + (d.words || 0) > 0) {
                alert(`Đã chuyển ${d.records} bản ghi lịch sử`
                    + (d.words ? ` và ${d.words} từ đã lưu` : '') + ' vào tài khoản.');
            }
        }
    } catch (e) { /* bỏ qua */ }
}

async function doLogout() {
    try { await fetch(authApi('/auth/logout'), { method: 'POST' }); }
    catch (e) { /* vẫn xoá phía client dù server lỗi */ }
    clearAuth();
    renderAuthWidget();
    refreshAfterAuthChange();
}

// Sau khi đổi trạng thái đăng nhập, danh tính (user_id) đổi → nạp lại các tab
// server-side đang mở để hiển thị đúng dữ liệu của danh tính mới.
function refreshAfterAuthChange() {
    const visible = id => { const el = document.getElementById(id);
        return el && !el.classList.contains('hidden'); };
    try { if (visible('mode-history') && window.loadHistoryList) loadHistoryList(); } catch (e) {}
    try { if (visible('mode-saved') && window.loadSavedWords) loadSavedWords(); } catch (e) {}
}

// Khôi phục phiên khi mở lại trang: xác thực token còn hạn không (dọn nếu hết).
async function verifyAuthOnLoad() {
    if (!isLoggedIn()) return;
    try {
        const res = await fetch(authApi('/auth/me'));
        if (res.status === 401) { clearAuth(); }
    } catch (e) { /* offline — giữ nguyên, thử lại lần sau */ }
    renderAuthWidget();
}

document.addEventListener('DOMContentLoaded', () => {
    renderAuthWidget();
    verifyAuthOnLoad();
});
