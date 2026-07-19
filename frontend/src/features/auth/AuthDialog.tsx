// Modal đăng nhập / đăng ký — port web/js/auth.js (buildAuthModal + onAuthSubmit +
// Google Identity Services) sang shadcn Dialog. Giữ nguyên class auth.css cho phần
// nội dung (tabs / divider / slot Google / error) nên giao diện khớp legacy.

import { useEffect, useRef, useState } from 'react';
import { create } from 'zustand';
import { useQueryClient } from '@tanstack/react-query';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { apiFetch, apiUrl } from '@/lib/api';
import { completeLogin, type AuthResponse } from './authActions';

type Tab = 'login' | 'register';

interface AuthDialogUi {
  open: boolean;
  tab: Tab;
  openDialog: (tab?: Tab) => void;
  setOpen: (open: boolean) => void;
}
export const useAuthDialog = create<AuthDialogUi>((set) => ({
  open: false,
  tab: 'login',
  openDialog: (tab = 'login') => set({ open: true, tab }),
  setOpen: (open) => set({ open }),
}));

// ── Google Identity Services ────────────────────────────────────────────
// Bật khi server có GOOGLE_CLIENT_ID (/auth/config). GIS bắt buộc nạp từ CDN Google
// (không self-host được) → nạp lười lúc mở modal, lỗi mạng thì ẩn nút, form thường vẫn chạy.
let _clientId: string | null = null; // null = chưa hỏi server; '' = server tắt
let _gisScript: Promise<void> | null = null;

async function fetchGoogleClientId(): Promise<string> {
  if (_clientId !== null) return _clientId;
  let id = '';
  try {
    const res = await fetch(apiUrl('/auth/config'));
    if (res.ok) id = (await res.json()).google_client_id || '';
  } catch {
    id = ''; // server tắt/offline → coi như không cấu hình Google
  }
  _clientId = id;
  return id;
}

function loadGisScript(): Promise<void> {
  if (_gisScript) return _gisScript;
  _gisScript = new Promise<void>((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://accounts.google.com/gsi/client';
    s.async = true;
    s.defer = true;
    s.onload = () => resolve();
    s.onerror = () => {
      _gisScript = null;
      reject(new Error('GIS load failed'));
    };
    document.head.appendChild(s);
  });
  return _gisScript;
}

export default function AuthDialog() {
  const open = useAuthDialog((s) => s.open);
  const tab = useAuthDialog((s) => s.tab);
  const setOpen = useAuthDialog((s) => s.setOpen);
  const qc = useQueryClient();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [googleOn, setGoogleOn] = useState(false);

  const googleSlot = useRef<HTMLDivElement | null>(null);
  const googleReady = useRef(false);

  function switchTab(next: Tab) {
    useAuthDialog.setState({ tab: next });
    setError('');
  }

  useEffect(() => {
    if (!open) {
      setPassword('');
      setConfirmPw('');
      setError('');
    }
  }, [open]);

  // Render nút Google khi mở modal (1 lần/phiên trang — GIS không cho render lại slot cũ).
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    (async () => {
      const clientId = await fetchGoogleClientId();
      if (cancelled || !clientId) return;
      setGoogleOn(true);
      if (googleReady.current) return;
      try {
        await loadGisScript();
        if (cancelled || !googleSlot.current) return;
        const google = (window as any).google;
        google.accounts.id.initialize({ client_id: clientId, callback: onGoogleCredential });
        google.accounts.id.renderButton(googleSlot.current, {
          theme: document.body.classList.contains('dark') ? 'filled_black' : 'outline',
          size: 'large',
          width: 320,
          text: 'signin_with',
          locale: 'vi',
        });
        googleReady.current = true;
      } catch {
        setGoogleOn(false); // chặn mạng/CDN → ẩn, còn form thường
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  async function onGoogleCredential(response: { credential: string }) {
    setError('');
    try {
      const res = await apiFetch('/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential: response.credential }),
        noRetry: true,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || 'Đăng nhập Google thất bại.');
        return;
      }
      setOpen(false);
      await completeLogin(data as AuthResponse, qc);
    } catch {
      setError('Không kết nối được máy chủ.');
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    const user = username.trim();
    if (!user || !password) {
      setError('Nhập đủ tên đăng nhập (hoặc email) và mật khẩu.');
      return;
    }
    if (tab === 'register') {
      if (password.length < 8) {
        setError('Mật khẩu phải ít nhất 8 ký tự.');
        return;
      }
      if (password !== confirmPw) {
        setError('Mật khẩu nhập lại không khớp.');
        return;
      }
    }

    setBusy(true);
    try {
      const res = await apiFetch(tab === 'login' ? '/auth/login' : '/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password }),
        noRetry: true, // 401/409 không đáng retry; tránh khoá form lâu
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || 'Có lỗi xảy ra. Thử lại.');
        return;
      }
      setOpen(false);
      await completeLogin(data as AuthResponse, qc);
    } catch {
      setError('Không kết nối được máy chủ.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>🔐 Tài khoản</DialogTitle>
        </DialogHeader>

        <div className="auth-tabs">
          <button type="button" className={'auth-tab' + (tab === 'login' ? ' active' : '')} onClick={() => switchTab('login')}>
            Đăng nhập
          </button>
          <button type="button" className={'auth-tab' + (tab === 'register' ? ' active' : '')} onClick={() => switchTab('register')}>
            Đăng ký
          </button>
        </div>

        <p className="auth-hint">
          Đăng nhập để lịch sử chấm bài đồng bộ trên mọi thiết bị. Không đăng nhập vẫn dùng được (lưu theo trình duyệt).
        </p>

        <form onSubmit={submit} className="flex flex-col gap-3" autoComplete="on">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="auth-username">Tên đăng nhập hoặc email</Label>
            <Input
              id="auth-username"
              name="username"
              autoComplete="username"
              placeholder="Nhập tên đăng nhập hoặc email"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="auth-password">Mật khẩu</Label>
            <Input
              id="auth-password"
              name="password"
              type="password"
              autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
              placeholder="ít nhất 8 ký tự"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          {tab === 'register' && (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="auth-confirm">Nhập lại mật khẩu</Label>
              <Input
                id="auth-confirm"
                name="confirm"
                type="password"
                autoComplete="new-password"
                value={confirmPw}
                onChange={(e) => setConfirmPw(e.target.value)}
              />
            </div>
          )}
          {error && <div className="auth-error">{error}</div>}
          <Button type="submit" disabled={busy}>
            {busy ? (tab === 'login' ? 'Đang đăng nhập…' : 'Đang tạo…') : tab === 'login' ? 'Đăng nhập' : 'Tạo tài khoản'}
          </Button>
        </form>

        {/* Ẩn hẳn khi server không cấu hình Google (hoặc CDN GIS lỗi). */}
        <div style={{ display: googleOn ? undefined : 'none' }}>
          <div className="auth-divider">
            <span>hoặc</span>
          </div>
          <div className="auth-google-slot" ref={googleSlot} />
        </div>
      </DialogContent>
    </Dialog>
  );
}
