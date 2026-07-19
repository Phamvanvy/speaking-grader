// Widget danh tính góc trên — port renderAuthWidget/verifyAuthOnLoad (web/js/auth.js).
// Modal đầy đủ (username/password + Google) nằm ở features/auth/AuthDialog.

import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { CircleUserRound } from 'lucide-react';
import { useAuthStore } from '../store/auth';
import { useAuthDialog } from '../features/auth/AuthDialog';
import { doLogout, verifyAuthOnLoad } from '../features/auth/authActions';

export default function AuthWidget() {
  const auth = useAuthStore((s) => s.auth);
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn);
  const openDialog = useAuthDialog((s) => s.openDialog);
  const qc = useQueryClient();

  // Khôi phục phiên khi mở lại trang (token hết hạn → dọn auth cục bộ).
  useEffect(() => {
    verifyAuthOnLoad(qc);
  }, [qc]);

  return (
    <div className="auth-widget" id="auth-widget">
      {isLoggedIn ? (
        <>
          <Link className="auth-user" to="/account" title="Xem thông tin tài khoản">
            <CircleUserRound size={17} strokeWidth={2} aria-hidden />
            <span className="auth-user__name">{auth?.username || auth?.user_id?.slice(0, 8)}</span>
          </Link>
          <button className="auth-link" onClick={() => doLogout(qc)}>
            Đăng xuất
          </button>
        </>
      ) : (
        <button className="auth-link auth-login-cta" onClick={() => openDialog('login')}>
          🔐 Đăng nhập
        </button>
      )}
    </div>
  );
}
