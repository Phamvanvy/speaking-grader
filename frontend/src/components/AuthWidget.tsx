import { useAuthStore } from '../store/auth';

// M0: widget tối giản — hiện trạng thái đăng nhập + logout. Modal đăng nhập/đăng ký
// đầy đủ (username/password + Google) sẽ port từ web/js/auth.js ở M4 (tab Saved cần
// đồng bộ tài khoản). Giữ id/khung để CSS auth.css áp đúng.
export default function AuthWidget() {
  const { auth, isLoggedIn, logout } = useAuthStore();
  return (
    <div className="auth-widget" id="auth-widget">
      {isLoggedIn ? (
        <>
          <span className="auth-user">👤 {auth?.username || auth?.user_id?.slice(0, 8)}</span>
          <button className="btn btn-secondary btn-inline" onClick={logout}>
            Đăng xuất
          </button>
        </>
      ) : (
        <span className="auth-anon" title="Đang dùng ẩn danh (lịch sử lưu theo trình duyệt này)">
          Ẩn danh
        </span>
      )}
    </div>
  );
}
