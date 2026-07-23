import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Navigate } from 'react-router-dom';
import { apiGet } from '../../lib/api';
import { useAuthStore } from '../../store/auth';
import HistoryTab from '../history/HistoryTab';

const PAGE_SIZE = 50;

interface AdminUser {
  user_id: string;
  username: string | null;
  record_count: number;
  last_at: string;
  first_at: string;
}
interface AdminUsersResp {
  total: number;
  limit: number;
  offset: number;
  users: AdminUser[];
}

function dateText(iso: string): string {
  const d = new Date(iso);
  return isNaN(+d) ? iso || '' : d.toLocaleString();
}

export default function AdminPage() {
  const isAdmin = useAuthStore((s) => s.isAdmin);
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<AdminUser | null>(null);

  const listQuery = useQuery({
    queryKey: ['admin', 'users', offset],
    queryFn: () => apiGet<AdminUsersResp>(`/admin/users?limit=${PAGE_SIZE}&offset=${offset}`),
    enabled: isAdmin,
  });

  // Chưa đăng nhập → về trang thi. Đã đăng nhập nhưng không phải admin → server
  // sẽ trả 403; hiện thông báo gọn (tab này lẽ ra đã bị ẩn).
  if (!isLoggedIn) return <Navigate to="/exam" replace />;

  const data = listQuery.data;
  const users: AdminUser[] = data?.users || [];
  const pages = data ? Math.max(1, Math.ceil(data.total / data.limit)) : 1;
  const page = data ? Math.floor(data.offset / data.limit) + 1 : 1;

  return (
    <div id="mode-admin">
      <div className="card">
        <div className="result-header">
          <h2>🛡️ Quản trị — người dùng &amp; lịch sử</h2>
          <button className="btn btn-secondary btn-inline" onClick={() => listQuery.refetch()}>
            ↻ Tải lại
          </button>
        </div>

        {!isAdmin && (
          <p className="history-empty">⛔ Tài khoản này không có quyền quản trị.</p>
        )}
        {isAdmin && listQuery.isLoading && <p className="history-empty">⏳ Đang tải…</p>}
        {isAdmin && listQuery.isError && (
          <p className="history-empty">⚠️ Không tải được danh sách người dùng (cần quyền admin).</p>
        )}
        {isAdmin && listQuery.isSuccess && users.length === 0 && (
          <p className="history-empty">Chưa có người dùng nào có lịch sử chấm bài.</p>
        )}

        {isAdmin && users.length > 0 && (
          <div className="history-list">
            {users.map((u) => {
              const isAnon = !u.username;
              const active = selected?.user_id === u.user_id;
              return (
                <div className="history-row" key={u.user_id}>
                  <span className={'history-badge ' + (isAnon ? 'batch' : 'exam')}>
                    {isAnon ? 'Ẩn danh' : 'Tài khoản'}
                  </span>
                  <div className="history-info">
                    <div className="history-title">{u.username || u.user_id}</div>
                    <div className="history-sub">
                      {[`${u.record_count} bản ghi`, `gần nhất: ${dateText(u.last_at)}`]
                        .filter(Boolean)
                        .join(' · ')}
                    </div>
                  </div>
                  <div className="history-actions">
                    <button
                      className={'btn btn-inline ' + (active ? 'btn-primary' : 'btn-secondary')}
                      onClick={() => setSelected(active ? null : u)}
                    >
                      {active ? 'Đang xem' : 'Xem lịch sử'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {data && pages > 1 && (
          <div className="history-pager">
            <button
              className="btn btn-secondary btn-inline"
              disabled={page <= 1}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              ‹ Trước
            </button>
            <span className="history-page">
              Trang {page}/{pages} · {data.total} người dùng
            </span>
            <button
              className="btn btn-secondary btn-inline"
              disabled={page >= pages}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Sau ›
            </button>
          </div>
        )}
      </div>

      {/* Lịch sử của user được chọn — tái dùng HistoryTab ở chế độ chỉ đọc.
          Audio/zip dùng token admin (authedResourceUrl) → backend bypass cho admin. */}
      {selected && (
        <div id="admin-user-history">
          <div className="result-header" style={{ margin: '1rem 0 0.5rem' }}>
            <h3>
              📋 Lịch sử của: <strong>{selected.username || selected.user_id}</strong>
            </h3>
            <button className="btn btn-secondary btn-inline" onClick={() => setSelected(null)}>
              Đóng
            </button>
          </div>
          <HistoryTab userIdOverride={selected.user_id} readOnly />
        </div>
      )}
    </div>
  );
}
