// Màn thông tin tài khoản (/account) — mở từ AuthWidget góc trên.
//
// Gộp mọi thứ "thuộc về người dùng" vào một chỗ: danh tính (tài khoản hoặc UUID ẩn
// danh), số liệu sử dụng, tuỳ chọn lưu lịch sử (privacy opt-out, cùng key với tab
// Lịch sử) và đổi mật khẩu. Chưa đăng nhập thì đây là trang mời đăng nhập — vẫn cho
// xem số liệu của danh tính ẩn danh hiện tại.

import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { CircleUserRound, KeyRound, LogOut, ShieldCheck, UserPlus } from 'lucide-react';
import { toast } from 'sonner';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { apiFetch, apiGet } from '@/lib/api';
import { getUserId, historySaveEnabled, setHistorySaveEnabled } from '@/lib/identity';
import { useAuthStore } from '@/store/auth';
import { useSavedWords } from '@/store/savedWords';
import { useAuthDialog } from '../auth/AuthDialog';
import { doLogout } from '../auth/authActions';

interface AccountInfo {
  username?: string;
  user_id: string;
  created_at?: string;
}

function dateText(iso?: string): string {
  if (!iso) return '--';
  const d = new Date(iso);
  return isNaN(+d) ? iso : d.toLocaleDateString();
}

/** Một ô số liệu (số to + nhãn nhỏ). */
function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="flex-1 min-w-[8rem] rounded-lg border border-border bg-muted/40 px-4 py-3 text-center">
      <div className="text-2xl font-extrabold leading-tight">{value}</div>
      <div className="text-xs text-muted-foreground mt-0.5">{label}</div>
    </div>
  );
}

export default function AccountPage() {
  const auth = useAuthStore((s) => s.auth);
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn);
  const openDialog = useAuthDialog((s) => s.openDialog);
  const qc = useQueryClient();
  const navigate = useNavigate();

  const userId = getUserId();
  const [saveHistory, setSaveHistory] = useState(historySaveEnabled());

  // Số liệu: /auth/me cho ngày tạo tài khoản, /history/list chỉ lấy `total` (limit=1),
  // từ đã lưu đọc thẳng /words (store Zustand có thể chưa refresh khi vào trang này).
  const meQuery = useQuery({
    queryKey: ['auth', 'me', userId],
    queryFn: () => apiGet<AccountInfo>('/auth/me'),
    enabled: isLoggedIn,
    retry: false,
  });
  const historyQuery = useQuery({
    queryKey: ['history', 'total', userId],
    queryFn: () => apiGet<{ total?: number }>(`/history/list?user_id=${encodeURIComponent(userId)}&limit=1`),
    retry: false,
  });
  const wordsQuery = useQuery({
    queryKey: ['words', 'count', userId],
    queryFn: () => apiGet<{ words?: unknown[] }>(`/words?user_id=${encodeURIComponent(userId)}`),
    retry: false,
    initialData: () => {
      const cached = useSavedWords.getState();
      return cached.loaded ? { words: cached.words } : undefined;
    },
  });

  function toggleHistory(on: boolean) {
    setHistorySaveEnabled(on);
    setSaveHistory(on);
    toast.success(on ? 'Đã bật lưu lịch sử chấm bài.' : 'Đã tắt lưu lịch sử — bài mới sẽ không được lưu.');
  }

  async function logout() {
    await doLogout(qc);
    navigate('/');
  }

  const displayName = auth?.username || (isLoggedIn ? auth?.user_id?.slice(0, 8) : 'Khách (chưa đăng nhập)');

  return (
    <div className="flex flex-col gap-4">
      {/* Danh tính */}
      <Card>
        <CardContent className="flex flex-wrap items-center gap-4 p-6">
          <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
            <CircleUserRound size={40} strokeWidth={1.5} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-xl font-bold break-words">{displayName}</div>
            <div className="text-sm text-muted-foreground">
              {isLoggedIn ? (
                <>
                  <ShieldCheck size={14} className="inline align-[-2px] mr-1" />
                  Tài khoản đồng bộ đa thiết bị
                  {meQuery.data?.created_at ? ` · tham gia ${dateText(meQuery.data.created_at)}` : ''}
                </>
              ) : (
                'Dữ liệu đang lưu theo trình duyệt này. Đăng nhập để đồng bộ sang máy khác.'
              )}
            </div>
            <div className="mt-1 font-mono text-xs text-muted-foreground break-all">ID: {userId}</div>
          </div>
          {isLoggedIn ? (
            <Button variant="outline" onClick={logout}>
              <LogOut size={16} className="mr-2" />
              Đăng xuất
            </Button>
          ) : (
            <Button onClick={() => openDialog('login')}>
              <UserPlus size={16} className="mr-2" />
              Đăng nhập / Đăng ký
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Số liệu sử dụng */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">📊 Hoạt động</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Stat value={historyQuery.data?.total ?? (historyQuery.isLoading ? '…' : 0)} label="Bài đã chấm" />
          <Stat value={wordsQuery.data?.words?.length ?? (wordsQuery.isLoading ? '…' : 0)} label="Từ đã lưu" />
        </CardContent>
      </Card>

      {/* Quyền riêng tư */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">🔒 Quyền riêng tư</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-between gap-4">
          <Label htmlFor="acc-save-history" className="font-normal leading-relaxed">
            Lưu lịch sử chấm bài trên máy chủ
            <span className="block text-xs text-muted-foreground">
              Tắt thì bài chấm mới không được gửi kèm danh tính và không xuất hiện ở tab Lịch sử.
            </span>
          </Label>
          <Switch id="acc-save-history" checked={saveHistory} onCheckedChange={toggleHistory} />
        </CardContent>
      </Card>

      {/* Đổi mật khẩu — chỉ có nghĩa với tài khoản username/password */}
      {isLoggedIn && <ChangePasswordCard />}
    </div>
  );
}

function ChangePasswordCard() {
  const [oldPw, setOldPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [confirmPw, setConfirmPw] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    if (newPw.length < 8) return setError('Mật khẩu mới phải ít nhất 8 ký tự.');
    if (newPw !== confirmPw) return setError('Mật khẩu nhập lại không khớp.');

    setBusy(true);
    try {
      const res = await apiFetch('/auth/change-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_password: oldPw, new_password: newPw }),
        noRetry: true, // 400 (sai mật khẩu cũ) không đáng retry
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || 'Đổi mật khẩu thất bại.');
        return;
      }
      setOldPw('');
      setNewPw('');
      setConfirmPw('');
      toast.success('Đã đổi mật khẩu.');
    } catch {
      setError('Không kết nối được máy chủ.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">
          <KeyRound size={16} className="inline align-[-3px] mr-2" />
          Đổi mật khẩu
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="flex max-w-sm flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="acc-old-pw">Mật khẩu hiện tại</Label>
            <Input
              id="acc-old-pw"
              type="password"
              autoComplete="current-password"
              value={oldPw}
              onChange={(e) => setOldPw(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="acc-new-pw">Mật khẩu mới</Label>
            <Input
              id="acc-new-pw"
              type="password"
              autoComplete="new-password"
              placeholder="ít nhất 8 ký tự"
              value={newPw}
              onChange={(e) => setNewPw(e.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="acc-confirm-pw">Nhập lại mật khẩu mới</Label>
            <Input
              id="acc-confirm-pw"
              type="password"
              autoComplete="new-password"
              value={confirmPw}
              onChange={(e) => setConfirmPw(e.target.value)}
            />
          </div>
          {error && <div className="auth-error">{error}</div>}
          <Button type="submit" disabled={busy || !oldPw || !newPw}>
            {busy ? 'Đang đổi…' : 'Đổi mật khẩu'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
