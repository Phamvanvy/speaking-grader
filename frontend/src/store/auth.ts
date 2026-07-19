// Zustand — auth/identity (CLIENT state). Mirror localStorage (identity.ts) để UI
// re-render khi login/logout. Token thực tế vẫn đọc từ localStorage trong apiClient
// (authToken()) nên 2 nguồn luôn khớp qua các setter ở đây.

import { create } from 'zustand';
import {
  authState,
  clearAuth,
  getUserId,
  setAuth as persistAuth,
  type AuthInfo,
} from '../lib/identity';

interface AuthStore {
  auth: AuthInfo | null;
  userId: string;
  isLoggedIn: boolean;
  login: (info: AuthInfo) => void;
  logout: () => void;
  refresh: () => void;
}

export const useAuthStore = create<AuthStore>((set) => ({
  auth: authState(),
  userId: getUserId(),
  isLoggedIn: !!authState()?.token,
  login: (info) => {
    persistAuth(info);
    set({ auth: info, userId: info.user_id, isLoggedIn: true });
  },
  logout: () => {
    clearAuth();
    set({ auth: null, userId: getUserId(), isLoggedIn: false });
  },
  // Đồng bộ lại từ localStorage (vd sau /auth/claim đổi UUID ẩn danh).
  refresh: () => {
    const a = authState();
    set({ auth: a, userId: getUserId(), isLoggedIn: !!a?.token });
  },
}));
