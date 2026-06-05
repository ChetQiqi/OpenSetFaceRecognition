import React, { createContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import type { User, UserRole } from '../types';
import * as authApi from '../api/auth';

export interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, email?: string, role?: UserRole) => Promise<void>;
  logout: () => void;
  isAdmin: boolean;
  isDeveloper: boolean;
}

export const AuthContext = createContext<AuthState | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // On mount, validate any stored token
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      setIsLoading(false);
      return;
    }
    authApi.getMe()
      .then((u) => setUser(u))
      .catch(() => {
        localStorage.removeItem('access_token');
        localStorage.removeItem('user_role');
      })
      .finally(() => setIsLoading(false));
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const tokenResp = await authApi.login(username, password);
    localStorage.setItem('access_token', tokenResp.access_token);
    localStorage.setItem('user_role', tokenResp.role);
    const u = await authApi.getMe();
    setUser(u);
  }, []);

  const register = useCallback(async (username: string, password: string, email?: string, role?: UserRole) => {
    await authApi.register({ username, password, email, role });
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_role');
    setUser(null);
  }, []);

  const value: AuthState = {
    user,
    isLoading,
    isAuthenticated: user !== null,
    login,
    register,
    logout,
    isAdmin: user?.role === 'admin',
    isDeveloper: user?.role === 'developer' || user?.role === 'admin',
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
