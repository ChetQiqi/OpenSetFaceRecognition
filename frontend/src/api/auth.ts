import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type { TokenResponse, User, RegisterRequest, UpdateUserRoleRequest } from '../types';

export function login(username: string, password: string): Promise<TokenResponse> {
  return apiPost<TokenResponse>('/auth/login', { username, password });
}

export function register(data: RegisterRequest): Promise<User> {
  return apiPost<User>('/auth/register', data);
}

export function getMe(): Promise<User> {
  return apiGet<User>('/auth/me');
}

export function listUsers(): Promise<User[]> {
  return apiGet<User[]>('/auth/users');
}

export function deleteUser(userId: number): Promise<{ deleted: boolean; user_id: number }> {
  return apiDelete<{ deleted: boolean; user_id: number }>(`/auth/users/${userId}`);
}

export function updateUserRole(userId: number, role: UpdateUserRoleRequest): Promise<User> {
  return apiPut<User>(`/auth/users/${userId}/role`, role);
}
