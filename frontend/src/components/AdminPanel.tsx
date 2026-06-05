import React, { useState, useEffect, useCallback } from 'react';
import { Trash2, RefreshCw, Plus, Shield, AlertTriangle } from 'lucide-react';
import type { User, UserRole } from '../types';
import { Language, LOCALES } from '../locales';
import { listUsers, deleteUser, register, updateUserRole } from '../api/auth';
import { ApiError } from '../api/client';

interface AdminPanelProps {
  addLog: (level: 'INFO' | 'WARNING' | 'ERROR' | 'RESULT', msg: string) => void;
  language?: Language;
}

export default function AdminPanel({ addLog, language = 'en' }: AdminPanelProps) {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'list' | 'register'>('list');

  // Register form
  const [regUsername, setRegUsername] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regEmail, setRegEmail] = useState('');
  const [regRole, setRegRole] = useState<UserRole>('viewer');
  const [registering, setRegistering] = useState(false);

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listUsers();
      setUsers(data);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Failed to load users';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchUsers(); }, [fetchUsers]);

  const handleDelete = async (userId: number, username: string) => {
    try {
      await deleteUser(userId);
      setUsers(prev => prev.filter(u => u.id !== userId));
      addLog('RESULT', `ADMIN: Deleted user "${username}"`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Delete failed';
      addLog('ERROR', `ADMIN: ${msg}`);
    }
  };

  const handleRoleChange = async (userId: number, newRole: UserRole, username: string) => {
    try {
      const updated = await updateUserRole(userId, { role: newRole });
      setUsers(prev => prev.map(u => u.id === userId ? updated : u));
      addLog('RESULT', `ADMIN: Changed "${username}" role to ${newRole}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Role update failed';
      addLog('ERROR', `ADMIN: ${msg}`);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!regUsername.trim() || !regPassword.trim()) return;

    setRegistering(true);
    try {
      const newUser = await register({
        username: regUsername,
        password: regPassword,
        email: regEmail || undefined,
        role: regRole,
      });
      setUsers(prev => [...prev, newUser]);
      setRegUsername('');
      setRegPassword('');
      setRegEmail('');
      addLog('RESULT', `ADMIN: Registered user "${newUser.username}" (${newUser.role})`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Registration failed';
      addLog('ERROR', `ADMIN: ${msg}`);
    } finally {
      setRegistering(false);
    }
  };

  const roleBadgeCls = (role: UserRole) => {
    switch (role) {
      case 'admin': return 'bg-rose-950/30 text-rose-400 border-rose-500/30';
      case 'developer': return 'bg-amber-950/30 text-amber-400 border-amber-500/30';
      default: return 'bg-cyan-950/30 text-cyan-400 border-cyan-500/30';
    }
  };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-[calc(100vh-4rem)] bg-[#090b11] text-slate-300 font-sans select-none">
      {/* Tab bar */}
      <div className="flex items-center justify-between border-b border-[#1f2533] pb-1">
        <div className="flex gap-2 text-xs font-mono font-bold uppercase leading-none">
          <button
            onClick={() => setActiveTab('list')}
            className={`px-4 py-2 border-b-2 transition-colors cursor-pointer ${
              activeTab === 'list' ? 'text-cyan-400 border-cyan-500' : 'text-slate-500 border-transparent hover:text-slate-300'
            }`}
          >
            <Shield className="w-3.5 h-3.5 inline mr-1" />
            {language === 'zh' ? '用户列表' : 'User List'}
          </button>
          <button
            onClick={() => setActiveTab('register')}
            className={`px-4 py-2 border-b-2 transition-colors cursor-pointer ${
              activeTab === 'register' ? 'text-cyan-400 border-cyan-500' : 'text-slate-500 border-transparent hover:text-slate-300'
            }`}
          >
            <Plus className="w-3.5 h-3.5 inline mr-1" />
            {language === 'zh' ? '注册新用户' : 'Register User'}
          </button>
        </div>
        <button
          onClick={fetchUsers}
          className="flex items-center gap-1 px-3 py-1.5 text-[10px] font-mono text-slate-500 hover:text-white uppercase transition-colors cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          {language === 'zh' ? '刷新' : 'Refresh'}
        </button>
      </div>

      {activeTab === 'list' ? (
        <>
          {error && (
            <div className="bg-red-950/20 border border-red-500/30 rounded-sm p-4 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <p className="text-xs text-red-300 font-mono">{error}</p>
            </div>
          )}

          <div className="overflow-x-auto border border-[#1d2330] rounded-sm bg-[#0a0d14]">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-[#11141e] border-b border-[#1e2330] text-[10px] font-mono uppercase text-[#64748b] font-bold">
                <tr>
                  <th className="py-2.5 px-4">ID</th>
                  <th className="py-2.5 px-4">{language === 'zh' ? '用户名' : 'Username'}</th>
                  <th className="py-2.5 px-4">Email</th>
                  <th className="py-2.5 px-4">{language === 'zh' ? '角色' : 'Role'}</th>
                  <th className="py-2.5 px-4">{language === 'zh' ? '状态' : 'Status'}</th>
                  <th className="py-2.5 px-4">{language === 'zh' ? '创建时间' : 'Created'}</th>
                  <th className="py-2.5 px-4 text-center">{language === 'zh' ? '操作' : 'Actions'}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#181d28]/60 text-slate-400 font-mono">
                {loading ? (
                  <tr>
                    <td colSpan={7} className="py-8 text-center">
                      <RefreshCw className="w-5 h-5 text-slate-500 animate-spin mx-auto" />
                    </td>
                  </tr>
                ) : users.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-8 text-center text-slate-500 uppercase text-[10px]">
                      {language === 'zh' ? '暂无用户' : 'No users found'}
                    </td>
                  </tr>
                ) : (
                  users.map(u => (
                    <tr key={u.id} className="hover:bg-[#121622]/40 transition-colors">
                      <td className="py-3 px-4 text-slate-500">{u.id}</td>
                      <td className="py-3 px-4 text-white font-semibold">{u.username}</td>
                      <td className="py-3 px-4 text-[10px]">{u.email || '—'}</td>
                      <td className="py-3 px-4">
                        <span className={`text-[9px] font-bold uppercase px-2 py-0.5 rounded-sm border ${roleBadgeCls(u.role)}`}>
                          {u.role}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`w-1.5 h-1.5 rounded-full inline-block mr-1.5 ${u.is_active ? 'bg-emerald-400' : 'bg-slate-600'}`} />
                        <span className={u.is_active ? 'text-emerald-400' : 'text-slate-500'}>
                          {u.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-[10px] text-slate-500">{u.created_at}</td>
                      <td className="py-3 px-4 text-center">
                        <div className="flex items-center justify-center gap-1.5">
                          <select
                            value={u.role}
                            onChange={e => handleRoleChange(u.id, e.target.value as UserRole, u.username)}
                            className="bg-slate-900 border border-[#1f2533] rounded-sm px-2 py-1 text-[10px] text-slate-300 font-mono focus:outline-none focus:border-cyan-500 cursor-pointer"
                          >
                            <option value="viewer">Viewer</option>
                            <option value="developer">Developer</option>
                            <option value="admin">Admin</option>
                          </select>
                          <button
                            onClick={() => handleDelete(u.id, u.username)}
                            className="p-1.5 text-slate-600 hover:text-rose-400 hover:bg-rose-950/20 rounded-sm cursor-pointer transition-colors"
                            title={language === 'zh' ? '删除用户' : 'Delete user'}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="max-w-md mx-auto bg-[#121622] rounded-sm p-6 border border-[#1e2330] space-y-5">
          <h4 className="text-xs font-bold font-mono tracking-wider text-[#38bdf8] uppercase">
            {language === 'zh' ? '注册新用户' : 'Register New User'}
          </h4>

          <form onSubmit={handleRegister} className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-[10px] text-slate-500 uppercase font-bold font-mono">{language === 'zh' ? '用户名' : 'Username'}</label>
              <input
                type="text" required value={regUsername}
                onChange={e => setRegUsername(e.target.value)}
                className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] text-slate-500 uppercase font-bold font-mono">{language === 'zh' ? '密码' : 'Password'}</label>
              <input
                type="password" required value={regPassword}
                onChange={e => setRegPassword(e.target.value)}
                className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] text-slate-500 uppercase font-bold font-mono">Email ({language === 'zh' ? '可选' : 'optional'})</label>
              <input
                type="email" value={regEmail}
                onChange={e => setRegEmail(e.target.value)}
                className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] text-slate-500 uppercase font-bold font-mono">{language === 'zh' ? '角色' : 'Role'}</label>
              <select
                value={regRole}
                onChange={e => setRegRole(e.target.value as UserRole)}
                className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500"
              >
                <option value="viewer">Viewer</option>
                <option value="developer">Developer</option>
                <option value="admin">Admin</option>
              </select>
            </div>

            <button
              type="submit" disabled={registering}
              className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-bold uppercase tracking-wider rounded-sm cursor-pointer disabled:cursor-not-allowed text-xs font-mono flex items-center justify-center gap-1.5"
            >
              {registering ? (
                <><RefreshCw className="w-3.5 h-3.5 animate-spin" />{language === 'zh' ? '注册中...' : 'Registering...'}</>
              ) : (
                <><Plus className="w-3.5 h-3.5" />{language === 'zh' ? '注册用户' : 'Register User'}</>
              )}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
