/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {
  LayoutDashboard,
  Users,
  Frame,
  Video,
  Terminal,
  ShieldAlert,
  HeartPulse,
  Radio,
  LogOut,
  Shield,
  Wand2,
} from 'lucide-react';
import { Language, LOCALES } from '../locales';
import { useAuth } from '../hooks/useAuth';
import type { User } from '../types';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onEmergencyLockdown: () => void;
  isLockedDown: boolean;
  language: Language;
  user: User | null;
}

export default function Sidebar({
  activeTab,
  setActiveTab,
  onEmergencyLockdown,
  isLockedDown,
  language,
  user,
}: SidebarProps) {
  const { logout } = useAuth();
  const t = LOCALES[language];
  const isDeveloper = user?.role === 'developer' || user?.role === 'admin';
  const isAdmin = user?.role === 'admin';

  const allMenuItems = [
    { id: 'dashboard', label: t.sidebar.dashboard, icon: LayoutDashboard, roles: ['viewer', 'developer', 'admin'] },
    { id: 'people', label: t.sidebar.people, icon: Users, roles: ['viewer', 'developer', 'admin'] },
    { id: 'portrait', label: t.sidebar.portrait, icon: Wand2, roles: ['viewer', 'developer', 'admin'] },
    { id: 'recognition', label: t.sidebar.recognition, icon: Frame, roles: ['viewer', 'developer', 'admin'] },
    { id: 'video', label: t.sidebar.video, icon: Video, roles: ['viewer', 'developer', 'admin'] },
    { id: 'developer', label: t.sidebar.developer, icon: Terminal, roles: ['developer', 'admin'] },
    { id: 'admin', label: isAdmin ? 'Admin Panel' : '', icon: Shield, roles: ['admin'] },
  ];

  const menuItems = allMenuItems.filter(
    (item) => user && item.roles.includes(user.role) && item.label
  );

  return (
    <aside className="w-68 bg-[#181c24] border-r border-[#262b35] flex flex-col justify-between h-screen text-[#94a3b8] shrink-0 font-sans select-none">
      {/* Upper Brand Section */}
      <div>
        <div className="p-6 border-b border-[#262b35] bg-[#12161f]">
          <h1 className="text-xl font-bold tracking-wider text-white flex items-center gap-2 font-mono">
            <span className="text-cyan-400 font-extrabold uppercase animate-pulse">E-</span>RECOGNITION
          </h1>
          <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-mono truncate">
            {t.sidebar.appName}
          </p>
        </div>

        {/* Navigation Tabs */}
        <nav className="p-3 space-y-1">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => !isLockedDown && setActiveTab(item.id)}
                disabled={isLockedDown}
                className={`w-full flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-sm transition-all duration-200 ${
                  isActive
                    ? 'bg-[#1e293b]/80 text-[#38bdf8] border-l-2 border-[#06b6d4] font-semibold'
                    : isLockedDown
                      ? 'opacity-40 cursor-not-allowed'
                      : 'hover:bg-[#1a202c] hover:text-white text-slate-400'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-[#22d3ee]' : 'text-slate-500'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Safety & Status Footing */}
      <div className="p-4 bg-[#12161f]/80 border-t border-[#262b35] space-y-4">
        {/* User info */}
        {user && (
          <div className="flex items-center gap-2 px-1">
            <div className="w-7 h-7 rounded-sm bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-[10px] font-bold text-cyan-400 uppercase">
              {user.username.charAt(0)}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-semibold text-slate-300 truncate">{user.username}</p>
              <p className="text-[9px] text-cyan-400 font-mono uppercase tracking-wider">{user.role}</p>
            </div>
          </div>
        )}

        {/* Emergency Button */}
        <button
          onClick={onEmergencyLockdown}
          className={`w-full py-3.5 px-4 rounded-sm font-semibold tracking-wider flex items-center justify-center gap-2 text-xs uppercase transition-all duration-300 ${
            isLockedDown
              ? 'bg-emerald-500 hover:bg-emerald-600 text-white animate-pulse shadow-lg shadow-emerald-500/20'
              : 'bg-red-950 hover:bg-red-800 border border-red-500/30 hover:border-red-500 text-red-200 hover:text-red-100 hover:scale-[1.02] active:scale-[0.98]'
          }`}
        >
          <ShieldAlert className="w-4 h-4" />
          <span>
            {isLockedDown
              ? language === 'zh'
                ? '恢复系统授权'
                : 'Authorize Re-entry'
              : t.sidebar.emergencyLockdown}
          </span>
        </button>

        {/* Logout */}
        <button
          onClick={logout}
          className="w-full py-2.5 px-4 rounded-sm font-semibold tracking-wider flex items-center justify-center gap-2 text-xs uppercase transition-all text-slate-500 hover:text-red-400 hover:bg-red-950/30"
        >
          <LogOut className="w-3.5 h-3.5" />
          <span>{language === 'zh' ? '登出系统' : 'Logout'}</span>
        </button>

        {/* Status indicators */}
        <div className="space-y-2 pt-1 font-mono text-[11px] text-slate-500">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <HeartPulse className="w-3.5 h-3.5 text-emerald-500" />
              <span>{language === 'zh' ? '系统生存雷达' : 'System Health'}</span>
            </div>
            <span className="text-emerald-400 font-semibold uppercase">
              {language === 'zh' ? '最优化' : 'Nominal'}
            </span>
          </div>

          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <Radio className="w-3.5 h-3.5 text-[#38bdf8] animate-pulse" />
              <span>{language === 'zh' ? '密级物理联锁' : 'Network Status'}</span>
            </div>
            <span className="text-[#38bdf8] uppercase">{language === 'zh' ? '特级安全' : 'Secure'}</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
