/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Search, Bell, Settings } from 'lucide-react';
import { Language, LOCALES } from '../locales';
import type { User } from '../types';

interface HeaderProps {
  title: string;
  searchPlaceholder?: string;
  onSearchChange?: (val: string) => void;
  searchValue?: string;
  toggleApiSimulator?: () => void;
  apiBadgeCount?: number;
  language: Language;
  onChangeLanguage: (lang: Language) => void;
  user?: User | null;
}

export default function Header({
  title,
  searchPlaceholder = 'Global search entities...',
  onSearchChange,
  searchValue = '',
  toggleApiSimulator,
  apiBadgeCount = 32,
  language,
  onChangeLanguage,
  user,
}: HeaderProps) {
  const t = LOCALES[language];

  // Generate a deterministic color from username for the avatar
  const avatarColor = user
    ? `hsl(${user.username.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0) % 360}, 60%, 50%)`
    : '#ffaa00';

  return (
    <header className="h-16 bg-[#11141e] border-b border-[#262b35] px-6 flex items-center justify-between font-sans shrink-0 select-none">
      {/* Title / Section Heading */}
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold tracking-widest text-[#94a3b8] uppercase font-mono flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          {title}
        </h2>
      </div>

      {/* Global Query Search Input */}
      <div className="relative w-96">
        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-slate-500">
          <Search className="w-4 h-4" />
        </span>
        <input
          type="text"
          value={searchValue}
          onChange={(e) => onSearchChange && onSearchChange(e.target.value)}
          placeholder={searchPlaceholder}
          className="w-full bg-[#181d26] border border-[#262b35] focus:border-[#0891b2] text-slate-200 pl-9 pr-4 py-1.5 rounded-sm text-xs outline-none focus:ring-1 focus:ring-cyan-900 transition-all font-sans"
        />
      </div>

      {/* Control Actions & Operator Bio */}
      <div className="flex items-center gap-4">
        {/* Language Selector Pill */}
        <div className="flex bg-[#1a202c] p-0.5 rounded-sm border border-[#262b35] mr-1">
          <button
            onClick={() => onChangeLanguage('en')}
            className={`px-2 py-1 text-[10px] font-mono leading-none rounded-sm font-bold transition-all cursor-pointer ${
              language === 'en'
                ? 'bg-cyan-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            EN
          </button>
          <button
            onClick={() => onChangeLanguage('zh')}
            className={`px-2 py-1 text-[10px] font-mono leading-none rounded-sm font-bold transition-all cursor-pointer ${
              language === 'zh'
                ? 'bg-cyan-500 text-slate-950 shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            中文
          </button>
        </div>

        {/* Toggle API simulated catalog panel */}
        <button
          onClick={toggleApiSimulator}
          title="Open Backend API Inspector"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-sm bg-[#1e293b] border border-[#334155]/60 hover:bg-[#334155] text-xs text-cyan-400 font-mono hover:text-white transition-all cursor-pointer select-none"
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
          </span>
          <span>
            {t.header.apiSimulator} ({apiBadgeCount})
          </span>
        </button>

        {/* Notifications Icon */}
        <button className="relative p-2 text-slate-400 hover:text-white hover:bg-[#1a202c] rounded-full transition-colors cursor-pointer">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full border border-[#11141e]" />
        </button>

        {/* Settings Icon */}
        <button className="p-2 text-slate-400 hover:text-white hover:bg-[#1a202c] rounded-full transition-colors cursor-pointer">
          <Settings className="w-4 h-4" />
        </button>

        {/* Operator Profile Block */}
        <div className="flex items-center gap-2 pl-2 border-l border-[#262b35]">
          <div className="text-right">
            <p className="text-[11px] font-semibold text-slate-300 font-mono">
              {user?.username ?? t.header.operatorRole}
            </p>
            <p className="text-[9px] text-[#06b6d4] font-mono tracking-widest font-bold">
              {user?.role?.toUpperCase() ?? t.header.levelAuth}
            </p>
          </div>
          <div
            className="w-8 h-8 rounded-sm border border-[#262b35] overflow-hidden select-none flex items-center justify-center text-xs font-bold text-white"
            style={{ backgroundColor: avatarColor }}
          >
            {user ? user.username.charAt(0).toUpperCase() : 'S'}
          </div>
        </div>
      </div>
    </header>
  );
}
