/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import LoginPage from './components/LoginPage';
import DashboardView from './components/DashboardView';
import ImageRecognitionView from './components/ImageRecognitionView';
import PeopleManagementView from './components/PeopleManagementView';
import VideoAnalysisView from './components/VideoAnalysisView';
import DeveloperConsoleView from './components/DeveloperConsoleView';
import AdminPanel from './components/AdminPanel';
import AIPortraitView from './components/AIPortraitView';
import ApiSimulator from './components/ApiSimulator';
import ToastContainer from './components/ToastContainer';
import { useAuth } from './hooks/useAuth';
import { LogLine } from './types';
import { LOCALES, Language } from './locales';
import { ShieldCheck, AlertOctagon } from 'lucide-react';

export default function App() {
  const { user, isAuthenticated, isLoading } = useAuth();

  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [language, setLanguage] = useState<Language>('zh');
  const [isLockedDown, setIsLockedDown] = useState<boolean>(false);
  const [logsList, setLogsList] = useState<LogLine[]>([]);
  const [isApiSimulatorOpen, setIsApiSimulatorOpen] = useState<boolean>(false);

  const addLogRef = useRef<(level: LogLine['level'], message: string) => void>();

  // System logging helper
  const addLog = useCallback((level: LogLine['level'], message: string) => {
    const now = new Date();
    const timestampStr =
      now.toLocaleTimeString('en-GB') +
      '.' +
      String(now.getMilliseconds()).padStart(3, '0').slice(0, 2);
    const newLog: LogLine = { timestamp: timestampStr, level, message };
    setLogsList((prev) => [...prev, newLog]);
  }, []);

  addLogRef.current = addLog;

  // Toggle lockdown trigger
  const handleEmergencyLockdown = () => {
    if (isLockedDown) {
      setIsLockedDown(false);
      addLog('INFO', 'SYSTEM_LOCKDOWN: Re-entry authorization confirmed.');
      addLog('RESULT', 'RE_ENTRY: All systems restored to standard operational frames.');
    } else {
      setIsLockedDown(true);
      addLog('ERROR', 'SYSTEM_LOCKDOWN: CRISIS ALERT triggered!');
      addLog('WARNING', 'LOCKDOWN: Isolating peripheral entry paths...');
    }
  };

  // Show loading spinner while checking stored token
  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-[#090b11]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs text-slate-500 font-mono uppercase tracking-widest">
            Initializing Sentinel Core...
          </p>
        </div>
      </div>
    );
  }

  // Auth gate — show login page if not authenticated
  if (!isAuthenticated) {
    return <LoginPage />;
  }

  // Rendering screen views — all mounted, hidden when inactive to preserve state
  const renderActiveView = () => {
    if (isLockedDown) return null;
    const isDeveloper = user?.role === 'developer' || user?.role === 'admin';

    return (
      <>
        <div className={activeTab === 'dashboard' ? '' : 'hidden'}>
          <DashboardView setActiveTab={setActiveTab} language={language} />
        </div>
        <div className={activeTab === 'people' ? '' : 'hidden'}>
          <PeopleManagementView addLog={addLog} language={language} />
        </div>
        <div className={activeTab === 'portrait' ? '' : 'hidden'}>
          <AIPortraitView addLog={addLog} language={language} />
        </div>
        <div className={activeTab === 'recognition' ? '' : 'hidden'}>
          <ImageRecognitionView logs={logsList} addLog={addLog} language={language} />
        </div>
        <div className={activeTab === 'video' ? '' : 'hidden'}>
          <VideoAnalysisView addLog={addLog} language={language} />
        </div>
        {isDeveloper && (
          <div className={activeTab === 'developer' ? '' : 'hidden'}>
            <DeveloperConsoleView addLog={addLog} language={language} />
          </div>
        )}
        {user?.role === 'admin' && (
          <div className={activeTab === 'admin' ? '' : 'hidden'}>
            <AdminPanel addLog={addLog} language={language} />
          </div>
        )}
      </>
    );
  };

  const getHeaderTitle = () => {
    const t = LOCALES[language];
    switch (activeTab) {
      case 'dashboard':
        return `System_v4.2.1 — ${t.sidebar.dashboard}`;
      case 'people':
        return t.people.title;
      case 'portrait':
        return t.portrait.title;
      case 'recognition':
        return t.recognition.title;
      case 'video':
        return t.video.cctvTitle;
      case 'developer':
        return t.developer.title;
      case 'admin':
        return 'Admin Panel';
      default:
        return t.sidebar.appName;
    }
  };

  const getHeaderSearchPlaceholder = () => {
    const t = LOCALES[language];
    switch (activeTab) {
      case 'people':
        return t.people.searchPlaceholder;
      case 'recognition':
        return t.header.placeholder;
      case 'developer':
        return t.header.placeholder;
      case 'admin':
        return 'Search users...';
      default:
        return t.header.placeholder;
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#090b11] text-slate-300 font-sans antialiased">
      {/* 1. Left Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onEmergencyLockdown={handleEmergencyLockdown}
        isLockedDown={isLockedDown}
        language={language}
        user={user}
      />

      {/* 2. Main content */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative">
        <Header
          title={getHeaderTitle()}
          searchPlaceholder={getHeaderSearchPlaceholder()}
          toggleApiSimulator={() => setIsApiSimulatorOpen((prev) => !prev)}
          language={language}
          onChangeLanguage={setLanguage}
          user={user}
        />

        <main className="flex-1 min-h-0 relative">
          {renderActiveView()}

          {/* Lockdown overlay */}
          {isLockedDown && (
            <div className="absolute inset-0 z-40 bg-black/95 flex flex-col items-center justify-center p-8 select-none font-mono">
              <div className="absolute inset-0 bg-red-950/10 animate-pulse pointer-events-none border-2 border-red-500/30" />
              <div className="w-full max-w-xl text-center space-y-8 relative z-10 p-10 bg-[#0d0708] border border-red-900 shadow-2xl shadow-red-500/10 rounded-sm">
                <div className="flex items-center justify-center gap-1.5 animate-bounce mb-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
                  <div className="w-3 h-3 rounded-full bg-red-600 border border-red-400" />
                </div>
                <div className="space-y-3">
                  <AlertOctagon className="w-16 h-16 text-red-500 mx-auto animate-pulse" />
                  <h2 className="text-xl font-black tracking-[0.2em] text-red-500 uppercase">
                    {LOCALES[language].lockdown.crisisAlert}
                  </h2>
                  <p className="text-xs text-red-300 max-w-md mx-auto leading-relaxed text-slate-300">
                    {LOCALES[language].lockdown.crisisDesc}
                  </p>
                </div>
                <div className="border border-red-950 bg-black/40 p-4 rounded-sm text-center">
                  <span className="text-[10px] text-red-400 font-extrabold uppercase tracking-widest block mb-2">
                    {LOCALES[language].lockdown.telemState}
                  </span>
                  <span className="text-red-500 font-black text-xs uppercase tracking-wider block animate-pulse">
                    {LOCALES[language].lockdown.telemMsg}
                  </span>
                </div>
                <button
                  onClick={handleEmergencyLockdown}
                  className="w-full py-4 bg-emerald-500 hover:bg-emerald-600 font-extrabold uppercase rounded-sm text-slate-950 text-xs tracking-widest transition-all cursor-pointer shadow-lg shadow-emerald-500/20 active:scale-[0.98]"
                >
                  {LOCALES[language].lockdown.deactivateBtn}
                </button>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* 3. API Simulator drawer */}
      <ApiSimulator
        isOpen={isApiSimulatorOpen}
        onClose={() => setIsApiSimulatorOpen(false)}
        language={language}
      />

      {/* 4. Toast notifications */}
      <ToastContainer />
    </div>
  );
}
