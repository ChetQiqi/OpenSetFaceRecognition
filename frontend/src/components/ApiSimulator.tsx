/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { API_ENDPOINTS } from '../mockData';
import { ApiEndpoint } from '../types';
import { 
  Play, 
  Terminal, 
  Search, 
  Tag, 
  ShieldAlert, 
  Check, 
  Settings, 
  Database,
  Globe,
  Lock,
  Zap,
  RefreshCw,
  Sliders,
  X
} from 'lucide-react';
import { Language, LOCALES } from '../locales';

interface ApiSimulatorProps {
  isOpen: boolean;
  onClose: () => void;
  language?: Language;
}

export default function ApiSimulator({ 
  isOpen, 
  onClose,
  language = 'en'
}: ApiSimulatorProps) {
  const [selectedEndpoint, setSelectedEndpoint] = useState<ApiEndpoint>(API_ENDPOINTS[0]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [requestBodyJson, setRequestBodyJson] = useState<string>(selectedEndpoint.requestSchema || '');
  const [simulationResponse, setSimulationResponse] = useState<string>('');
  const [isSending, setIsSending] = useState(false);
  const [simElapsedTime, setSimElapsedTime] = useState<number>(0);

  // Categories index
  const categories = ['ALL', 'AUTH', 'CORE/STATS', 'IDENTITY', 'INFERENCE', 'CAMERA', 'MODEL', 'DEVELOPER/EVAL'];

  // Filter endpoints
  const filteredEndpoints = API_ENDPOINTS.filter(ep => {
    const matchesSearch = ep.path.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          ep.summary.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'ALL' || ep.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  // Hot swap selected endpoint
  const handleSelectEndpoint = (ep: ApiEndpoint) => {
    setSelectedEndpoint(ep);
    setRequestBodyJson(ep.requestSchema || '');
    setSimulationResponse('');
  };

  // Run mock endpoint simulation
  const handleTestEndpointRun = () => {
    if (isSending) return;
    setIsSending(true);
    setSimulationResponse('');
    
    // Simulate real network delay
    const start = Date.now();
    const mockDelay = 150 + Math.random() * 200;

    setTimeout(() => {
      setIsSending(false);
      setSimElapsedTime(Math.round(Date.now() - start));
      setSimulationResponse(epToRealResponse(selectedEndpoint, requestBodyJson));
    }, mockDelay);
  };

  // Resolve JSON response based on endpoint paths
  const epToRealResponse = (ep: ApiEndpoint, reqBody: string): string => {
    if (ep.responseSchema) return ep.responseSchema;
    
    // Fallback standard responses if missing
    return JSON.stringify({
      status: "ok",
      timestamp: new Date().toISOString(),
      path: ep.path,
      method: ep.method,
      message: "Endpoint execution mock response complete"
    }, null, 2);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-[840px] bg-[#11141e] border-l border-[#262b35] shadow-2xl z-50 flex flex-col h-screen text-slate-300 font-sans select-none animate-[slideIn_0.3s_ease-out]">
      
      {/* 1. Header with dismissal */}
      <div className="h-16 border-b border-[#262b35] px-6 flex items-center justify-between bg-[#151a26]/90 relative shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-cyan-400 animate-pulse" />
          <h3 className="text-sm font-bold font-mono text-white tracking-widest uppercase">
            {language === 'zh' ? '后台 API 接口安全调用模拟器与文档库' : 'Backend APIs Simulator & documentation'}
          </h3>
        </div>
        <button 
          onClick={onClose}
          className="p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded-sm transition-colors cursor-pointer"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* 2. Main split view layout */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        
        {/* Left endpoint list (sidebar of drawer) */}
        <div className="w-72 border-r border-[#262b35] flex flex-col bg-[#0d1017]">
          
          {/* Keyword Search */}
          <div className="p-3 border-b border-[#262b35]">
            <div className="relative">
              <span className="absolute inset-y-0 left-0 flex items-center pl-2 text-slate-500">
                <Search className="w-3.5 h-3.5" />
              </span>
              <input
                type="text"
                placeholder={language === 'zh' ? '检索接口路由及说明...' : 'Find endpoint...'}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-[#151922] border border-[#232a35] text-slate-200 pl-8 pr-3 py-1 text-[11px] rounded-sm focus:outline-none focus:border-cyan-500 font-mono"
              />
            </div>
          </div>

          {/* Category Pill Filters */}
          <div className="p-2 border-b border-[#262b35] flex flex-wrap gap-1 bg-[#12161f]/60 max-h-[140px] overflow-y-auto">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2 py-0.5 text-[9px] font-mono rounded-sm transition-colors border cursor-pointer ${
                  selectedCategory === cat
                    ? 'bg-[#1e293b]/80 border-cyan-500/50 text-[#22d3ee] font-bold'
                    : 'bg-slate-900/60 border-slate-800 text-slate-500 hover:text-slate-300'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Scrolling directory listing */}
          <div className="flex-1 overflow-y-auto divide-y divide-[#20293a]/40 p-2 space-y-1">
            {filteredEndpoints.map((ep, idx) => {
              const isSelected = selectedEndpoint.path === ep.path && selectedEndpoint.method === ep.method;
              
              const methodColors = 
                ep.method === 'POST' ? 'text-emerald-400 border-emerald-950 bg-emerald-950/20' :
                ep.method === 'DELETE' ? 'text-rose-400 border-rose-950 bg-rose-950/20' :
                ep.method === 'PUT' ? 'text-amber-400 border-amber-950 bg-amber-950/20' : 
                'text-[#38bdf8] border-indigo-950 bg-indigo-950/20';

              return (
                <button
                  key={idx}
                  onClick={() => handleSelectEndpoint(ep)}
                  className={`w-full text-left p-2.5 rounded-sm flex flex-col gap-1 transition-all focus:outline-none cursor-pointer ${
                    isSelected
                      ? 'bg-[#1e2533] border-l-2 border-[#06b6d4] font-bold'
                      : 'hover:bg-[#151922] text-slate-400'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`text-[8px] px-1 border rounded-sm font-black font-mono leading-none ${methodColors}`}>
                      {ep.method}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400 truncate tracking-wide">
                      {ep.path}
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-500 truncate leading-none mt-0.5 font-sans">
                    {ep.summary}
                  </p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right endpoint inspector and simulator runner */}
        <div className="flex-1 flex flex-col overflow-y-auto p-6 space-y-6 min-w-0 bg-[#121622]/40">
          
          {/* Method + Path info */}
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className={`text-xs py-1 px-3 border rounded-sm font-black font-mono tracking-wider ${
                selectedEndpoint.method === 'POST' ? 'text-emerald-400 border-emerald-950 bg-emerald-950/20' :
                selectedEndpoint.method === 'DELETE' ? 'text-rose-400 border-rose-950 bg-rose-950/20' :
                selectedEndpoint.method === 'PUT' ? 'text-amber-400 border-amber-950 bg-amber-950/20' : 
                'text-[#38bdf8] border-indigo-950 bg-indigo-950/20'
              }`}>
                {selectedEndpoint.method}
              </span>
              <span className="text-sm font-mono text-cyan-400 font-bold tracking-wider select-text flex-1 truncate">
                {selectedEndpoint.path}
              </span>
            </div>
            
            <p className="text-xs text-slate-300 font-medium leading-relaxed font-sans max-w-xl">
              {selectedEndpoint.summary}
            </p>

            {/* Permission badge details */}
            <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500 pt-1">
              <span className="flex items-center gap-1">
                <Lock className="w-3 h-3" />
                {language === 'zh' ? '安全认证权限: ' : 'Auth permission: '}<span className="text-slate-300 font-bold uppercase">{selectedEndpoint.permission}</span>
              </span>
              <span>•</span>
              <span className="uppercase text-slate-400">{language === 'zh' ? `划分类别: ${selectedEndpoint.category}` : `Category: ${selectedEndpoint.category}`}</span>
            </div>
          </div>

          {/* Interactive Request details */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Left box: Parameters / Body schema */}
            <div className="border border-[#232c3f]/80 p-4 rounded-sm bg-[#0a0d14] flex flex-col">
              <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest font-bold block mb-3 leading-none">
                {language === 'zh' ? '请求入参声明规范 (JSON / Form Body)' : 'Request specifications (JSON/Form Body)'}
              </span>
              
              {selectedEndpoint.requestSchema ? (
                <textarea
                  value={requestBodyJson}
                  onChange={(e) => setRequestBodyJson(e.target.value)}
                  className="flex-1 w-full bg-[#0d1017] border border-[#1e232e] rounded-sm p-3 text-[10.5px] font-mono text-emerald-400 focus:outline-none focus:border-cyan-500/80 leading-relaxed font-bold h-[160px] resize-none select-text"
                />
              ) : (
                <div className="flex-1 flex items-center justify-center border border-[#1a202c] rounded-sm bg-[#0d1017] text-slate-500/80 font-mono text-[10px] uppercase h-[160px] select-none text-center p-4">
                  {language === 'zh' ? 'GET 探测请求无需设置请求正文参数。' : 'No request body arguments expected for GET requests.'}
                </div>
              )}
            </div>

            {/* Right box: Simulated test terminal trigger */}
            <div className="border border-[#232c3f]/80 p-4 rounded-sm bg-[#0a0d14]/40 flex flex-col justify-between gap-4">
              <div>
                <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest font-bold block mb-2 leading-none">
                  {language === 'zh' ? '物理沙盒调用控制台' : 'Mock Execution console'}
                </span>
                <p className="text-[10px] text-slate-400 leading-relaxed max-w-xs font-sans">
                  {language === 'zh' ? '点击下方按钮，向本应用发出模拟客户端网络请求，并动态检测对应的接口报文返回数组。' : 'Click below to dispatch a simulated client request to this endpoint and inspect the resulting payload response array dynamically.'}
                </p>
              </div>

              <button
                onClick={handleTestEndpointRun}
                disabled={isSending}
                className={`w-full py-3.5 rounded-sm font-mono text-xs uppercase font-extrabold tracking-widest flex items-center justify-center gap-2 transition-all cursor-pointer ${
                  isSending
                    ? 'bg-[#172033] text-cyan-500 border border-cyan-800/40 cursor-not-allowed'
                    : 'bg-cyan-500 hover:bg-cyan-600 text-slate-950 shadow-md shadow-cyan-500/15'
                }`}
              >
                {isSending ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                    <span>{language === 'zh' ? '接口分发运行中...' : 'Dispatched API...'}</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>{language === 'zh' ? '分发调用此网关接口' : 'Dispatch Simulated Call'}</span>
                  </>
                )}
              </button>
            </div>

          </div>

          {/* Full Screen response panel stdout terminal style */}
          <div className="border border-[#232c3f]/80 rounded-sm p-5 bg-[#0a0d14] flex flex-col space-y-3">
            <div className="flex justify-between items-center pb-2 border-b border-[#1c222f]">
              <h5 className="text-[9px] font-mono text-slate-500 uppercase tracking-widest leading-none font-bold">
                {language === 'zh' ? '网关仿真响应报文载荷数据 (Response Header & Code)' : 'Simulation Response Payload (JSON)'}
              </h5>
              {simElapsedTime > 0 && (
                <span className="text-[8px] font-mono text-emerald-400 font-black tracking-wide uppercase leading-none">
                  {language === 'zh' ? `200 OK • 网关解算耗时 ${simElapsedTime}ms` : `200 OK • ${simElapsedTime}ms latency`}
                </span>
              )}
            </div>

            {/* JetBrains Mono response block */}
            <pre className="p-4 bg-[#070a0f] border border-slate-900 rounded-sm text-[10.5px] font-mono text-cyan-400 h-[240px] overflow-auto hover:border-slate-800 select-text font-bold leading-relaxed scrollbar-thin">
              {simulationResponse ? simulationResponse : (
                <span className="opacity-20 uppercase font-bold text-[10px]">
                  {language === 'zh' ? '等待仿真终端挂网分发接口响应...' : 'Awaiting endpoint simulation execution...'}
                </span>
              )}
            </pre>
          </div>

        </div>

      </div>

    </div>
  );
}
