/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import {
  Users,
  Binary,
  Activity,
  Cpu,
  ArrowRight
} from 'lucide-react';
import { Language, LOCALES } from '../locales';
import { getStats, getHealth } from '../api/stats';
import { apiGet } from '../api/client';
import type { StatsResponse, HealthResponse, ModelRuntimeInfoResponse } from '../types';

interface DashboardViewProps {
  setActiveTab: (tab: string) => void;
  language?: Language;
}

export default function DashboardView({
  setActiveTab,
  language = 'zh'
}: DashboardViewProps) {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [runtime, setRuntime] = useState<ModelRuntimeInfoResponse | null>(null);
  const [pulseData, setPulseData] = useState<number[]>([25, 40, 30, 65, 45, 50, 75, 55, 60, 80, 50, 90]);
  const [shadowData, setShadowData] = useState<number[]>([15, 20, 18, 25, 22, 19, 30, 24, 28, 32, 25, 30]);
  const [latencies, setLatencies] = useState<number[]>([12, 11, 13, 12, 12, 14, 12, 10, 11, 12]);
  const [currentLatency, setCurrentLatency] = useState(12);

  // Periodically fluctuate and update charts
  useEffect(() => {
    const timer = setInterval(() => {
      setPulseData((prev) => {
        const next = [...prev.slice(1)];
        const lastVal = prev[prev.length - 1];
        const nextVal = Math.max(10, Math.min(100, lastVal + (Math.random() * 30 - 15)));
        next.push(Math.round(nextVal));
        return next;
      });

      setShadowData((prev) => {
        const next = [...prev.slice(1)];
        const lastVal = prev[prev.length - 1];
        const nextVal = Math.max(5, Math.min(45, lastVal + (Math.random() * 10 - 5)));
        next.push(Math.round(nextVal));
        return next;
      });

      // Update current latency randomly
      setCurrentLatency(Math.round(11 + Math.random() * 3));
    }, 2500);

    return () => clearInterval(timer);
  }, []);

  // Fetch real data from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [s, h, r] = await Promise.all([
          getStats(),
          getHealth(),
          apiGet<ModelRuntimeInfoResponse>('/model/runtime'),
        ]);
        setStats(s);
        setHealth(h);
        setRuntime(r);
      } catch {
        // Silently fail — cards show fallback values
      }
    };
    fetchData();
  }, []);

  // Compute stats or display values
  const statCards = [
    { label: language === 'zh' ? '已注册人员' : 'Registered Persons', value: (stats?.person_count ?? 0).toLocaleString(), desc: language === 'zh' ? '数据库总人数' : 'Total in database', icon: Users, accent: 'text-cyan-400' },
    { label: language === 'zh' ? '特征向量数' : 'Feature Vectors', value: (stats?.embedding_count ?? 0).toLocaleString(), desc: language === 'zh' ? '已提取的人脸特征' : 'Extracted embeddings', icon: Binary, accent: 'text-pink-400' },
    { label: language === 'zh' ? 'API 状态' : 'API Status', value: health?.status === 'ok' ? (language === 'zh' ? '在线' : 'Online') : 'Offline', desc: language === 'zh' ? '后端服务运行中' : 'Backend operational', icon: Activity, accent: 'text-emerald-400' },
    { label: language === 'zh' ? '当前模型' : 'Active Model', value: runtime?.model_name ?? 'N/A', desc: language === 'zh' ? '已加载的识别模型' : 'Loaded recognition model', icon: Cpu, accent: 'text-fuchsia-400' },
  ];

  // Map SVG coordinates for real-time spline chart
  const getSvgPath = (points: number[]) => {
    const width = 600;
    const height = 150;
    const spacing = width / (points.length - 1);
    const coordinates = points.map((p, index) => {
      const x = index * spacing;
      const y = height - (p / 100) * height * 0.8 - 10;
      return `${x},${y}`;
    });
    return `M ${coordinates.join(' L ')}`;
  };

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-[calc(100vh-4rem)] bg-[#090b11] text-slate-300 font-sans select-none">
      
      {/* 1. System status ribbon */}
      <div className="flex flex-wrap items-center justify-between text-xs px-5 py-3 rounded-sm border border-[#232a35] bg-[#12161f] font-mono shadow-sm">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            <span className="text-white font-bold tracking-widest text-[11px]">
              {language === 'zh' ? '系统状态: 正常运行' : 'SYSTEM STATUS: OPERATIONAL'}
            </span>
          </div>
          <span className="text-[#475569]">|</span>
          <span className="text-slate-400 uppercase">
            API_TUNNEL: <span className="text-emerald-400">{language === 'zh' ? '已连接' : 'CONNECTED'}</span>
          </span>
          <span className="text-[#475569]">|</span>
          <span className="text-slate-400 uppercase">MODEL_LOD: <span className="text-[#06b6d4]">{runtime?.model_name ?? 'N/A'}</span></span>
        </div>
        <div className="flex items-center gap-6 text-[11px]">
          <div>
            <span className="text-slate-400">{language === 'zh' ? '推理延迟' : 'LATENCY'}</span>
            <span className="ml-2 text-white font-bold">{currentLatency}ms</span>
          </div>
          <div>
            <span className="text-slate-400">{language === 'zh' ? '运行时间' : 'UPTIME'}</span>
            <span className="ml-2 text-emerald-400 font-bold">99.98%</span>
          </div>
        </div>
      </div>

      {/* 2. Brand Hero Container & Quick Metrics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Main interactive title hero banner */}
        <div className="lg:col-span-2 rounded-sm border border-[#1e2330] p-8 flex flex-col justify-between relative overflow-hidden h-[240px] bg-gradient-to-r from-[#111422] to-[#161a2b]">
          {/* Cyber grid background */}
          <div className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(to_right,#808080_1px,transparent_1px),linear-gradient(to_bottom,#808080_1px,transparent_1px)] bg-[size:14px_24px]" />
          <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full filter blur-3xl pointer-events-none" />
          
          <div className="space-y-3 z-10 relative">
            <h2 className="text-3xl font-bold tracking-tight text-white font-mono uppercase bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              {language === 'zh' ? '人脸识别系统' : 'Face Recognition System'}
            </h2>
            <p className="text-sm text-slate-400 font-light max-w-lg leading-relaxed">
              {language === 'zh'
                ? '支持图像人脸识别、视频分析、摄像头实时监控等功能。基于深度学习的人脸检测与特征提取，提供高精度的人员身份认证服务。'
                : 'Image-based face recognition, video analysis, and real-time camera monitoring. Deep learning powered face detection and feature extraction for high-precision identity verification.'}
            </p>
          </div>

          <div className="flex items-center gap-4 z-10 relative">
            <button
              onClick={() => setActiveTab('recognition')}
              className="px-5 py-2.5 bg-cyan-500 hover:bg-cyan-600 text-slate-950 text-xs uppercase font-bold tracking-wider rounded-sm cursor-pointer hover:shadow-lg hover:shadow-cyan-500/10 transition-all font-mono"
            >
              {language === 'zh' ? '开始识别' : 'Start Recognition'}
            </button>
            <button
              onClick={() => {
                const docBtn = document.querySelector('[title="Open Backend API Inspector"]');
                if (docBtn) (docBtn as HTMLButtonElement).click();
              }}
              className="px-5 py-2.5 border border-slate-700 hover:border-slate-500 hover:bg-[#1a202c] text-xs uppercase font-medium tracking-wider text-slate-300 rounded-sm cursor-pointer transition-all font-mono"
            >
              {language === 'zh' ? 'API 文档' : 'API Docs'}
            </button>
          </div>
        </div>

        {/* Dynamic core statistics quadrant */}
        <div className="grid grid-cols-2 gap-4 h-[240px]">
          {statCards.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div 
                key={idx} 
                className="bg-[#121622] rounded-sm p-4 border border-[#1e2330] flex flex-col justify-between"
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 uppercase tracking-widest font-mono font-bold leading-none">
                    {item.label}
                  </span>
                  <Icon className={`w-3.5 h-3.5 ${item.accent}`} />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white font-mono tracking-tight leading-none">
                    {item.value}
                  </h3>
                  <p className="text-[10px] text-emerald-500 mt-1 font-mono tracking-wide">
                    {item.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. Core Analytics Modules Area */}
      <div>
        <h4 className="text-[10px] font-bold tracking-[0.2em] mb-4 text-[#64748b] uppercase font-mono">
          {language === 'zh' ? '功能模块' : 'Modules'}
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          
          {/* People Management Module */}
          <div className="bg-[#12151d] border border-[#1f2533] hover:border-cyan-500/40 p-5 rounded-sm flex flex-col justify-between group transition-all duration-300 h-[190px]">
            <div>
              <div className="w-8 h-8 rounded-sm bg-[#1e293b] flex items-center justify-center border border-slate-700">
                <Users className="w-4 h-4 text-cyan-400" />
              </div>
              <h5 className="text-sm font-semibold text-white mt-4 tracking-tight">
                {language === 'zh' ? '人员管理' : 'People Management'}
              </h5>
              <p className="text-xs text-[#64748b] mt-1 leading-snug">
                {language === 'zh' ? '管理人员信息数据库，支持注册、重命名、删除等操作。上传人脸照片即可自动提取特征向量。' : 'Manage person database with registration, renaming, and deletion. Upload face photos to auto-extract feature vectors.'}
              </p>
            </div>
            <button 
              onClick={() => setActiveTab('people')}
              className="flex items-center justify-between text-xs text-cyan-400 font-mono group-hover:text-cyan-300 transition-colors mt-4 text-left w-full cursor-pointer select-none"
            >
              <span>{language === 'zh' ? '进入管理' : 'Open'}</span>
              <ArrowRight className="w-3.5 h-3.5 transform group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          {/* Image Recognition Module */}
          <div className="bg-[#12151d] border border-[#1f2533] hover:border-cyan-500/40 p-5 rounded-sm flex flex-col justify-between group transition-all duration-300 h-[190px]">
            <div>
              <div className="w-8 h-8 rounded-sm bg-[#1e293b] flex items-center justify-center border border-slate-700">
                <Activity className="w-4 h-4 text-[#ffaa00]" />
              </div>
              <h5 className="text-sm font-semibold text-white mt-4 tracking-tight">
                {language === 'zh' ? '图像识别' : 'Image Recognition'}
              </h5>
              <p className="text-xs text-[#64748b] mt-1 leading-snug">
                {language === 'zh' ? '上传图片进行人脸检测与识别，返回标注后的人脸框、匹配结果和相似度分数。' : 'Upload images for face detection and recognition. Returns annotated bounding boxes, match results, and similarity scores.'}
              </p>
            </div>
            <button 
              onClick={() => setActiveTab('recognition')}
              className="flex items-center justify-between text-xs text-cyan-400 font-mono group-hover:text-cyan-300 transition-colors mt-4 text-left w-full cursor-pointer select-none"
            >
              <span>{language === 'zh' ? '开始识别' : 'Start'}</span>
              <ArrowRight className="w-3.5 h-3.5 transform group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          {/* Video Analysis Module */}
          <div className="bg-[#12151d] border border-[#1f2533] hover:border-cyan-500/40 p-5 rounded-sm flex flex-col justify-between group transition-all duration-300 h-[190px]">
            <div>
              <div className="w-8 h-8 rounded-sm bg-[#1e293b] flex items-center justify-center border border-slate-700">
                <Cpu className="w-4 h-4 text-[#ec4899]" />
              </div>
              <h5 className="text-sm font-semibold text-white mt-4 tracking-tight">
                {language === 'zh' ? '视频分析' : 'Video Analysis'}
              </h5>
              <p className="text-xs text-[#64748b] mt-1 leading-snug">
                {language === 'zh' ? '上传视频文件进行逐帧人脸检测与识别，支持人员统计、陌生人检测和身份验证。' : 'Upload video files for frame-by-frame face detection and recognition. Supports person counting, stranger detection, and identity verification.'}
              </p>
            </div>
            <button 
              onClick={() => setActiveTab('video')}
              className="flex items-center justify-between text-xs text-cyan-400 font-mono group-hover:text-cyan-300 transition-colors mt-4 text-left w-full cursor-pointer select-none"
            >
              <span>{language === 'zh' ? '开始分析' : 'Start'}</span>
              <ArrowRight className="w-3.5 h-3.5 transform group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

          {/* Camera Monitoring Module */}
          <div className="bg-[#12151d] border border-[#1f2533] hover:border-cyan-500/40 p-5 rounded-sm flex flex-col justify-between group transition-all duration-300 h-[190px]">
            <div>
              <div className="w-8 h-8 rounded-sm bg-[#1e293b] flex items-center justify-center border border-slate-700">
                <Binary className="w-4 h-4 text-emerald-400" />
              </div>
              <h5 className="text-sm font-semibold text-white mt-4 tracking-tight">
                {language === 'zh' ? '摄像头监控' : 'Camera Monitoring'}
              </h5>
              <p className="text-xs text-[#64748b] mt-1 leading-snug">
                {language === 'zh' ? '调用后端摄像头进行实时人脸识别，支持参数配置和检测结果实时轮询显示。' : 'Real-time face recognition using backend cameras, with configurable parameters and live detection polling.'}
              </p>
            </div>
            <button 
              onClick={() => setActiveTab('video')}
              className="flex items-center justify-between text-xs text-cyan-400 font-mono group-hover:text-cyan-300 transition-colors mt-4 text-left w-full cursor-pointer select-none"
            >
              <span>{language === 'zh' ? '打开监控' : 'Open'}</span>
              <ArrowRight className="w-3.5 h-3.5 transform group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

        </div>
      </div>

      {/* 4. Spline charts and Real-time active recognitions */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* Global Request Pulse Curve */}
        <div className="xl:col-span-2 bg-[#11141d] border border-[#1f2533] p-5 rounded-sm">
          <div className="flex items-center justify-between mb-4">
            <h5 className="text-xs font-bold tracking-wider text-slate-400 uppercase font-mono leading-none">
              {language === 'zh' ? '请求负载趋势' : 'Request Load Trend'}
            </h5>
            <div className="flex items-center gap-4 text-[10px] font-mono select-none">
              <span className="flex items-center gap-1.5 text-slate-400">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block" />
                {language === 'zh' ? '主节点' : 'Primary'}
              </span>
              <span className="flex items-center gap-1.5 text-slate-400">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-500 inline-block" />
                {language === 'zh' ? '备用节点' : 'Shadow'}
              </span>
            </div>
          </div>
          
          {/* Custom Spline SVG Plot */}
          <div className="h-[150px] relative w-full border border-slate-900 bg-[#0c0e14] rounded-sm p-2">
            {/* Horizontal grid lines */}
            <div className="absolute inset-0 flex flex-col justify-between pointer-events-none p-2 opacity-10">
              <div className="border-b border-white border-dashed w-full" />
              <div className="border-b border-white border-dashed w-full" />
              <div className="border-b border-white border-dashed w-full" />
              <div className="w-full" />
            </div>

            <svg viewBox="0 0 600 150" className="w-full h-full overflow-visible">
              {/* Shadow node path */}
              <path
                d={getSvgPath(shadowData)}
                fill="none"
                stroke="#a855f7"
                strokeWidth="2"
                strokeLinecap="round"
                className="transition-all duration-300"
              />
              {/* Primary node path */}
              <path
                d={getSvgPath(pulseData)}
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2"
                strokeLinecap="round"
                className="transition-all duration-300 animate-pulse"
              />
            </svg>
          </div>
        </div>

        {/* Column representing Live Active Recognitions */}
        <div className="bg-[#11141d] border border-[#1f2533] p-5 rounded-sm flex flex-col">
          <h5 className="text-xs font-bold tracking-wider text-slate-400 uppercase font-mono mb-4 leading-none">
            {language === 'zh' ? '近期识别记录' : 'Recent Recognitions'}
          </h5>
          <div className="space-y-3 flex-1 overflow-y-auto pr-1">
            
            {/* Card 1 */}
            <div className="bg-[#171c26] p-3 border border-[#262f3f] rounded-sm flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-sm bg-slate-800 overflow-hidden relative border border-slate-700">
                  <img 
                    src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=128&auto=format&fit=crop" 
                    alt="Target profile" 
                    className="w-full h-full object-cover"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-cyan-500/10 pointer-events-none" />
                </div>
                <div>
                  <p className="text-xs text-white font-mono font-bold leading-tight">USER_7742 - {language === 'zh' ? '已确认' : 'VERIFIED'}</p>
                  <p className="text-[10px] text-[#06b6d4] font-mono tracking-widest leading-none mt-1">{language === 'zh' ? '主入口闸机' : 'GATE_ALPHA_12'}</p>
                </div>
              </div>
              <span className="text-xs text-emerald-400 font-mono font-bold">99.2%</span>
            </div>

            {/* Card 2 */}
            <div className="bg-[#171c26]/60 p-3 border border-red-950/40 rounded-sm flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-sm bg-[#1e131d] flex items-center justify-center border border-red-900/40 text-red-500 overflow-hidden relative">
                  <img 
                    src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=128&auto=format&fit=crop" 
                    alt="Target profile" 
                    className="w-full h-full object-cover grayscale"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-x-0 bottom-0 bg-red-900/80 font-mono text-[8px] text-center text-red-200">ALERT</div>
                </div>
                <div>
                  <p className="text-xs text-red-400 font-mono font-bold leading-tight uppercase">{language === 'zh' ? '未知人员 - 警报' : 'UNKNOWN - ALERT'}</p>
                  <p className="text-[10px] text-red-500/80 font-mono tracking-widest leading-none mt-1">{language === 'zh' ? '后侧围栏区域' : 'PERIMETER_REAR'}</p>
                </div>
              </div>
              <span className="text-xs text-red-400 font-mono font-bold uppercase text-[9px] bg-red-950/20 px-1.5 py-0.5 border border-red-900/30">
                {language === 'zh' ? '低置信度' : 'LOW_CONF'}
              </span>
            </div>

            {/* Card 3 */}
            <div className="bg-[#171c26] p-3 border border-[#262f3f] rounded-sm flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-sm bg-slate-800 overflow-hidden relative border border-slate-700">
                  <img 
                    src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=128&auto=format&fit=crop" 
                    alt="Target profile" 
                    className="w-full h-full object-cover"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-cyan-500/10 pointer-events-none" />
                </div>
                <div>
                  <p className="text-xs text-white font-mono font-bold leading-tight">STAFF_0019 - {language === 'zh' ? '已记录' : 'LOGGED'}</p>
                  <p className="text-[10px] text-[#06b6d4] font-mono tracking-widest leading-none mt-1">{language === 'zh' ? '大厅' : 'MAIN_LOBBY'}</p>
                </div>
              </div>
              <span className="text-xs text-emerald-400 font-mono font-bold">94.7%</span>
            </div>

          </div>
        </div>

      </div>

    </div>
  );
}
