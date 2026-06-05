/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Video,
  Cpu,
  Upload,
  Play,
  Square,
  ShieldAlert,
  Focus,
  Camera,
  AlertTriangle,
} from 'lucide-react';
import { LogLine, VideoRecognitionResponse, CameraStatusResponse, PersonCard } from '../types';
import { Language, LOCALES } from '../locales';
import { recognizeVideo } from '../api/recognition';
import { startCamera, stopCamera, getCameraStatus, clearCamera } from '../api/camera';
import { ApiError } from '../api/client';
import ProgressBar from './ProgressBar';

interface VideoAnalysisViewProps {
  addLog: (level: LogLine['level'], msg: string) => void;
  language?: Language;
}

type SourceMode = 'file' | 'camera';

export default function VideoAnalysisView({
  addLog,
  language = 'en'
}: VideoAnalysisViewProps) {
  const t = LOCALES[language];

  // ── Shared params ──
  const [skipFrames, setSkipFrames] = useState(5);
  const [threshold, setThreshold] = useState(45);
  const [stableFrames, setStableFrames] = useState(3);
  const [mode, setMode] = useState('视频中找人');
  const [verifyTarget, setVerifyTarget] = useState('');

  // ── Source mode ──
  const [sourceMode, setSourceMode] = useState<SourceMode>('file');

  // ── File mode state ──
  const [dragActive, setDragActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoResult, setVideoResult] = useState<VideoRecognitionResponse | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoPlayerRef = useRef<HTMLVideoElement>(null);

  // ── Camera mode state ──
  const [cameraId, setCameraId] = useState(0);
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<CameraStatusResponse | null>(null);
  const [cameraAccumulatedStats, setCameraAccumulatedStats] = useState<Record<string, number>>({});
  const [cameraAccumulatedFrames, setCameraAccumulatedFrames] = useState(0);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const startSimulatedProgress = useCallback(() => {
    if (progressRef.current) clearInterval(progressRef.current);
    setProgress(0);
    const start = Date.now();
    progressRef.current = setInterval(() => {
      const elapsed = (Date.now() - start) / 1000;
      // Fast initial climb to 20% in 0.8s, then slow to 90% over ~30s
      const pct = elapsed < 0.8
        ? (elapsed / 0.8) * 20
        : 20 + Math.min((elapsed - 0.8) / 30, 1) * 70;
      setProgress(Math.round(Math.min(pct, 90)));
    }, 200);
  }, []);

  const stopSimulatedProgress = useCallback(() => {
    if (progressRef.current) { clearInterval(progressRef.current); progressRef.current = null; }
  }, []);

  // ── File mode handlers ──

  const processVideo = async (file: File) => {
    setIsProcessing(true);
    setError(null);
    setVideoResult(null);
    setVideoUrl(null);
    startSimulatedProgress();
    addLog('INFO', `VIDEO_PIPELINE: Processing ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`);

    try {
      const data = await recognizeVideo(
        file,
        skipFrames,
        threshold / 100,
        stableFrames,
        mode,
        verifyTarget || undefined,
      );
      setProgress(100);
      setVideoResult(data);
      setVideoUrl(`data:video/mp4;base64,${data.video_base64}`);

      const totalPersons = Object.values(data.person_counts).reduce((a, b) => a + b, 0);
      addLog('RESULT', `VIDEO: ${data.processed_frames}/${data.total_frames} frames, ${totalPersons} persons, ${data.stranger_count} strangers, ${data.fps.toFixed(1)} FPS`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Video processing failed';
      setError(msg);
      addLog('ERROR', `VIDEO_PIPELINE: ${msg}`);
    } finally {
      setIsProcessing(false);
      stopSimulatedProgress();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      processVideo(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      const file = e.dataTransfer.files[0];
      setVideoFile(file);
      processVideo(file);
    }
  };

  // ── Camera mode handlers ──

  const toggleCamera = async () => {
    if (isCameraRunning) {
      // Stop
      if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
      try {
        await stopCamera();
        addLog('INFO', 'CAMERA: Stream stopped.');
      } catch {
        addLog('ERROR', 'CAMERA: Failed to stop stream.');
      }
      setIsCameraRunning(false);
      setCameraStatus(null);
    } else {
      // Start
      try {
        await clearCamera();
        setCameraAccumulatedStats({});
        setCameraAccumulatedFrames(0);
        const res = await startCamera({
          camera_id: cameraId,
          skip_frames: skipFrames,
          threshold: threshold / 100,
          stable_frames: stableFrames,
          mode,
          verify_target: verifyTarget || null,
        });
        setIsCameraRunning(true);
        addLog('INPUT_STREAM', `CAMERA: ${res.message}`);
        // Start polling
        pollingRef.current = setInterval(async () => {
          try {
            const status = await getCameraStatus();
            setCameraStatus(status);
            if (status.total_frames > 0) {
              setCameraAccumulatedFrames(prev => Math.max(prev, status.total_frames));
            }
            if (status.stats) {
              setCameraAccumulatedStats(prev => {
                const merged = { ...prev };
                for (const [name, count] of Object.entries(status.stats)) {
                  merged[name] = Math.max(merged[name] || 0, count);
                }
                return merged;
              });
            }
            if (!status.running) {
              setIsCameraRunning(false);
              if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
            }
          } catch {
            // Silently skip poll errors
          }
        }, 2000);
      } catch (err) {
        const msg = err instanceof ApiError ? err.detail : 'Camera start failed';
        addLog('ERROR', `CAMERA: ${msg}`);
      }
    }
  };

  const thresholdPercent = Math.round(threshold);

  return (
    <div className="p-6 grid grid-cols-1 xl:grid-cols-3 gap-6 overflow-y-auto h-[calc(100vh-4rem)] bg-[#090b11] text-slate-300 font-sans select-none">
      {/* LEFT: Controls */}
      <div className="space-y-6">
        {/* Source selection */}
        <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-4 font-mono text-xs">
          <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase leading-none">
            {language === 'zh' ? '视频源' : 'Video Source'}
          </h4>

          <div className="flex gap-2">
            <button
              onClick={() => setSourceMode('file')}
              className={`flex-1 py-2 text-xs font-bold uppercase tracking-wider rounded-sm border transition-all cursor-pointer ${
                sourceMode === 'file'
                  ? 'bg-cyan-950/30 border-cyan-500 text-cyan-400'
                  : 'border-[#1f2533] bg-slate-900/40 text-slate-500 hover:text-slate-300'
              }`}
            >
              <Upload className="w-3.5 h-3.5 inline mr-1" />
              {language === 'zh' ? '上传文件' : 'File'}
            </button>
            <button
              onClick={() => setSourceMode('camera')}
              className={`flex-1 py-2 text-xs font-bold uppercase tracking-wider rounded-sm border transition-all cursor-pointer ${
                sourceMode === 'camera'
                  ? 'bg-cyan-950/30 border-cyan-500 text-cyan-400'
                  : 'border-[#1f2533] bg-slate-900/40 text-slate-500 hover:text-slate-300'
              }`}
            >
              <Camera className="w-3.5 h-3.5 inline mr-1" />
              {language === 'zh' ? '摄像头' : 'Camera'}
            </button>
          </div>

          {sourceMode === 'file' ? (
            <>
              <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/*" style={{ display: 'none' }} />
              <div
                onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`aspect-video w-full rounded-sm border-2 border-dashed flex flex-col items-center justify-center p-6 text-center transition-all cursor-pointer ${
                  dragActive ? 'border-cyan-500 bg-cyan-500/5' : 'border-[#1e2330] hover:border-slate-700 bg-slate-900/60'
                }`}
              >
                <Video className="w-8 h-8 text-slate-500 mb-3" />
                <p className="text-xs font-bold text-white font-mono uppercase tracking-wider">
                  {language === 'zh' ? '拖曳或点击上传视频' : 'Drop or Click'}
                </p>
                <p className="text-[10px] text-slate-500 mt-2 font-mono uppercase">MP4, AVI, MOV</p>
              </div>
              {videoFile && (
                <p className="text-[10px] text-cyan-400 truncate">
                  {videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(1)} MB)
                </p>
              )}
            </>
          ) : (
            <div className="space-y-3">
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold">{language === 'zh' ? '摄像头ID' : 'Camera ID'}</label>
                <input
                  type="number" min="0" max="9" value={cameraId}
                  onChange={(e) => setCameraId(Number(e.target.value))}
                  className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500"
                />
              </div>
              <button
                onClick={toggleCamera}
                className={`w-full py-3 font-bold uppercase tracking-wider rounded-sm cursor-pointer transition-colors flex items-center justify-center gap-1.5 ${
                  isCameraRunning
                    ? 'bg-rose-950/40 hover:bg-rose-850/50 border border-rose-500/30 text-rose-200'
                    : 'bg-cyan-500 hover:bg-cyan-600 text-slate-950'
                }`}
              >
                {isCameraRunning ? (
                  <><Square className="w-4 h-4" /><span>{language === 'zh' ? '停止' : 'Stop'}</span></>
                ) : (
                  <><Play className="w-4 h-4 fill-current" /><span>{language === 'zh' ? '启动摄像头' : 'Start Camera'}</span></>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Parameters */}
        <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-4 font-mono text-xs">
          <h4 className="text-[10px] font-bold tracking-[0.2em] text-[#64748b] uppercase leading-none">
            {language === 'zh' ? '处理参数' : 'Parameters'}
          </h4>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 uppercase font-bold text-[10px]">{language === 'zh' ? '跳帧数' : 'Skip Frames'}</span>
              <span className="text-white font-extrabold">{skipFrames}</span>
            </div>
            <input
              type="range" min="0" max="30" value={skipFrames}
              onChange={(e) => setSkipFrames(Number(e.target.value))}
              className="w-full accent-cyan-500 h-1 bg-slate-800 rounded-sm cursor-pointer"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 uppercase font-bold text-[10px]">{language === 'zh' ? '相似度阈值' : 'Threshold'}</span>
              <span className="text-white font-extrabold">{(threshold / 100).toFixed(2)}</span>
            </div>
            <input
              type="range" min="0" max="99" value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full accent-cyan-500 h-1 bg-slate-800 rounded-sm cursor-pointer"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 uppercase font-bold text-[10px]">{language === 'zh' ? '稳定帧数' : 'Stable Frames'}</span>
              <span className="text-white font-extrabold">{stableFrames}</span>
            </div>
            <input
              type="range" min="1" max="15" value={stableFrames}
              onChange={(e) => setStableFrames(Number(e.target.value))}
              className="w-full accent-cyan-500 h-1 bg-slate-800 rounded-sm cursor-pointer"
            />
          </div>

          <div className="space-y-1">
            <label className="text-[10px] text-slate-500 uppercase font-bold">{language === 'zh' ? '识别模式' : 'Mode'}</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500"
            >
              <option value="视频中找人">{language === 'zh' ? '视频中找人' : 'Find Person'}</option>
              <option value="1:N人脸识别">{language === 'zh' ? '1:N人脸识别' : '1:N Recognition'}</option>
              <option value="陌生人检测">{language === 'zh' ? '陌生人检测' : 'Stranger Detection'}</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] text-slate-500 uppercase font-bold">
              {language === 'zh' ? '验证目标 (可选)' : 'Verify Target (optional)'}
            </label>
            <input
              type="text" value={verifyTarget}
              onChange={(e) => setVerifyTarget(e.target.value)}
              placeholder={language === 'zh' ? '输入人员姓名...' : 'Enter person name...'}
              className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-cyan-500 placeholder:text-slate-600"
            />
          </div>
        </div>

        {/* Error display */}
        {error && (
          <div className="bg-red-950/20 border border-red-500/30 rounded-sm p-4 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <p className="text-xs text-red-300 font-mono">{error}</p>
          </div>
        )}
      </div>

      {/* RIGHT: Video Player + Results */}
      <div className="xl:col-span-2 space-y-6">
        <div className="bg-[#121622] rounded-sm p-6 border border-[#1d2331] min-h-[440px] flex flex-col">
          <div className="flex justify-between items-center border-b border-[#262f3f]/40 pb-3">
            <h3 className="text-xs font-bold font-mono tracking-widest text-white uppercase flex items-center gap-2">
              <Focus className="w-4 h-4 text-cyan-400" />
              {language === 'zh' ? '视频分析渲染面板' : 'Video Analysis Renderer'}
            </h3>
            <div className="flex items-center gap-3">
              {videoResult && (
                <span className="text-[10px] text-slate-500 font-mono">
                  <span className="text-cyan-400 font-bold">{videoResult.processed_frames}</span>/{videoResult.total_frames} frames
                </span>
              )}
              {videoResult && (
                <button
                  onClick={() => { setVideoResult(null); setVideoUrl(null); setVideoFile(null); }}
                  className="text-[10px] font-mono uppercase tracking-wider text-slate-500 hover:text-red-400 border border-[#1f2533] hover:border-red-500/30 px-2 py-1 rounded-sm transition-colors cursor-pointer"
                >
                  {language === 'zh' ? '清空' : 'Clear'}
                </button>
              )}
            </div>
          </div>

          {/* Video display area */}
          <div className="flex-1 my-4 rounded-sm bg-[#0a0d14] border border-slate-900 overflow-hidden relative min-h-[300px] flex items-center justify-center">
            {isProcessing ? (
              <div className="flex flex-col items-center justify-center gap-3 w-2/3 max-w-sm">
                <ProgressBar value={progress} label={language === 'zh' ? '视频处理中...' : 'Processing Video...'} />
              </div>
            ) : videoUrl ? (
              <video
                ref={videoPlayerRef}
                src={videoUrl}
                controls
                className="w-full h-full object-contain"
              >
                {language === 'zh' ? '您的浏览器不支持视频标签' : 'Your browser does not support the video tag'}
              </video>
            ) : (
              <div className="text-center space-y-2 opacity-30 select-none">
                <Video className="w-12 h-12 text-slate-500 mx-auto" />
                <p className="text-xs font-mono uppercase tracking-widest text-slate-400">
                  {language === 'zh' ? '等待视频处理' : 'Awaiting video input'}
                </p>
                <p className="text-[10px] font-mono text-slate-500">
                  {sourceMode === 'file'
                    ? (language === 'zh' ? '上传视频文件开始分析' : 'Upload a video file to begin')
                    : (language === 'zh' ? '启动摄像头开始分析' : 'Start camera to begin')}
                </p>
              </div>
            )}
          </div>

          {/* Telemetry footer */}
          <div className="border-t border-[#1f2533] pt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-center font-mono text-xs">
            <div>
              <p className="text-[8px] text-slate-500 uppercase font-bold tracking-widest mb-1.5 leading-none">
                {language === 'zh' ? '帧率' : 'FPS'}
              </p>
              <span className="text-[10px] text-cyan-400 leading-none">
                {videoResult ? videoResult.fps.toFixed(1) : (cameraStatus ? cameraStatus.fps.toFixed(1) : '--')}
              </span>
            </div>
            <div>
              <p className="text-[8px] text-slate-500 uppercase font-bold tracking-widest mb-1.5 leading-none">
                {language === 'zh' ? '总帧数' : 'Total Frames'}
              </p>
              <span className="text-[10px] text-slate-300 leading-none">
                {videoResult ? videoResult.total_frames : (cameraStatus ? cameraStatus.total_frames : '--')}
              </span>
            </div>
            <div>
              <p className="text-[8px] text-slate-500 uppercase font-bold tracking-widest mb-1.5 leading-none">
                {language === 'zh' ? '检测人数' : 'Persons'}
              </p>
              <span className="text-[10px] text-slate-300 leading-none">
                {videoResult
                  ? Object.values(videoResult.person_counts).reduce((a: number, b: number) => a + b, 0)
                  : (cameraStatus ? (cameraStatus.stats ? Object.values(cameraStatus.stats).reduce((a: number, b: number) => a + b, 0) : '--') : '--')}
              </span>
            </div>
            <div>
              <p className="text-[8px] text-slate-500 uppercase font-bold tracking-widest mb-1.5 leading-none">
                {language === 'zh' ? '陌生人' : 'Strangers'}
              </p>
              <span className="text-[10px] text-amber-500 leading-none">
                {videoResult ? videoResult.stranger_count : '--'}
              </span>
            </div>
          </div>
        </div>

        {/* Person counts breakdown */}
        {videoResult && Object.keys(videoResult.person_counts).length > 0 && (
          <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-3">
            <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase font-mono leading-none">
              {language === 'zh' ? '人员出现次数' : 'Person Appearances'}
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(videoResult.person_counts).map(([name, count]) => (
                <div key={name} className="bg-[#171c26] border border-[#2a3447] rounded-sm px-3 py-2 flex justify-between items-center">
                  <span className="text-[10px] font-mono text-slate-300">{name}</span>
                  <span className="text-[10px] font-mono font-bold text-cyan-400">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Person cards from video */}
        {videoResult && videoResult.person_cards && videoResult.person_cards.length > 0 && (
          <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-3">
            <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase font-mono leading-none">
              {language === 'zh' ? '人物资料卡' : 'Person Cards'}
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {videoResult.person_cards.map((card, i) => {
                const faceImg = card.gallery_face_base64;
                const genderLabel = card.gender === 'male' ? '男' : card.gender === 'female' ? '女' : '未指定';
                return (
                  <div key={i} className="bg-[#171c26] border border-[#2a3447] rounded-sm p-3 flex gap-3 items-start">
                    <div className="w-12 h-12 rounded-sm bg-slate-800 border border-[#2a3447] overflow-hidden shrink-0 flex items-center justify-center">
                      {faceImg ? (
                        <img src={`data:image/jpeg;base64,${faceImg}`} alt={card.name} className="w-full h-full object-cover" />
                      ) : (
                        <span className="text-[8px] text-slate-500 font-mono text-center leading-tight">{language === 'zh' ? '暂无' : 'N/A'}</span>
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-[10px] text-white font-mono font-bold truncate">{card.name}</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        <span className="text-[8px] text-slate-500 bg-slate-800 px-1 py-0.5 rounded-sm font-mono">{genderLabel}</span>
                        {card.birth_date && (
                          <span className="text-[8px] text-slate-500 bg-slate-800 px-1 py-0.5 rounded-sm font-mono">{card.birth_date}</span>
                        )}
                        <span className="text-[8px] text-cyan-400 bg-cyan-950/30 px-1 py-0.5 rounded-sm font-mono">{card.embedding_count} imgs</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Camera accumulated stats */}
        {Object.keys(cameraAccumulatedStats).length > 0 && (
          <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase font-mono leading-none">
                {language === 'zh' ? '累计识别人员' : 'Accumulated Persons'}
              </h4>
              <button
                onClick={async () => {
                  try { await clearCamera(); } catch {}
                  setCameraAccumulatedStats({});
                  setCameraAccumulatedFrames(0);
                  setCameraStatus(null);
                }}
                className="text-[10px] font-mono uppercase tracking-wider text-slate-500 hover:text-red-400 border border-[#1f2533] hover:border-red-500/30 px-2 py-1 rounded-sm transition-colors cursor-pointer"
              >
                {language === 'zh' ? '清空' : 'Clear'}
              </button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(cameraAccumulatedStats).map(([name, count]) => (
                <div key={name} className="bg-[#171c26] border border-[#2a3447] rounded-sm px-3 py-2 flex justify-between items-center">
                  <span className="text-[10px] font-mono text-slate-300">{name}</span>
                  <span className="text-[10px] font-mono font-bold text-cyan-400">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Camera live person cards */}
        {cameraStatus && cameraStatus.person_cards && cameraStatus.person_cards.length > 0 && (
          <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-3">
            <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase font-mono leading-none">
              {language === 'zh' ? '实时资料卡' : 'Live Person Cards'}
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {cameraStatus.person_cards.map((card, i) => {
                const faceImg = card.gallery_face_base64;
                const genderLabel = card.gender === 'male' ? '男' : card.gender === 'female' ? '女' : '未指定';
                return (
                  <div key={i} className={`bg-[#171c26] border rounded-sm p-3 flex gap-3 items-start ${
                    card.accepted ? 'border-emerald-500/30' : 'border-amber-500/30'
                  }`}>
                    <div className="w-14 h-14 rounded-sm bg-slate-800 border border-[#2a3447] overflow-hidden shrink-0 flex items-center justify-center">
                      {faceImg ? (
                        <img src={`data:image/jpeg;base64,${faceImg}`} alt={card.name} className="w-full h-full object-cover" />
                      ) : (
                        <span className="text-[9px] text-slate-500 font-mono text-center leading-tight">{language === 'zh' ? '暂无' : 'N/A'}</span>
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-xs text-white font-mono font-bold truncate">{card.name}</p>
                      <p className={`text-[10px] font-mono font-bold mt-0.5 ${
                        card.score >= 0.7 ? 'text-emerald-400' : card.score >= 0.4 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {(card.score * 100).toFixed(1)}%
                      </p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        <span className="text-[8px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded-sm font-mono">{genderLabel}</span>
                        {card.birth_date && (
                          <span className="text-[8px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded-sm font-mono">{card.birth_date}</span>
                        )}
                        <span className="text-[8px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded-sm font-mono">{card.embedding_count} samples</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Verify result */}
        {videoResult?.verify_result && (
          <div className={`rounded-sm p-4 border ${
            videoResult.verify_result.passed
              ? 'bg-emerald-950/10 border-emerald-500/30'
              : 'bg-red-950/10 border-red-500/30'
          }`}>
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className={videoResult.verify_result.passed ? 'text-emerald-400' : 'text-red-400'}>
                {videoResult.verify_result.passed ? 'VERIFIED' : 'NOT FOUND'}
              </span>
              <span className="text-slate-500">
                Target: <span className="text-white font-bold">{videoResult.verify_result.target}</span>
              </span>
              <span className="text-slate-500">
                Frames: <span className="text-white">{videoResult.verify_result.frames}</span>
              </span>
            </div>
          </div>
        )}

        {/* Camera live results */}
        {isCameraRunning && cameraStatus && cameraStatus.results.length > 0 && (
          <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-3 max-h-[240px] overflow-y-auto">
            <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase font-mono leading-none">
              {language === 'zh' ? '实时检测结果' : 'Live Detections'}
            </h4>
            {cameraStatus.results.map((r, i) => (
              <div key={i} className="flex items-center justify-between bg-[#171c26] p-2 border border-[#2a3447] rounded-sm text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${r.accepted ? 'bg-emerald-400' : 'bg-red-400'}`} />
                  <span className="text-white font-bold">{r.display_name || r.name}</span>
                </div>
                <span className={r.accepted ? 'text-emerald-400' : 'text-red-400'}>
                  {(r.score * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Log feed */}
        <div className="bg-[#121622] border border-[#1d2331] rounded-sm p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-[10px] font-bold tracking-[0.2em] text-[#64748b] uppercase font-mono leading-none">
              {language === 'zh' ? '系统日志' : 'Live_Log_Feed'}
            </h4>
            <span className="text-[9px] font-mono text-rose-500 font-extrabold uppercase bg-rose-950/20 border border-rose-900/40 py-0.5 px-2 rounded-sm">
              <ShieldAlert className="w-3 h-3 text-rose-500 fill-rose-500 inline mr-1" />
              {language === 'zh' ? '监视中' : 'Monitor'}
            </span>
          </div>
          <div className="bg-[#0b0e14] border border-[#1e232c] rounded-sm p-4 text-[10px] font-mono text-slate-400 space-y-2 h-[120px] overflow-y-auto">
            {/* Log lines come from parent via addLog - displayed elsewhere. Show camera status */}
            <div className="flex gap-2 leading-relaxed">
              <span className="text-[10px] text-cyan-400 font-extrabold">[SYS]</span>
              <span>
                {sourceMode === 'camera'
                  ? (isCameraRunning
                      ? `Camera #${cameraId} active — ${cameraStatus ? `${cameraStatus.fps.toFixed(1)} FPS` : 'polling...'}`
                      : `Camera #${cameraId} standby`)
                  : (videoFile ? `${videoFile.name} loaded` : (language === 'zh' ? '等待视频上传...' : 'Awaiting video upload...'))}
              </span>
            </div>
            {cameraStatus?.verify_status && (
              <div className="flex gap-2 leading-relaxed">
                <span className="text-[10px] text-amber-400 font-extrabold">[VERIFY]</span>
                <span>{cameraStatus.verify_status} ({cameraStatus.verify_count} frames)</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
