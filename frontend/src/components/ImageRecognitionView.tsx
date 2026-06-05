/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  Upload,
  Cpu,
  Zap,
  Camera,
  Activity,
  ShieldAlert,
  Lock,
  AlertTriangle,
} from 'lucide-react';
import { LogLine, ImageRecognitionResponse, PersonCard } from '../types';
import { Language, LOCALES } from '../locales';
import { recognizeImage } from '../api/recognition';
import { ApiError } from '../api/client';
import ProgressBar from './ProgressBar';

interface ImageRecognitionViewProps {
  logs: LogLine[];
  addLog: (level: LogLine['level'], msg: string) => void;
  language?: Language;
}

export default function ImageRecognitionView({
  logs,
  addLog,
  language = 'en'
}: ImageRecognitionViewProps) {
  const [dragActive, setDragActive] = useState(false);
  const [confidence, setConfidence] = useState<number>(45);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [queryImage, setQueryImage] = useState<string | null>(null);
  const [result, setResult] = useState<ImageRecognitionResponse | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startSimulatedProgress = useCallback(() => {
    if (progressRef.current) clearInterval(progressRef.current);
    setProgress(0);
    const start = Date.now();
    progressRef.current = setInterval(() => {
      const elapsed = (Date.now() - start) / 1000;
      // Fast initial climb to 30% in 0.5s, then slow to 90% over ~15s
      const pct = elapsed < 0.5
        ? (elapsed / 0.5) * 30
        : 30 + Math.min((elapsed - 0.5) / 15, 1) * 60;
      setProgress(Math.round(Math.min(pct, 90)));
    }, 150);
  }, []);

  const stopSimulatedProgress = useCallback(() => {
    if (progressRef.current) { clearInterval(progressRef.current); progressRef.current = null; }
  }, []);

  const runRecognition = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    startSimulatedProgress();
    addLog('INFO', `ANALYSIS_CORE: Running recognition with threshold ${(confidence / 100).toFixed(2)}`);
    try {
      const data = await recognizeImage(file, confidence / 100);
      setProgress(100);
      setResult(data);
      const accepted = data.results.filter(r => r.accepted).length;
      addLog('RESULT', `RECOGNITION: ${data.results.length} face(s) found, ${accepted} accepted.`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Recognition failed';
      setError(msg);
      addLog('ERROR', `RECOGNITION: ${msg}`);
    } finally {
      setIsAnalyzing(false);
      stopSimulatedProgress();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (ev.target?.result) {
          setQueryImage(ev.target.result as string);
        }
      };
      reader.readAsDataURL(file);
      runRecognition(file);
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (ev.target?.result) setQueryImage(ev.target.result as string);
      };
      reader.readAsDataURL(file);
      runRecognition(file);
    }
  };

  const toggleWebcam = async () => {
    if (isCameraActive) {
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      setIsCameraActive(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 400, height: 400 } });
        if (videoRef.current) videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsCameraActive(true);
        addLog('INPUT_STREAM', 'SENSOR: Camera stream active.');
      } catch {
        addLog('ERROR', 'SYSTEM_CORE: Camera access denied.');
      }
    }
  };

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
            setQueryImage(canvas.toDataURL('image/jpeg'));
            toggleWebcam();
            runRecognition(file);
          }
        }, 'image/jpeg');
      }
    }
  };

  const topResult = result?.results?.[0];
  const annotatedImage = result?.annotated_image_base64;
  // Build lookup: person name → gallery face base64 from matched_pairs
  const galleryFaces: Record<string, string> = {};
  if (result?.matched_pairs) {
    result.matched_pairs.forEach(p => {
      if (p.gallery_face_base64) galleryFaces[p.name] = p.gallery_face_base64;
    });
  }
  const topGalleryFace = topResult ? (galleryFaces[topResult.name] || galleryFaces[topResult.display_name]) : null;

  const thresholdPercent = Math.round(confidence);

  return (
    <div className="p-6 grid grid-cols-1 xl:grid-cols-3 gap-6 overflow-y-auto h-[calc(100vh-4rem)] bg-[#090b11] text-slate-300 font-sans select-none">
      {/* LEFT: Settings and upload */}
      <div className="space-y-6">
        <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-5">
          <div className="flex items-center justify-between">
            <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase font-mono leading-none">
              {language === 'zh' ? '图像上传' : 'UPLOAD_SOURCE'}
            </h4>
            <button
              onClick={toggleWebcam}
              className={`p-1.5 rounded-sm border transition-all flex items-center gap-1.5 text-xs font-mono cursor-pointer ${
                isCameraActive
                  ? 'bg-red-950/30 border-red-500/50 text-red-400 font-bold'
                  : 'bg-slate-800/80 border-slate-700 text-cyan-400 hover:bg-slate-700'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              <span>{isCameraActive ? (language === 'zh' ? '关闭摄像头' : 'Disconnect') : (language === 'zh' ? '摄像头' : 'Camera')}</span>
            </button>
          </div>

          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />

          {isCameraActive ? (
            <div className="relative aspect-square w-full rounded-sm border border-[#06b6d4]/40 bg-black/80 flex flex-col justify-end overflow-hidden p-4">
              <video ref={videoRef} autoPlay playsInline className="absolute inset-0 w-full h-full object-cover scale-x-[-1]" />
              <div className="absolute inset-0 border border-cyan-500/30 pointer-events-none">
                <div className="absolute top-5 left-5 w-6 h-6 border-t-2 border-l-2 border-[#06b6d4]" />
                <div className="absolute top-5 right-5 w-6 h-6 border-t-2 border-r-2 border-[#06b6d4]" />
                <div className="absolute bottom-5 left-5 w-6 h-6 border-b-2 border-l-2 border-[#06b6d4]" />
                <div className="absolute bottom-5 right-5 w-6 h-6 border-b-2 border-r-2 border-[#06b6d4]" />
              </div>
              <button onClick={captureFrame} className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold uppercase text-xs tracking-wider z-10 rounded-sm cursor-pointer font-mono">
                {language === 'zh' ? '拍摄快照' : 'Capture Snapshot'}
              </button>
            </div>
          ) : (
            <div
              onDragEnter={handleDrag} onDragOver={handleDrag} onDragLeave={handleDrag} onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`aspect-square w-full rounded-sm border-2 border-dashed flex flex-col items-center justify-center p-6 text-center transition-all cursor-pointer ${
                dragActive ? 'border-cyan-500 bg-cyan-500/5' : 'border-[#1e2330] hover:border-slate-700 bg-slate-900/60'
              }`}
            >
              <Upload className="w-8 h-8 text-slate-500 mb-3" />
              <p className="text-xs font-bold text-white font-mono uppercase tracking-wider">
                {language === 'zh' ? '拖曳或点击上传' : 'Drop or Click'}
              </p>
              <p className="text-[10px] text-slate-500 mt-2 font-mono uppercase">JPG, PNG</p>
            </div>
          )}

          <canvas ref={canvasRef} style={{ display: 'none' }} />

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-500 uppercase font-bold tracking-wider">
                {language === 'zh' ? '相似度阈值' : 'Threshold'}
              </span>
              <span className="text-white font-extrabold">{(confidence / 100).toFixed(2)}</span>
            </div>
            <input
              type="range" min="0" max="99" value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              className="w-full accent-cyan-500 h-1 bg-slate-800 rounded-sm cursor-pointer"
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

      {/* RIGHT: Results */}
      <div className="xl:col-span-2 space-y-6">
        <div className="bg-[#121622] rounded-sm p-6 border border-[#1d2331] space-y-6 flex flex-col min-h-[440px]">
          <div className="flex justify-between items-center border-b border-[#262f3f]/40 pb-3">
            <h3 className="text-xs font-bold font-mono tracking-widest text-white uppercase flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              {language === 'zh' ? '识别结果' : 'Recognition Results'}
            </h3>
            {result && (
              <button
                onClick={() => { setResult(null); setQueryImage(null); }}
                className="text-[10px] font-mono uppercase tracking-wider text-slate-500 hover:text-red-400 border border-[#1f2533] hover:border-red-500/30 px-2 py-1 rounded-sm transition-colors cursor-pointer"
              >
                {language === 'zh' ? '清空' : 'Clear'}
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative">
            {/* Query Image */}
            <div className="border border-[#232a35] bg-black/60 p-4 rounded-sm flex flex-col min-h-[280px] relative overflow-hidden">
              <div className="flex justify-between font-mono text-[10px] leading-none mb-3 text-slate-400 z-10">
                <span>{language === 'zh' ? '查询图像' : 'QUERY_IMAGE'}</span>
              </div>
              <div className="flex-1 flex items-center justify-center overflow-hidden relative border border-slate-900/60 rounded-sm bg-[#121622]/40">
                {annotatedImage ? (
                  <img src={`data:image/jpeg;base64,${annotatedImage}`} alt="Annotated result" className="w-full h-full object-contain" />
                ) : queryImage ? (
                  <img src={queryImage} alt="Query" className="w-full h-full object-cover" />
                ) : (
                  <div className="text-center font-mono opacity-20 p-4">
                    <Activity className="w-12 h-12 text-slate-500 mx-auto mb-2" />
                    <p className="text-[10px] uppercase">{language === 'zh' ? '等待上传图像' : 'Awaiting input'}</p>
                  </div>
                )}
                {isAnalyzing && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/70 px-8">
                    <ProgressBar value={progress} label={language === 'zh' ? '识别中...' : 'Processing...'} />
                  </div>
                )}
              </div>
            </div>

            {/* Match Result */}
            <div className="border border-[#232a35] bg-black/60 p-4 rounded-sm flex flex-col relative overflow-hidden">
              <div className="flex justify-between font-mono text-[10px] leading-none mb-3 text-slate-400 z-10">
                <span>{language === 'zh' ? '匹配结果' : 'MATCH_RESULT'}</span>
                {topResult && (
                  <span className={`${topResult.accepted ? 'text-emerald-400' : 'text-red-400'}`}>
                    {topResult.accepted ? 'ACCEPTED' : 'REJECTED'}
                  </span>
                )}
              </div>
              <div className="flex-1 min-h-[220px] flex items-center justify-center overflow-hidden relative border border-slate-900/60 rounded-sm bg-[#121622]/40">
                {topResult ? (
                  <>
                    {topGalleryFace ? (
                      <img
                        src={`data:image/jpeg;base64,${topGalleryFace}`}
                        alt={topResult.display_name}
                        className="max-w-full max-h-full object-contain p-2"
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center gap-2">
                        <span className="text-cyan-400 font-mono text-lg font-bold">{topResult.display_name}</span>
                        <span className="text-slate-400 font-mono text-sm">Score: {(topResult.score * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    <div className="absolute inset-0 pointer-events-none">
                      <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-cyan-400/80" />
                      <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-cyan-400/80" />
                      <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-cyan-400/80" />
                      <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-cyan-400/80" />
                    </div>
                  </>
                ) : (
                  <div className="text-center font-mono opacity-20 p-4">
                    <Lock className="w-12 h-12 text-slate-500 mx-auto mb-2" />
                    <p className="text-[10px] uppercase">{language === 'zh' ? '等待识别结果' : 'Awaiting result'}</p>
                  </div>
                )}
              </div>
              {/* 信息栏：移到图片下方，不再遮挡人脸 */}
              {topResult && !isAnalyzing && (
                <div className="mt-3 bg-[#0f1319] border border-[#232a35] rounded-sm px-4 py-2.5 flex items-center justify-between">
                  <div>
                    <p className="text-xs text-cyan-400 font-mono font-bold">{topResult.display_name}</p>
                    <p className={`text-[10px] font-mono font-bold mt-0.5 ${
                      topResult.accepted ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {topResult.label}
                    </p>
                  </div>
                  <span className="text-xl font-black font-mono text-white">
                    {(topResult.score * 100).toFixed(0)}<span className="text-sm text-slate-400">%</span>
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Results capsules */}
          {result && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-[#171c26] p-3 text-center border border-[#2a3447] rounded-sm">
                <span className="block text-[8px] font-mono text-slate-500 uppercase tracking-widest mb-1 font-bold">
                  {language === 'zh' ? '检测到人脸' : 'Faces Detected'}
                </span>
                <span className="text-[10px] font-mono text-white">{result.results.length}</span>
              </div>
              <div className="bg-[#171c26] p-3 text-center border border-[#2a3447] rounded-sm">
                <span className="block text-[8px] font-mono text-slate-500 uppercase tracking-widest mb-1 font-bold">
                  {language === 'zh' ? '通过阈值' : 'Accepted'}
                </span>
                <span className="text-[10px] font-mono text-emerald-400">{result.results.filter(r => r.accepted).length}</span>
              </div>
              <div className="bg-[#171c26] p-3 text-center border border-[#2a3447] rounded-sm">
                <span className="block text-[8px] font-mono text-slate-500 uppercase tracking-widest mb-1 font-bold">
                  {language === 'zh' ? '匹配对数' : 'Matched Pairs'}
                </span>
                <span className="text-[10px] font-mono text-cyan-400">{result.matched_pairs.length}</span>
              </div>
            </div>
          )}

          {/* Person cards */}
          {result && result.person_cards && result.person_cards.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase font-mono leading-none">
                {language === 'zh' ? '人物资料卡' : 'Person Cards'}
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {result.person_cards.map((card, i) => {
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
                          <span className="text-[9px] text-slate-500 font-mono text-center leading-tight px-1">{language === 'zh' ? '暂无照片' : 'No Photo'}</span>
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

          {/* All results list */}
          {result && result.results.length > 0 && (
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {result.results.map((r, i) => {
                const face = galleryFaces[r.name] || galleryFaces[r.display_name];
                return (
                  <div key={i} className="flex items-center justify-between bg-[#171c26] p-2 border border-[#2a3447] rounded-sm text-xs font-mono">
                    <div className="flex items-center gap-3 min-w-0">
                      <span className={`w-2 h-2 rounded-full shrink-0 ${r.accepted ? 'bg-emerald-400' : 'bg-red-400'}`} />
                      {face ? (
                        <img src={`data:image/jpeg;base64,${face}`} alt={r.display_name}
                          className="w-8 h-8 rounded-sm object-cover border border-[#2a3447] shrink-0" />
                      ) : (
                        <div className="w-8 h-8 rounded-sm bg-slate-800 border border-[#2a3447] flex items-center justify-center shrink-0">
                          <span className="text-[9px] text-slate-500 font-bold">{(r.display_name || r.name)[0]}</span>
                        </div>
                      )}
                      <div className="min-w-0">
                        <span className="text-white font-bold truncate block">{r.display_name || r.name}</span>
                        <span className="text-[9px] text-slate-500">[{r.box.map(b => Math.round(b)).join(', ')}]</span>
                      </div>
                    </div>
                    <span className={`shrink-0 ml-2 ${r.accepted ? 'text-emerald-400' : 'text-red-400'}`}>
                      {(r.score * 100).toFixed(1)}% — {r.label}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

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
            {logs.slice(-6).map((item, idx) => {
              const cls =
                item.level === 'WARNING' ? 'text-rose-400' :
                item.level === 'ERROR' ? 'text-red-500 font-black' :
                item.level === 'RESULT' ? 'text-emerald-400' :
                item.level === 'AI_PROC' ? 'text-cyan-400' :
                'text-slate-400';
              return (
                <div key={idx} className="flex gap-2 leading-relaxed">
                  <span className="text-[10px] text-slate-600 select-none">[{item.timestamp}]</span>
                  <span className={`font-extrabold text-[9px] ${cls}`}>{item.level}:</span>
                  <span>{item.message}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
