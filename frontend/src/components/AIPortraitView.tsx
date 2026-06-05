/**
 * AI Portrait Generation View
 *
 * PhotoMaker V2 + Stable Diffusion XL 的人物肖像风格转换页面。
 * 从数据库选择已注册人员 → 选择风格 → 生成 AI 肖像 → 展示/下载。
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Wand2,
  User,
  Palette,
  ChevronDown,
  Download,
  Trash2,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Image as ImageIcon,
  Info,
  Clock,
} from 'lucide-react';
import { Language, LOCALES } from '../locales';
import { LogLine, PortraitPersonItem, PortraitStyleItem, PortraitGenerateResponse } from '../types';
import * as portraitApi from '../api/portrait';
import { ApiError } from '../api/client';
import LoadingSpinner from './LoadingSpinner';
import ErrorState from './ErrorState';
import EmptyState from './EmptyState';

// ── 风格元数据 (fallback 当 API 不可用时) ──
const FALLBACK_STYLES: PortraitStyleItem[] = [
  { key: 'business', label: '💼 商务照' },
  { key: 'id_photo', label: '🪪 证件照' },
  { key: 'ancient_chinese', label: '🏮 古风写真' },
  { key: 'anime', label: '🎨 动漫风格' },
  { key: 'cyberpunk', label: '🤖 赛博朋克' },
  { key: 'professional', label: '👔 职业形象照' },
];

interface AIPortraitViewProps {
  addLog: (level: LogLine['level'], msg: string) => void;
  language?: Language;
}

export default function AIPortraitView({ addLog, language = 'zh' }: AIPortraitViewProps) {
  const t = LOCALES[language];

  // ── 数据状态 ──
  const [persons, setPersons] = useState<PortraitPersonItem[]>([]);
  const [styles, setStyles] = useState<PortraitStyleItem[]>(FALLBACK_STYLES);
  const [loadingPersons, setLoadingPersons] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // ── 选择状态 ──
  const [selectedPerson, setSelectedPerson] = useState<string>('');
  const [originalImageB64, setOriginalImageB64] = useState<string>('');
  const [serverImagePath, setServerImagePath] = useState<string>('');
  const [imagePaths, setImagePaths] = useState<string[]>([]);
  const [loadingImage, setLoadingImage] = useState(false);
  const [selectedStyle, setSelectedStyle] = useState<string>('business');

  // ── 高级选项 ──
  const [seed, setSeed] = useState<number>(-1);
  const [numSteps, setNumSteps] = useState<number>(50);
  const [guidance, setGuidance] = useState<number>(5.0);
  const [mergeStep, setMergeStep] = useState<number>(10);

  // ── 生成状态 ──
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<PortraitGenerateResponse | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [unloading, setUnloading] = useState(false);
  const [unloadedMsg, setUnloadedMsg] = useState<string | null>(null);

  // ── 加载人员列表和风格列表 ──
  const fetchData = useCallback(async () => {
    setLoadingPersons(true);
    setError(null);
    try {
      const [personsRes, stylesRes] = await Promise.allSettled([
        portraitApi.getPersons(),
        portraitApi.getStyles(),
      ]);

      if (personsRes.status === 'fulfilled') {
        const withPhotos = (personsRes.value.persons || []).filter(
          (p) => p.image_paths && p.image_paths.length > 0
        );
        setPersons(withPhotos);
        if (withPhotos.length > 0 && !selectedPerson) {
          setSelectedPerson(withPhotos[0].name);
        }
        addLog('RESULT', `Loaded ${withPhotos.length} registered persons with photos`);
      } else {
        setError('Failed to load persons');
      }

      if (stylesRes.status === 'fulfilled' && stylesRes.value.styles?.length > 0) {
        setStyles(stylesRes.value.styles);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : 'Failed to load data');
      addLog('ERROR', `Portrait init: ${err}`);
    } finally {
      setLoadingPersons(false);
    }
  }, [addLog]);

  // ── 检查模型状态 ──
  const checkStatus = useCallback(async () => {
    try {
      const status = await portraitApi.getStatus();
      setModelLoaded(status.loaded);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    fetchData();
    checkStatus();
  }, [fetchData, checkStatus]);

  // ── 加载选定人员的原始照片 ──
  useEffect(() => {
    if (!selectedPerson) return;
    setLoadingImage(true);
    setError(null);
    portraitApi
      .getPersonImage(selectedPerson)
      .then((data) => {
        setOriginalImageB64(data.image_base64);
        setServerImagePath(data.image_path);
        const person = persons.find((p) => p.name === selectedPerson);
        setImagePaths(person?.image_paths || []);
        addLog('INFO', `Loaded photo for: ${selectedPerson}`);
      })
      .catch((err) => {
        setError(err instanceof ApiError ? err.detail : 'Failed to load photo');
        addLog('ERROR', `Load photo: ${err}`);
      })
      .finally(() => setLoadingImage(false));
  }, [selectedPerson, persons, addLog]);

  // ── 生成 ──
  const handleGenerate = useCallback(async () => {
    if (!serverImagePath || !selectedStyle) return;
    setGenerating(true);
    setError(null);
    setResult(null);
    setUnloadedMsg(null);
    addLog('AI_PROC', `Starting portrait generation: ${selectedPerson} → ${selectedStyle}`);

    try {
      const res = await portraitApi.generate({
        person_name: selectedPerson,
        image_path: serverImagePath,
        style: selectedStyle,
        seed: seed <= 0 ? null : seed,
        num_inference_steps: numSteps,
        guidance_scale: guidance,
        start_merge_step: mergeStep,
      });
      setResult(res);
      setModelLoaded(true);
      addLog('RESULT', `Portrait generated in ${res.generation_time_seconds}s (seed=${res.seed})`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Generation failed';
      setError(msg);
      addLog('ERROR', `Portrait generation: ${msg}`);
    } finally {
      setGenerating(false);
    }
  }, [selectedPerson, serverImagePath, selectedStyle, seed, numSteps, guidance, mergeStep, addLog]);

  // ── 卸载模型 ──
  const handleUnload = useCallback(async () => {
    setUnloading(true);
    try {
      await portraitApi.unloadModel();
      setModelLoaded(false);
      setUnloadedMsg(t.portrait.unloadedMsg);
      addLog('INFO', 'PhotoMaker pipeline unloaded, GPU memory freed');
    } catch (err) {
      addLog('ERROR', `Unload failed: ${err}`);
    } finally {
      setUnloading(false);
    }
  }, [t, addLog]);

  // ── 渲染 ──
  return (
    <div className="p-6 space-y-6 overflow-y-auto h-[calc(100vh-4rem)] bg-[#090b11] text-slate-300 font-sans select-none">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white font-mono tracking-wide flex items-center gap-2">
            <Wand2 className="w-5 h-5 text-cyan-400" />
            {t.portrait.title}
          </h2>
          <p className="text-xs text-slate-500 mt-1">{t.portrait.subtitle}</p>
        </div>
        {/* 模型状态指示器 */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-sm bg-[#121622] border border-[#1f2533] text-xs">
          <div className={`w-2 h-2 rounded-full ${modelLoaded ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
          <span className="text-slate-400 font-mono">
            {modelLoaded ? 'PhotoMaker Ready' : 'Model Offline'}
          </span>
        </div>
      </div>

      {/* 加载态 */}
      {loadingPersons && <LoadingSpinner />}

      {/* 错误态 */}
      {!loadingPersons && error && !result && (
        <ErrorState message={error} onRetry={fetchData} />
      )}

      {/* 空人员态 */}
      {!loadingPersons && !error && persons.length === 0 && (
        <EmptyState message={t.portrait.noPersons} />
      )}

      {/* 主内容 */}
      {!loadingPersons && persons.length > 0 && (
        <>
          {/* ── 步骤 1+2: 选择人员 + 原始照片 ── */}
          <div className="grid grid-cols-3 gap-6">
            {/* Person selector */}
            <div className="col-span-1 space-y-4">
              <div className="bg-[#121622] border border-[#1f2533] rounded-sm p-4">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 font-mono">
                  {t.portrait.step1}
                </h3>
                <div className="relative">
                  <select
                    value={selectedPerson}
                    onChange={(e) => setSelectedPerson(e.target.value)}
                    className="w-full bg-[#090b11] border border-[#1f2533] rounded-sm px-3 py-2.5 text-sm text-slate-200
                               focus:outline-none focus:border-cyan-500/50 appearance-none cursor-pointer font-mono"
                  >
                    {persons.map((p) => (
                      <option key={p.name} value={p.name}>
                        {p.name} ({p.image_paths.length} photos)
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
                </div>
              </div>

              {/* Person info */}
              <div className="bg-[#121622] border border-[#1f2533] rounded-sm p-4 space-y-2">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">
                  {t.portrait.step2}
                </h3>
                {loadingImage ? (
                  <div className="flex items-center justify-center h-40">
                    <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                  </div>
                ) : originalImageB64 ? (
                  <div className="relative group">
                    <img
                      src={`data:image/jpeg;base64,${originalImageB64}`}
                      alt={selectedPerson}
                      className="w-full aspect-square object-cover rounded-sm border border-[#1f2533]"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded-sm flex items-end p-2">
                      <span className="text-[10px] text-white font-mono truncate">{selectedPerson}</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-40 text-slate-600 text-xs">
                    {t.portrait.noPhoto}
                  </div>
                )}
                <div className="text-[10px] text-slate-600 font-mono space-y-0.5">
                  <p>{t.portrait.photosCount}: {imagePaths.length}</p>
                  <p className="truncate" title={serverImagePath}>
                    {t.portrait.serverPath}: {serverImagePath}
                  </p>
                </div>
              </div>
            </div>

            {/* ── 步骤 3: 风格选择 ── */}
            <div className="col-span-2 bg-[#121622] border border-[#1f2533] rounded-sm p-4">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4 font-mono">
                {t.portrait.step3}
              </h3>
              <div className="grid grid-cols-3 gap-3">
                {styles.map((style) => {
                  const isActive = selectedStyle === style.key;
                  return (
                    <button
                      key={style.key}
                      onClick={() => setSelectedStyle(style.key)}
                      className={`p-4 rounded-sm border text-center transition-all duration-200 cursor-pointer ${
                        isActive
                          ? 'bg-cyan-500/10 border-cyan-500/50 shadow-lg shadow-cyan-500/10'
                          : 'bg-[#090b11] border-[#1f2533] hover:border-slate-500/50 hover:bg-[#1a202c]'
                      }`}
                    >
                      <div className="text-2xl mb-1">{style.label.slice(0, 2)}</div>
                      <div className={`text-xs font-semibold font-mono ${isActive ? 'text-cyan-400' : 'text-slate-300'}`}>
                        {style.label.slice(2)}
                      </div>
                      <div className={`w-2 h-2 rounded-full mx-auto mt-2 ${isActive ? 'bg-cyan-400' : 'bg-slate-700'}`} />
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* ── 高级选项 ── */}
          <details className="bg-[#121622] border border-[#1f2533] rounded-sm">
            <summary className="p-3 text-xs font-semibold text-slate-400 uppercase tracking-wider cursor-pointer font-mono hover:text-slate-300">
              ⚙️ {t.portrait.advancedOptions}
            </summary>
            <div className="grid grid-cols-4 gap-4 p-4 pt-0">
              <div>
                <label className="text-[10px] text-slate-500 font-mono uppercase block mb-1">
                  {t.portrait.seed}
                </label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value) || -1)}
                  className="w-full bg-[#090b11] border border-[#1f2533] rounded-sm px-3 py-2 text-sm text-slate-200
                             focus:outline-none focus:border-cyan-500/50 font-mono"
                />
              </div>
              <div>
                <label className="text-[10px] text-slate-500 font-mono uppercase block mb-1">
                  {t.portrait.numSteps} ({numSteps})
                </label>
                <input
                  type="range"
                  min={20}
                  max={100}
                  step={5}
                  value={numSteps}
                  onChange={(e) => setNumSteps(parseInt(e.target.value))}
                  className="w-full accent-cyan-500"
                />
              </div>
              <div>
                <label className="text-[10px] text-slate-500 font-mono uppercase block mb-1">
                  {t.portrait.guidanceScale} ({guidance.toFixed(1)})
                </label>
                <input
                  type="range"
                  min={1.0}
                  max={15.0}
                  step={0.5}
                  value={guidance}
                  onChange={(e) => setGuidance(parseFloat(e.target.value))}
                  className="w-full accent-cyan-500"
                />
              </div>
              <div>
                <label className="text-[10px] text-slate-500 font-mono uppercase block mb-1" title={t.portrait.mergeStepHelp}>
                  {t.portrait.mergeStep} ({mergeStep})
                </label>
                <input
                  type="range"
                  min={0}
                  max={30}
                  step={1}
                  value={mergeStep}
                  onChange={(e) => setMergeStep(parseInt(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <p className="text-[9px] text-slate-600 mt-0.5">{t.portrait.mergeStepHelp}</p>
              </div>
            </div>
          </details>

          {/* ── 步骤 4: 生成按钮 ── */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleGenerate}
              disabled={generating || !serverImagePath}
              className="flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500
                         text-white font-bold text-sm rounded-sm transition-all duration-200 shadow-lg shadow-cyan-500/20
                         active:scale-[0.98] disabled:cursor-not-allowed cursor-pointer font-mono uppercase tracking-wider"
            >
              {generating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {t.portrait.generating}
                </>
              ) : (
                <>
                  <Wand2 className="w-4 h-4" />
                  {t.portrait.generateBtn}
                </>
              )}
            </button>

            {/* 模型状态提示 */}
            {modelLoaded ? (
              <span className="flex items-center gap-1.5 text-[11px] text-emerald-400 font-mono">
                <CheckCircle2 className="w-3.5 h-3.5" />
                {t.portrait.modelLoaded}
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-[11px] text-amber-400 font-mono">
                <Info className="w-3.5 h-3.5" />
                {t.portrait.modelNotLoaded}
              </span>
            )}
          </div>

          {/* ── 生成中进度 ── */}
          {generating && (
            <div className="bg-[#121622] border border-[#1f2533] rounded-sm p-6 flex flex-col items-center gap-3">
              <div className="w-10 h-10 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-cyan-400 font-mono">{t.portrait.generating}</p>
              <p className="text-[10px] text-slate-500">Loading PhotoMaker + SDXL pipeline...</p>
            </div>
          )}

          {/* ── 错误提示 ── */}
          {error && (
            <div className="bg-red-950/30 border border-red-500/30 rounded-sm p-4 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <p className="text-sm text-red-300 font-semibold">Generation Failed</p>
                <p className="text-xs text-red-400/80 mt-1 font-mono">{error}</p>
              </div>
            </div>
          )}

          {/* ── 步骤 5: 结果展示 ── */}
          {result && (
            <div className="bg-[#121622] border border-[#1f2533] rounded-sm p-6">
              <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-4 font-mono flex items-center gap-2">
                <ImageIcon className="w-4 h-4" />
                {t.portrait.resultTitle}
              </h3>

              <div className="grid grid-cols-2 gap-6">
                {/* 生成图像 */}
                <div>
                  <img
                    src={`data:image/png;base64,${result.result_image_base64}`}
                    alt={`${selectedPerson} - ${result.style_label}`}
                    className="w-full rounded-sm border border-[#1f2533]"
                  />
                  <a
                    href={`data:image/png;base64,${result.result_image_base64}`}
                    download={`${selectedPerson}_${result.style}_seed${result.seed}.png`}
                    className="mt-3 flex items-center justify-center gap-2 w-full py-2 bg-slate-800 hover:bg-slate-700
                               text-slate-300 text-xs font-mono rounded-sm transition-colors cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    {t.portrait.downloadBtn}
                  </a>
                </div>

                {/* 元信息 */}
                <div className="space-y-3 text-xs font-mono">
                  <h4 className="text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
                    {t.portrait.details}
                  </h4>
                  <div className="grid grid-cols-2 gap-2">
                    <DetailRow label={t.portrait.styleKey} value={result.style_label} />
                    <DetailRow label={t.portrait.seedNum} value={String(result.seed)} />
                    <DetailRow label={t.portrait.elapsed} value={`${result.generation_time_seconds}s`} />
                    <DetailRow
                      label={t.portrait.dimensions}
                      value={`${result.width} × ${result.height}`}
                    />
                    <DetailRow label={t.portrait.steps} value={String(numSteps)} />
                    <DetailRow label={t.portrait.guidance} value={guidance.toFixed(1)} />
                  </div>

                  <div>
                    <label className="text-[10px] text-slate-500 uppercase block mb-1">
                      {t.portrait.savePath}
                    </label>
                    <code className="block text-[10px] text-cyan-400 bg-[#090b11] p-2 rounded-sm break-all border border-[#1f2533]">
                      {result.output_path}
                    </code>
                  </div>

                  <details>
                    <summary className="text-[10px] text-slate-500 uppercase cursor-pointer hover:text-slate-400">
                      {t.portrait.promptUsed}
                    </summary>
                    <div className="mt-2 space-y-2">
                      <div>
                        <label className="text-[9px] text-slate-600 uppercase">{t.portrait.positivePrompt}</label>
                        <p className="text-[10px] text-slate-400 bg-[#090b11] p-2 rounded-sm border border-[#1f2533] mt-0.5 break-all">
                          {result.prompt_used}
                        </p>
                      </div>
                      <div>
                        <label className="text-[9px] text-slate-600 uppercase">{t.portrait.negativePrompt}</label>
                        <p className="text-[10px] text-slate-500 bg-[#090b11] p-2 rounded-sm border border-[#1f2533] mt-0.5 break-all">
                          nsfw, worst quality, low quality, blurry, distorted face, bad anatomy, cartoon, painting
                        </p>
                      </div>
                    </div>
                  </details>
                </div>
              </div>
            </div>
          )}

          {/* ── 卸载模型 ── */}
          <div className="border-t border-[#1f2533] pt-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="w-3.5 h-3.5 text-slate-500" />
              <span className="text-[10px] text-slate-500 font-mono">
                PhotoMaker + SDXL · 1024×1024
              </span>
            </div>
            <div className="flex items-center gap-3">
              {unloadedMsg && (
                <span className="text-[10px] text-emerald-400 font-mono">{unloadedMsg}</span>
              )}
              <button
                onClick={handleUnload}
                disabled={unloading || !modelLoaded}
                className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] text-slate-500 hover:text-red-400
                           bg-[#121622] border border-[#1f2533] hover:border-red-500/30 rounded-sm
                           transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer font-mono uppercase"
              >
                {unloading ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Trash2 className="w-3 h-3" />
                )}
                {t.portrait.unloadBtn}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** 详情行小组件 */
function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-[9px] text-slate-600 uppercase">{label}</span>
      <p className="text-xs text-slate-300 mt-0.5">{value}</p>
    </div>
  );
}
