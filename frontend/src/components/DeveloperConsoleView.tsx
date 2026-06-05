/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Cpu,
  RefreshCw,
  Plus,
  Trash2,
  Terminal,
  Play,
  Square,
  Upload,
  AlertTriangle,
  Download,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar,
} from 'recharts';
import { ModelItem, LogLine, EvalJobStatus, BenchmarkSummaryResponse, ModelRuntimeInfoResponse } from '../types';
import { Language, LOCALES } from '../locales';
import { listModels, uploadModelWithProgress, activateModel, deleteModel, getRuntime } from '../api/models';
import ProgressBar from './ProgressBar';
import {
  getBenchmark,
  submitLfwEval,
  submitIjbEval,
  submitThresholdSweep,
  getEvalStatus,
} from '../api/developer';
import { ApiError } from '../api/client';

interface DeveloperConsoleViewProps {
  addLog: (level: LogLine['level'], msg: string) => void;
  language?: Language;
}

type ConsoleTab = 'lfw' | 'ijb' | 'sweep' | 'benchmark';

export default function DeveloperConsoleView({
  addLog,
  language = 'en'
}: DeveloperConsoleViewProps) {
  // ── Model state ──
  const [modelsList, setModelsList] = useState<ModelItem[]>([]);
  const [runtime, setRuntime] = useState<ModelRuntimeInfoResponse | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelError, setModelError] = useState<string | null>(null);

  // ── Tab state ──
  const [activeConsoleTab, setActiveConsoleTab] = useState<ConsoleTab>('lfw');

  // ── Eval state ──
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalJobId, setEvalJobId] = useState<string | null>(null);
  const [evalProgress, setEvalProgress] = useState(0);
  const [evalLogs, setEvalLogs] = useState<string[]>([]);
  const [evalResult, setEvalResult] = useState<Record<string, unknown> | null>(null);
  const [evalResultType, setEvalResultType] = useState<string>('');
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Benchmark state ──
  const [benchmark, setBenchmark] = useState<BenchmarkSummaryResponse | null>(null);

  // ── Modal state ──
  const [showDeployModelModal, setShowDeployModelModal] = useState(false);
  const [newModelName, setNewModelName] = useState('');
  const [newModelBackbone, setNewModelBackbone] = useState('iresnet100');
  const [newModelSize, setNewModelSize] = useState('512');
  const [newModelFile, setNewModelFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Eval form state ──
  const [lfwForm, setLfwForm] = useState({
    weights_path: 'weights/model_best.pt',
    backbone: 'iresnet100',
    data_root: 'data',
    datasets: ['lfw'],
    batch_size: 512,
  });
  const [ijbForm, setIjbForm] = useState({
    weights_path: 'weights/model_best.pt',
    backbone: 'iresnet100',
    image_path: 'data/IJBB',
    target: 'IJBB',
    batch_size: 512,
    use_norm_score: true,
    use_detector_score: true,
    use_flip_test: true,
    result_dir: '',
  });
  const [sweepForm, setSweepForm] = useState({
    weights_path: 'weights/model_best.pt',
    backbone: 'iresnet100',
    image_dir: 'data',
    db_path: 'benchmark/YTF_100p.db',
    thresholds: '0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70',
    device: 'auto',
  });

  const t = LOCALES[language];

  // ── Fetch models on mount ──
  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    setModelError(null);
    try {
      const [mlist, rt] = await Promise.all([listModels(), getRuntime()]);
      setModelsList(mlist.models);
      setRuntime(rt);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Failed to load models';
      setModelError(msg);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  useEffect(() => { fetchModels(); }, [fetchModels]);

  // ── Fetch benchmark when tab switched ──
  useEffect(() => {
    if (activeConsoleTab === 'benchmark' && !benchmark) {
      getBenchmark()
        .then(setBenchmark)
        .catch(() => {});
    }
  }, [activeConsoleTab, benchmark]);

  // ── Cleanup polling ──
  useEffect(() => {
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, []);

  // ── Eval helpers ──

  const appendLog = (msg: string) => {
    setEvalLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const startPolling = (jobId: string) => {
    setEvalJobId(jobId);
    setIsEvaluating(true);
    setEvalProgress(0);
    setEvalResult(null);
    setEvalResultType('');

    pollingRef.current = setInterval(async () => {
      try {
        const status: EvalJobStatus = await getEvalStatus(jobId);
        setEvalProgress(status.progress);
        if (status.progress_msg) appendLog(status.progress_msg);

        if (status.status === 'done') {
          clearInterval(pollingRef.current!);
          pollingRef.current = null;
          setIsEvaluating(false);
          setEvalResultType(status.job_type);
          setEvalResult(status.result);
          appendLog(`JOB COMPLETE: ${status.job_type} finished in ${status.elapsed_seconds.toFixed(1)}s`);
          addLog('RESULT', `EVAL: ${status.job_type} completed (${status.elapsed_seconds.toFixed(1)}s)`);
        } else if (status.status === 'error') {
          clearInterval(pollingRef.current!);
          pollingRef.current = null;
          setIsEvaluating(false);
          appendLog(`JOB ERROR: ${status.error || 'Unknown error'}`);
          addLog('ERROR', `EVAL: ${status.job_type} failed — ${status.error || 'unknown'}`);
        }
      } catch {
        // Silently skip poll errors
      }
    }, 2000);
  };

  const cancelEval = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setIsEvaluating(false);
    appendLog('Evaluation cancelled by user.');
    addLog('WARNING', 'EVAL: Job cancelled by user.');
  };

  // ── Model actions ──

  const handleActivateModel = async (modelId: number) => {
    try {
      await activateModel(modelId);
      setModelsList(prev => prev.map(m => ({ ...m, is_active: m.id === modelId })));
      addLog('INFO', `MODEL: Activated model #${modelId}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Activation failed';
      addLog('ERROR', `MODEL: ${msg}`);
    }
  };

  const handleDeleteModel = async (modelId: number) => {
    try {
      await deleteModel(modelId);
      setModelsList(prev => prev.filter(m => m.id !== modelId));
      addLog('INFO', `MODEL: Deleted model #${modelId}`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Delete failed';
      addLog('ERROR', `MODEL: ${msg}`);
    }
  };

  const handleUploadModel = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newModelName.trim() || !newModelFile) return;

    setUploading(true);
    setUploadProgress(0);
    try {
      const uploaded = await uploadModelWithProgress(
        newModelFile, newModelName, newModelBackbone, Number(newModelSize),
        setUploadProgress,
      );
      setModelsList(prev => [...prev, uploaded]);
      setShowDeployModelModal(false);
      setNewModelName('');
      setNewModelFile(null);
      addLog('INFO', `MODEL: Uploaded ${uploaded.name} (${uploaded.backbone}, ${uploaded.embedding_size}-dim)`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Upload failed';
      addLog('ERROR', `MODEL_UPLOAD: ${msg}`);
    } finally {
      setUploading(false);
    }
  };

  // ── Dataset checkbox helper ──
  const toggleDataset = (ds: string) => {
    setLfwForm(prev => ({
      ...prev,
      datasets: prev.datasets.includes(ds)
        ? prev.datasets.filter(d => d !== ds)
        : [...prev.datasets, ds],
    }));
  };

  const availableDatasets = ['lfw', 'calfw', 'cplfw', 'agedb_30', 'cfp_fp', 'vgg2_fp'];

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-[calc(100vh-4rem)] bg-[#090b11] text-slate-300 font-sans select-none">
      {/* ── TAB BAR ── */}
      <div className="flex flex-wrap items-center justify-between border-b border-[#1f2533] pb-1 gap-4">
        <div className="flex gap-2 text-xs font-mono font-bold uppercase leading-none">
          {(['lfw', 'ijb', 'sweep', 'benchmark'] as ConsoleTab[]).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveConsoleTab(tab)}
              className={`px-4 py-2 border-b-2 transition-colors cursor-pointer ${
                activeConsoleTab === tab
                  ? 'text-cyan-400 border-cyan-500 font-extrabold'
                  : 'text-slate-500 border-transparent hover:text-slate-300'
              }`}
            >
              {tab === 'lfw' ? 'LFW Eval' : tab === 'ijb' ? 'IJB Eval' : tab === 'sweep' ? 'Threshold Sweep' : 'Benchmark'}
            </button>
          ))}
        </div>

        {isEvaluating ? (
          <button
            onClick={cancelEval}
            className="px-4 py-2 rounded-sm font-mono text-xs uppercase tracking-wider flex items-center gap-1.5 bg-rose-950/40 hover:bg-rose-850/50 border border-rose-500/30 text-rose-200 cursor-pointer"
          >
            <Square className="w-3.5 h-3.5" />
            {language === 'zh' ? `取消 (${Math.round(evalProgress)}%)` : `Cancel (${Math.round(evalProgress)}%)`}
          </button>
        ) : (
          <span className="text-[9px] text-slate-600 font-mono uppercase">
            {runtime ? `${runtime.model_name} @ ${runtime.device}` : ''}
          </span>
        )}
      </div>

      {/* ── EVAL FORM PANELS ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Eval form */}
        <div className="bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-4 font-mono text-xs">
          <h4 className="text-[10px] font-bold tracking-[0.2em] text-slate-400 uppercase leading-none">
            {language === 'zh' ? '评估参数' : 'Eval Parameters'}
          </h4>

          {activeConsoleTab === 'lfw' && (
            <>
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Weights Path</label>
                <input type="text" value={lfwForm.weights_path} onChange={e => setLfwForm(p => ({ ...p, weights_path: e.target.value }))}
                  className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Backbone</label>
                  <select value={lfwForm.backbone} onChange={e => setLfwForm(p => ({ ...p, backbone: e.target.value }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500">
                    <option value="iresnet50">iResNet-50</option>
                    <option value="iresnet100">iResNet-100</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Batch Size</label>
                  <input type="number" value={lfwForm.batch_size} onChange={e => setLfwForm(p => ({ ...p, batch_size: Number(e.target.value) }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Data Root</label>
                <input type="text" value={lfwForm.data_root} onChange={e => setLfwForm(p => ({ ...p, data_root: e.target.value }))}
                  className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Datasets</label>
                <div className="flex flex-wrap gap-1.5">
                  {availableDatasets.map(ds => (
                    <button key={ds} onClick={() => toggleDataset(ds)}
                      className={`px-2 py-1 text-[9px] rounded-sm border font-bold uppercase transition-colors cursor-pointer ${
                        lfwForm.datasets.includes(ds)
                          ? 'bg-cyan-950/30 border-cyan-500 text-cyan-400'
                          : 'border-[#1f2533] bg-slate-900/40 text-slate-500 hover:text-slate-300'
                      }`}>{ds}</button>
                  ))}
                </div>
              </div>
              <button
                onClick={() => {
                  appendLog(`Submitting LFW eval: ${lfwForm.backbone}, datasets=[${lfwForm.datasets.join(',')}]`);
                  submitLfwEval(lfwForm).then(res => {
                    appendLog(`Job submitted: ${res.job_id}`);
                    startPolling(res.job_id);
                  }).catch(err => {
                    appendLog(`ERROR: ${err instanceof ApiError ? err.detail : 'Submission failed'}`);
                  });
                }}
                disabled={isEvaluating}
                className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-bold uppercase tracking-wider rounded-sm cursor-pointer disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                {language === 'zh' ? '提交LFW评估' : 'Submit LFW Eval'}
              </button>
            </>
          )}

          {activeConsoleTab === 'ijb' && (
            <>
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Weights Path</label>
                <input type="text" value={ijbForm.weights_path} onChange={e => setIjbForm(p => ({ ...p, weights_path: e.target.value }))}
                  className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Backbone</label>
                  <select value={ijbForm.backbone} onChange={e => setIjbForm(p => ({ ...p, backbone: e.target.value }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500">
                    <option value="iresnet50">iResNet-50</option>
                    <option value="iresnet100">iResNet-100</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Target</label>
                  <select value={ijbForm.target} onChange={e => setIjbForm(p => ({ ...p, target: e.target.value }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500">
                    <option value="IJBB">IJB-B</option>
                    <option value="IJBC">IJB-C</option>
                  </select>
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Image Path</label>
                <input type="text" value={ijbForm.image_path} onChange={e => setIjbForm(p => ({ ...p, image_path: e.target.value }))}
                  className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Options</label>
                <div className="flex flex-wrap gap-1.5">
                  {(['use_norm_score', 'use_detector_score', 'use_flip_test'] as const).map(opt => (
                    <button key={opt} onClick={() => setIjbForm(p => ({ ...p, [opt]: !p[opt] }))}
                      className={`px-2 py-1 text-[9px] rounded-sm border font-bold uppercase transition-colors cursor-pointer ${
                        ijbForm[opt] ? 'bg-cyan-950/30 border-cyan-500 text-cyan-400' : 'border-[#1f2533] bg-slate-900/40 text-slate-500'
                      }`}>{opt.replace(/_/g, ' ')}</button>
                  ))}
                </div>
              </div>
              <button
                onClick={() => {
                  appendLog(`Submitting IJB eval: ${ijbForm.target}, backbone=${ijbForm.backbone}`);
                  submitIjbEval(ijbForm).then(res => {
                    appendLog(`Job submitted: ${res.job_id}`);
                    startPolling(res.job_id);
                  }).catch(err => {
                    appendLog(`ERROR: ${err instanceof ApiError ? err.detail : 'Submission failed'}`);
                  });
                }}
                disabled={isEvaluating}
                className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-bold uppercase tracking-wider rounded-sm cursor-pointer disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                {language === 'zh' ? '提交IJB评估' : 'Submit IJB Eval'}
              </button>
            </>
          )}

          {activeConsoleTab === 'sweep' && (
            <>
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Weights Path</label>
                <input type="text" value={sweepForm.weights_path} onChange={e => setSweepForm(p => ({ ...p, weights_path: e.target.value }))}
                  className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Image Dir</label>
                  <input type="text" value={sweepForm.image_dir} onChange={e => setSweepForm(p => ({ ...p, image_dir: e.target.value }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">DB Path</label>
                  <input type="text" value={sweepForm.db_path} onChange={e => setSweepForm(p => ({ ...p, db_path: e.target.value }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[10px] text-slate-500 uppercase font-bold">Thresholds (CSV)</label>
                <input type="text" value={sweepForm.thresholds} onChange={e => setSweepForm(p => ({ ...p, thresholds: e.target.value }))}
                  className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Backbone</label>
                  <select value={sweepForm.backbone} onChange={e => setSweepForm(p => ({ ...p, backbone: e.target.value }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500">
                    <option value="iresnet50">iResNet-50</option>
                    <option value="iresnet100">iResNet-100</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-[10px] text-slate-500 uppercase font-bold">Device</label>
                  <select value={sweepForm.device} onChange={e => setSweepForm(p => ({ ...p, device: e.target.value }))}
                    className="w-full bg-slate-900 border border-[#1f2533] rounded-sm px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500">
                    <option value="auto">Auto</option>
                    <option value="cuda:0">CUDA:0</option>
                    <option value="cpu">CPU</option>
                  </select>
                </div>
              </div>
              <button
                onClick={() => {
                  appendLog(`Submitting threshold sweep: ${sweepForm.backbone}`);
                  submitThresholdSweep(sweepForm).then(res => {
                    appendLog(`Job submitted: ${res.job_id}`);
                    startPolling(res.job_id);
                  }).catch(err => {
                    appendLog(`ERROR: ${err instanceof ApiError ? err.detail : 'Submission failed'}`);
                  });
                }}
                disabled={isEvaluating}
                className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-500 text-slate-950 font-bold uppercase tracking-wider rounded-sm cursor-pointer disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                {language === 'zh' ? '提交阈值扫频' : 'Submit Sweep'}
              </button>
            </>
          )}

          {activeConsoleTab === 'benchmark' && (
            <div className="space-y-3">
              <button
                onClick={() => {
                  setBenchmark(null);
                  getBenchmark().then(setBenchmark).catch(err => {
                    addLog('ERROR', `BENCHMARK: ${err instanceof ApiError ? err.detail : 'Failed'}`);
                  });
                }}
                className="w-full py-3 bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold uppercase tracking-wider rounded-sm cursor-pointer flex items-center justify-center gap-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                {language === 'zh' ? '刷新基准数据' : 'Refresh Benchmark'}
              </button>

              {benchmark && (
                <div className="space-y-2 pt-2 border-t border-[#1e2330]">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-500">{language === 'zh' ? '数据库' : 'Database'}</span>
                    <span className="text-white font-bold truncate ml-2">{benchmark.database_path}</span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-500">{language === 'zh' ? '人员数' : 'Persons'}</span>
                    <span className="text-cyan-400 font-bold">{benchmark.person_count}</span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-500">{language === 'zh' ? '特征数' : 'Embeddings'}</span>
                    <span className="text-cyan-400 font-bold">{benchmark.embedding_count}</span>
                  </div>
                  {benchmark.top_persons.length > 0 && (
                    <div className="pt-2">
                      <p className="text-[10px] text-slate-500 uppercase font-bold mb-1.5">{language === 'zh' ? 'Top 人员' : 'Top Persons'}</p>
                      {benchmark.top_persons.map((p, i) => (
                        <div key={i} className="flex justify-between text-[10px] py-0.5">
                          <span className="text-slate-300">{p.name}</span>
                          <span className="text-slate-500">{p.embedding_count}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Progress bar */}
          {isEvaluating && (
            <div className="space-y-1.5 pt-2 border-t border-[#1e2330]">
              <div className="flex justify-between text-[10px]">
                <span className="text-slate-500 uppercase">{language === 'zh' ? '进度' : 'Progress'}</span>
                <span className="text-cyan-400 font-bold">{Math.round(evalProgress)}%</span>
              </div>
              <div className="w-full h-1.5 bg-slate-800 rounded-sm overflow-hidden">
                <div className="h-full bg-cyan-500 transition-all duration-300" style={{ width: `${evalProgress}%` }} />
              </div>
            </div>
          )}
        </div>

        {/* Results / ROC area */}
        <div className="lg:col-span-2 bg-[#121622] rounded-sm p-5 border border-[#1e2330] space-y-4">
          <h4 className="text-xs font-bold font-mono tracking-widest text-slate-400 uppercase leading-none">
            {language === 'zh' ? '评估结果' : 'Evaluation Results'}
          </h4>

          {evalResult ? (
            <EvalResultCharts result={evalResult} jobType={evalResultType} language={language} />
          ) : (
            <div className="flex-1 flex items-center justify-center min-h-[200px] bg-[#0c0e14] border border-slate-900 rounded-sm">
              <div className="text-center opacity-30">
                <Terminal className="w-10 h-10 text-slate-500 mx-auto mb-2" />
                <p className="text-[10px] font-mono uppercase tracking-widest text-slate-400">
                  {isEvaluating
                    ? (language === 'zh' ? '评估运行中...' : 'Evaluation running...')
                    : (language === 'zh' ? '提交评估以查看结果' : 'Submit an eval to see results')}
                </p>
              </div>
            </div>
          )}

          {/* Quick stats */}
          {runtime && (
            <div className="grid grid-cols-4 gap-3 pt-2 border-t border-[#1e2330]">
              <div className="text-center">
                <p className="text-[8px] text-slate-500 font-mono uppercase tracking-widest">{language === 'zh' ? '模型' : 'Model'}</p>
                <p className="text-[10px] font-bold font-mono text-white mt-1 truncate">{runtime.model_name}</p>
              </div>
              <div className="text-center">
                <p className="text-[8px] text-slate-500 font-mono uppercase tracking-widest">{language === 'zh' ? '设备' : 'Device'}</p>
                <p className="text-[10px] font-bold font-mono text-cyan-400 mt-1">{runtime.device}</p>
              </div>
              <div className="text-center">
                <p className="text-[8px] text-slate-500 font-mono uppercase tracking-widest">{language === 'zh' ? '图像尺寸' : 'Img Size'}</p>
                <p className="text-[10px] font-bold font-mono text-white mt-1">{runtime.img_size}</p>
              </div>
              <div className="text-center">
                <p className="text-[8px] text-slate-500 font-mono uppercase tracking-widest">{language === 'zh' ? '底库大小' : 'Gallery'}</p>
                <p className="text-[10px] font-bold font-mono text-white mt-1">{runtime.gallery_size}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── REGISTERED MODELS TABLE ── */}
      <div className="bg-[#121622] border border-[#1e2330] rounded-sm p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold font-mono tracking-widest text-[#94a3b8] uppercase leading-none">
            {language === 'zh' ? '已注册模型' : 'Registered Models'}
          </h4>
          <button
            onClick={() => setShowDeployModelModal(true)}
            className="px-3.5 py-1.5 bg-cyan-500 hover:bg-cyan-600 text-slate-950 text-[10px] font-mono tracking-wider font-extrabold uppercase rounded-sm flex items-center gap-1 cursor-pointer transition-transform active:scale-95"
          >
            <Plus className="w-3.5 h-3.5 stroke-[3]" />
            {language === 'zh' ? '上传模型' : 'Upload Model'}
          </button>
        </div>

        {modelError && (
          <div className="bg-red-950/20 border border-red-500/30 rounded-sm p-3 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <p className="text-xs text-red-300 font-mono">{modelError}</p>
          </div>
        )}

        <div className="overflow-x-auto w-full border border-[#1d2330] rounded-sm bg-[#0a0d14]">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-[#11141e] border-b border-[#1e2330] text-[10px] font-mono uppercase text-[#64748b] font-bold">
              <tr>
                <th className="py-2.5 px-4">{language === 'zh' ? '状态' : 'Status'}</th>
                <th className="py-2.5 px-4">{language === 'zh' ? '名称 / 骨干' : 'Name / Backbone'}</th>
                <th className="py-2.5 px-4">{language === 'zh' ? '框架' : 'Framework'}</th>
                <th className="py-2.5 px-4">{language === 'zh' ? '特征维度' : 'Embed'}</th>
                <th className="py-2.5 px-4">{language === 'zh' ? '创建时间' : 'Created'}</th>
                <th className="py-2.5 px-4 text-center">{language === 'zh' ? '操作' : 'Actions'}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#181d28]/60 text-slate-400 font-mono">
              {modelsLoading ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center">
                    <RefreshCw className="w-5 h-5 text-slate-500 animate-spin mx-auto" />
                  </td>
                </tr>
              ) : modelsList.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500 uppercase text-[10px]">
                    {language === 'zh' ? '暂无模型' : 'No models registered'}
                  </td>
                </tr>
              ) : (
                modelsList.map(model => (
                  <tr key={model.id} className="hover:bg-[#121622]/40 transition-colors">
                    <td className="py-3.5 px-4">
                      {model.is_active ? (
                        <span className="flex items-center gap-1.5 text-cyan-400 font-bold uppercase text-[9px] bg-cyan-950/20 border border-cyan-900/40 px-2 py-0.5 rounded-sm">
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping inline-block" />
                          {language === 'zh' ? '活跃' : 'Active'}
                        </span>
                      ) : (
                        <span className="flex items-center gap-1.5 text-slate-500 uppercase text-[9px]">
                          <span className="w-1.5 h-1.5 rounded-full bg-slate-600 inline-block" />
                          {language === 'zh' ? '待命' : 'Standby'}
                        </span>
                      )}
                    </td>
                    <td className="py-3.5 px-4">
                      <p className="text-white text-xs font-semibold">{model.name}</p>
                      <p className="text-[10px] text-slate-500 mt-0.5">{model.backbone}</p>
                    </td>
                    <td className="py-3.5 px-4">{model.framework}</td>
                    <td className="py-3.5 px-4 text-[#38bdf8]">{model.embedding_size}-dim</td>
                    <td className="py-3.5 px-4 text-slate-500 text-[10px]">{model.created_at}</td>
                    <td className="py-3.5 px-4 text-center">
                      <div className="flex items-center justify-center gap-1">
                        {!model.is_active && (
                          <button
                            onClick={() => handleActivateModel(model.id)}
                            className="px-3 py-1 bg-slate-800 hover:bg-slate-700 hover:text-white border border-slate-700 text-slate-400 text-[9px] uppercase font-bold rounded-sm cursor-pointer transition-colors"
                          >
                            {language === 'zh' ? '激活' : 'Activate'}
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteModel(model.id)}
                          className="p-1.5 text-slate-600 hover:text-rose-400 hover:bg-rose-950/20 rounded-sm cursor-pointer transition-colors"
                          title={language === 'zh' ? '删除' : 'Delete'}
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
      </div>

      {/* ── STDOUT TERMINAL ── */}
      <div className="bg-[#121622] border border-[#1e2330] rounded-sm p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-[#212733] pb-2">
          <h4 className="text-[9px] font-bold tracking-[0.2em] text-[#64748b] uppercase font-mono leading-none flex items-center gap-1">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            STDOUT — {language === 'zh' ? '系统监视器' : 'System Monitor'}
          </h4>
        </div>
        <div className="bg-[#0b0e14] border border-[#1e232c] rounded-sm p-4 text-[10px] font-mono text-[#06b6d4] space-y-1 h-[150px] overflow-y-auto">
          {evalLogs.length === 0 ? (
            <span className="text-slate-600">{language === 'zh' ? '等待评估任务...' : 'Awaiting eval job...'}</span>
          ) : (
            evalLogs.map((log, index) => (
              <div key={index} className="leading-relaxed">{log}</div>
            ))
          )}
        </div>
      </div>

      {/* ── MODEL UPLOAD MODAL ── */}
      {showDeployModelModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 font-mono text-xs">
          <div className="w-full max-w-md bg-[#121622] border border-[#262f3f] rounded-sm p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-[#2a354c] pb-3">
              <h4 className="text-[#38bdf8] font-bold uppercase tracking-wider">
                {language === 'zh' ? '上传模型权重' : 'Upload Model Weights'}
              </h4>
              <button
                onClick={() => setShowDeployModelModal(false)}
                className="text-slate-500 hover:text-white font-extrabold cursor-pointer text-sm"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleUploadModel} className="space-y-4">
              <div className="space-y-2">
                <label className="text-slate-400 uppercase font-bold">{language === 'zh' ? '模型名称' : 'Model Name'}</label>
                <input
                  type="text" required value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  placeholder="e.g. ResNet_Plus_v9"
                  className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-slate-400 uppercase font-bold">{language === 'zh' ? '骨干网络' : 'Backbone'}</label>
                  <select
                    value={newModelBackbone}
                    onChange={(e) => setNewModelBackbone(e.target.value)}
                    className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none"
                  >
                    <option value="iresnet50">iResNet-50</option>
                    <option value="iresnet100">iResNet-100</option>
                    <option value="ghostnet_v5">GhostNet-v5</option>
                    <option value="mobilenet_v3">MobileNet-v3</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-slate-400 uppercase font-bold">{language === 'zh' ? '特征维度' : 'Embed Dim'}</label>
                  <select
                    value={newModelSize}
                    onChange={(e) => setNewModelSize(e.target.value)}
                    className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none"
                  >
                    <option value="128">128-dim</option>
                    <option value="256">256-dim</option>
                    <option value="512">512-dim</option>
                    <option value="1024">1024-dim</option>
                  </select>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-slate-400 uppercase font-bold">{language === 'zh' ? '权重文件 (.pt/.onnx)' : 'Weights File (.pt/.onnx)'}</label>
                <input type="file" ref={fileInputRef} accept=".pt,.onnx,.pth"
                  onChange={(e) => { if (e.target.files?.[0]) setNewModelFile(e.target.files[0]); }}
                  className="w-full text-slate-400 text-[11px] file:mr-3 file:py-2 file:px-4 file:rounded-sm file:border-0 file:text-xs file:font-bold file:bg-cyan-500 file:text-slate-950 hover:file:bg-cyan-600 file:cursor-pointer"
                />
                {newModelFile && (
                  <p className="text-[10px] text-cyan-400">{newModelFile.name} ({(newModelFile.size / 1024 / 1024).toFixed(2)} MB)</p>
                )}
              </div>

              <div className="pt-4 flex gap-3">
                <button
                  type="submit" disabled={uploading || !newModelFile}
                  className="flex-1 py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-500 font-bold uppercase text-slate-950 tracking-wider rounded-sm cursor-pointer disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-1.5"
                >
                  {uploading ? (
                    <span className="flex-1">
                      <ProgressBar value={uploadProgress} size="sm" />
                    </span>
                  ) : (
                    <><Upload className="w-3.5 h-3.5" />{language === 'zh' ? '确认上传' : 'Upload'}</>
                  )}
                </button>
                <button
                  type="button" onClick={() => setShowDeployModelModal(false)}
                  className="w-1/3 py-3 bg-[#1e2533] hover:bg-[#293245] font-bold uppercase text-slate-300 rounded-sm cursor-pointer transition-colors"
                >
                  {language === 'zh' ? '取消' : 'Cancel'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Chart color palette ──
const CHART_COLORS = {
  tpr: '#4CAF50',
  fpr: '#f75f5f',
  frr: '#a64ff7',
  rank1: '#4f8ef7',
  tar: '#f7a64f',
  f1: '#5fcf7a',
  grid: '#1e2330',
  axis: '#64748b',
  bg: '#0c0e14',
};

function asNumArray(v: unknown): number[] {
  if (Array.isArray(v)) return v.map(Number);
  return [];
}

function triggerDownload(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function buildVerificationCSV(result: Record<string, unknown>): string {
  const datasets = (result.datasets ?? {}) as Record<string, { acc?: number; threshold?: number; num_pairs?: number }>;
  const tprValues = (result.tpr_values ?? {}) as Record<string, number>;
  const lines: string[] = [];
  lines.push(`mean_accuracy,${result.mean_accuracy}`);
  lines.push(`roc_auc,${result.roc_auc}`);
  lines.push('');
  lines.push('dataset,accuracy,threshold,pairs');
  for (const [name, info] of Object.entries(datasets)) {
    lines.push(`${name},${info.acc ?? ''},${info.threshold ?? ''},${info.num_pairs ?? ''}`);
  }
  lines.push('');
  lines.push('far_level,tar');
  for (const [farKey, tarVal] of Object.entries(tprValues)) {
    lines.push(`${farKey},${tarVal}`);
  }
  return lines.join('\n');
}

function buildThresholdSweepCSV(result: Record<string, unknown>): string {
  const summary = (result.summary ?? []) as Array<Record<string, unknown>>;
  const lines: string[] = [];
  lines.push('threshold,rank1_accuracy,far,frr,tar_at_far_1e3,f1_score,auc_score,num_samples');
  for (const row of summary) {
    lines.push([
      row.threshold, row.rank1_accuracy, row.far, row.frr,
      row.tar_at_far_1e3, row.f1_score, row.auc_score, row.num_samples,
    ].join(','));
  }
  return lines.join('\n');
}

// ── Eval result chart renderer ──
function EvalResultCharts({ result, jobType, language }: { result: Record<string, unknown>; jobType: string; language: Language }) {
  const isVerification = jobType === 'lfw_verification' || jobType === 'ijb_evaluation';
  const isSweep = jobType === 'threshold_sweep';

  const handleExportJSON = () => {
    triggerDownload(JSON.stringify(result, null, 2), `eval_${jobType}_${Date.now()}.json`, 'application/json');
  };

  const handleExportCSV = () => {
    let csv: string;
    if (isVerification) {
      csv = buildVerificationCSV(result);
    } else if (isSweep) {
      csv = buildThresholdSweepCSV(result);
    } else {
      csv = JSON.stringify(result);
    }
    triggerDownload(csv, `eval_${jobType}_${Date.now()}.csv`, 'text/csv');
  };

  const btnCls = "flex items-center gap-1 px-3 py-1.5 text-[10px] font-mono font-bold uppercase tracking-wider rounded-sm border border-[#1e2330] bg-[#0c0e14] text-slate-400 hover:text-white hover:border-slate-500 cursor-pointer transition-colors";

  return (
    <div className="space-y-3">
      {/* Export toolbar */}
      <div className="flex items-center gap-2">
        <button onClick={handleExportJSON} className={btnCls}>
          <Download className="w-3 h-3" /> JSON
        </button>
        <button onClick={handleExportCSV} className={btnCls}>
          <Download className="w-3 h-3" /> CSV
        </button>
        <span className="text-[9px] text-slate-600 font-mono ml-auto">
          {language === 'zh' ? '导出评估结果' : 'Export Results'}
        </span>
      </div>

      {isVerification && <VerificationCharts result={result} language={language} />}
      {isSweep && <ThresholdSweepCharts result={result} language={language} />}
      {!isVerification && !isSweep && (
        <div className="bg-[#0c0e14] border border-slate-900 rounded-sm p-4 max-h-[400px] overflow-y-auto">
          <pre className="text-[10px] font-mono text-cyan-400 whitespace-pre-wrap">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ── LFW / IJB verification charts ──
function VerificationCharts({ result, language }: { result: Record<string, unknown>; language: Language }) {
  const fpr = asNumArray(result.fpr);
  const tpr = asNumArray(result.tpr);
  const rocAuc = Number(result.roc_auc ?? 0);
  const tprValues = (result.tpr_values ?? {}) as Record<string, number>;
  const datasets = (result.datasets ?? {}) as Record<string, { acc?: number; threshold?: number }>;
  const meanAcc = Number(result.mean_accuracy ?? 0);
  const target = String(result.target ?? '');

  // Build ROC curve data
  const rocData = fpr.map((x, i) => ({ fpr: x, tpr: tpr[i] ?? 0 }));

  return (
    <div className="space-y-4">
      {/* Metric cards row */}
      <div className="grid grid-cols-4 gap-3">
        <MetricCard label={language === 'zh' ? '平均准确率' : 'Mean Accuracy'} value={`${(meanAcc * 100).toFixed(2)}%`} color="text-cyan-400" />
        <MetricCard label="ROC AUC" value={`${rocAuc.toFixed(4)}%`} color="text-emerald-400" />
        {target && <MetricCard label="Target" value={target} color="text-amber-400" />}
        <MetricCard label={language === 'zh' ? '数据集数' : 'Datasets'} value={String(Object.keys(datasets).length)} color="text-violet-400" />
      </div>

      {/* TAR@FAR cards */}
      {Object.keys(tprValues).length > 0 && (
        <div className="grid grid-cols-4 gap-3">
          {Object.entries(tprValues).map(([farKey, tarVal]) => (
            <div key={farKey} className="bg-[#0c0e14] border border-[#1e2330] rounded-sm p-3 text-center">
              <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">TAR @ FAR={farKey}</p>
              <p className="text-sm font-bold text-[#4f8ef7] font-mono mt-1">{Number(tarVal).toFixed(3)}%</p>
            </div>
          ))}
        </div>
      )}

      {/* ROC Curve */}
      {rocData.length > 1 && (
        <div className="bg-[#0c0e14] border border-[#1e2330] rounded-sm p-3">
          <p className="text-[10px] text-slate-500 font-mono uppercase font-bold mb-2">
            {language === 'zh' ? 'ROC 曲线' : 'ROC Curve'}
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rocData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
              <XAxis dataKey="fpr" tick={{ fontSize: 10, fill: CHART_COLORS.axis }}
                label={{ value: 'FPR', position: 'insideBottomRight', offset: -5, style: { fontSize: 10, fill: CHART_COLORS.axis } }} />
              <YAxis domain={[0.3, 1]} tick={{ fontSize: 10, fill: CHART_COLORS.axis }}
                label={{ value: 'TPR', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: CHART_COLORS.axis } }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#121622', border: '1px solid #1e2330', borderRadius: 4, fontSize: 11 }}
                labelFormatter={(v) => `FPR: ${Number(v).toExponential(3)}`}
                formatter={(val: number) => [(val as number).toFixed(4), 'TPR']}
              />
              <Line type="monotone" dataKey="tpr" stroke={CHART_COLORS.tpr} dot={false} strokeWidth={2} name="TPR" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Per-dataset accuracy table */}
      {Object.keys(datasets).length > 0 && (
        <div className="bg-[#0c0e14] border border-[#1e2330] rounded-sm p-3">
          <p className="text-[10px] text-slate-500 font-mono uppercase font-bold mb-2">
            {language === 'zh' ? '各数据集准确率' : 'Per-Dataset Accuracy'}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[11px] font-mono">
              <thead>
                <tr className="text-slate-500 uppercase text-[9px] border-b border-[#1e2330]">
                  <th className="py-1.5 pr-4">{language === 'zh' ? '数据集' : 'Dataset'}</th>
                  <th className="py-1.5 pr-4">{language === 'zh' ? '准确率' : 'Accuracy'}</th>
                  <th className="py-1.5">{language === 'zh' ? '阈值' : 'Threshold'}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#181d28]/40">
                {Object.entries(datasets).map(([name, info]) => (
                  <tr key={name} className="text-slate-300">
                    <td className="py-1 pr-4 font-bold text-white">{name}</td>
                    <td className="py-1 pr-4 text-cyan-400">{info.acc != null ? `${(info.acc * 100).toFixed(2)}%` : '-'}</td>
                    <td className="py-1 text-slate-500">{info.threshold?.toFixed(4) ?? '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Threshold sweep charts ──
function ThresholdSweepCharts({ result, language }: { result: Record<string, unknown>; language: Language }) {
  const chartData = (result.chart_data ?? {}) as Record<string, number[]>;
  const summary = (result.summary ?? []) as Array<Record<string, unknown>>;

  const thresholds = asNumArray(chartData.thresholds);
  const rank1 = asNumArray(chartData.rank1_accuracy);
  const tar = asNumArray(chartData.tar_at_far_1e3);
  const f1 = asNumArray(chartData.f1_score);
  const far = asNumArray(chartData.far);
  const frr = asNumArray(chartData.frr);

  // Build combined data
  const lineData = thresholds.map((t, i) => ({
    threshold: t,
    rank1: rank1[i] != null ? rank1[i] * 100 : null,
    tar: tar[i] != null ? tar[i] * 100 : null,
    f1: f1[i] != null ? f1[i] * 100 : null,
    far: far[i] ?? null,
    frr: frr[i] ?? null,
  }));

  return (
    <div className="space-y-4">
      {/* Metrics vs Threshold chart */}
      {lineData.length > 0 && (
        <div className="bg-[#0c0e14] border border-[#1e2330] rounded-sm p-3">
          <p className="text-[10px] text-slate-500 font-mono uppercase font-bold mb-2">
            {language === 'zh' ? '识别指标 vs 阈值' : 'Recognition Metrics vs Threshold'}
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={lineData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
              <XAxis dataKey="threshold" tick={{ fontSize: 10, fill: CHART_COLORS.axis }}
                label={{ value: 'Threshold', position: 'insideBottomRight', offset: -5, style: { fontSize: 10, fill: CHART_COLORS.axis } }} />
              <YAxis tick={{ fontSize: 10, fill: CHART_COLORS.axis }}
                label={{ value: '%', angle: -90, position: 'insideLeft', style: { fontSize: 10, fill: CHART_COLORS.axis } }} />
              <Tooltip contentStyle={{ backgroundColor: '#121622', border: '1px solid #1e2330', borderRadius: 4, fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="rank1" stroke={CHART_COLORS.rank1} dot={false} strokeWidth={1.5} name="Rank-1 Acc (%)" />
              <Line type="monotone" dataKey="tar" stroke={CHART_COLORS.tar} dot={false} strokeWidth={1.5} name="TAR@FAR=1e-3 (%)" />
              <Line type="monotone" dataKey="f1" stroke={CHART_COLORS.f1} dot={false} strokeWidth={1.5} name="F1-Score (%)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* FAR / FRR chart */}
      {lineData.length > 0 && (far.some(v => v != null) || frr.some(v => v != null)) && (
        <div className="bg-[#0c0e14] border border-[#1e2330] rounded-sm p-3">
          <p className="text-[10px] text-slate-500 font-mono uppercase font-bold mb-2">
            {language === 'zh' ? 'FAR / FRR vs 阈值' : 'FAR / FRR vs Threshold'}
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={lineData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
              <XAxis dataKey="threshold" tick={{ fontSize: 10, fill: CHART_COLORS.axis }}
                label={{ value: 'Threshold', position: 'insideBottomRight', offset: -5, style: { fontSize: 10, fill: CHART_COLORS.axis } }} />
              <YAxis tick={{ fontSize: 10, fill: CHART_COLORS.axis }} />
              <Tooltip contentStyle={{ backgroundColor: '#121622', border: '1px solid #1e2330', borderRadius: 4, fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="far" stroke={CHART_COLORS.fpr} dot={false} strokeWidth={1.5} name="FAR" />
              <Line type="monotone" dataKey="frr" stroke={CHART_COLORS.frr} dot={false} strokeWidth={1.5} name="FRR" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Summary table */}
      {summary.length > 0 && (
        <div className="bg-[#0c0e14] border border-[#1e2330] rounded-sm p-3 max-h-[200px] overflow-y-auto">
          <p className="text-[10px] text-slate-500 font-mono uppercase font-bold mb-2">
            {language === 'zh' ? '汇总表' : 'Summary Table'}
          </p>
          <table className="w-full text-left text-[10px] font-mono">
            <thead>
              <tr className="text-slate-500 uppercase text-[9px] border-b border-[#1e2330]">
                <th className="py-1 pr-2">Thr</th>
                <th className="py-1 pr-2">Rank-1</th>
                <th className="py-1 pr-2">FAR</th>
                <th className="py-1 pr-2">FRR</th>
                <th className="py-1 pr-2">TAR@1e-3</th>
                <th className="py-1 pr-2">F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#181d28]/40">
              {summary.map((row, i) => (
                <tr key={i} className="text-slate-400">
                  <td className="py-0.5 pr-2 text-cyan-400">{Number(row.threshold).toFixed(2)}</td>
                  <td className="py-0.5 pr-2">{(Number(row.rank1_accuracy) * 100).toFixed(1)}%</td>
                  <td className="py-0.5 pr-2">{Number(row.far).toExponential(2)}</td>
                  <td className="py-0.5 pr-2">{Number(row.frr).toFixed(4)}</td>
                  <td className="py-0.5 pr-2">{(Number(row.tar_at_far_1e3) * 100).toFixed(2)}%</td>
                  <td className="py-0.5">{(Number(row.f1_score) * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="bg-[#0c0e14] border border-[#1e2330] rounded-sm p-3 text-center">
      <p className="text-[9px] text-slate-500 font-mono uppercase tracking-wider">{label}</p>
      <p className={`text-lg font-bold font-mono mt-0.5 ${color}`}>{value}</p>
    </div>
  );
}
