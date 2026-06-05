import React from 'react';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import { useToast } from '../hooks/useToast';
import type { Toast } from '../context/ToastContext';

const iconMap: Record<Toast['type'], React.ReactNode> = {
  success: <CheckCircle className="w-4 h-4 text-emerald-400" />,
  error: <AlertCircle className="w-4 h-4 text-red-400" />,
  warning: <AlertTriangle className="w-4 h-4 text-amber-400" />,
  info: <Info className="w-4 h-4 text-cyan-400" />,
};

const borderMap: Record<Toast['type'], string> = {
  success: 'border-emerald-500/30',
  error: 'border-red-500/30',
  warning: 'border-amber-500/30',
  info: 'border-cyan-500/30',
};

export default function ToastContainer() {
  const { toasts, removeToast } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm">
      {toasts.map(toast => (
        <div
          key={toast.id}
          className={`bg-[#121622] border ${borderMap[toast.type]} rounded-sm px-4 py-3 flex items-start gap-2 shadow-lg animate-in slide-in-from-right`}
        >
          <span className="shrink-0 mt-0.5">{iconMap[toast.type]}</span>
          <p className="text-xs font-mono text-slate-300 flex-1">{toast.message}</p>
          <button
            onClick={() => removeToast(toast.id)}
            className="text-slate-500 hover:text-white shrink-0 cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}
