import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <AlertTriangle className="w-10 h-10 text-red-400" />
      <p className="text-xs font-mono text-red-300 max-w-md text-center">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-1.5 px-4 py-2 bg-red-950/30 border border-red-500/30 text-red-300 text-[10px] font-mono uppercase font-bold rounded-sm hover:bg-red-950/50 transition-colors cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Retry
        </button>
      )}
    </div>
  );
}
