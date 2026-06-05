import React from 'react';
import { RefreshCw } from 'lucide-react';

interface LoadingSpinnerProps {
  message?: string;
}

export default function LoadingSpinner({ message = 'Loading...' }: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
      <p className="text-[10px] font-mono uppercase tracking-widest text-slate-500">{message}</p>
    </div>
  );
}
