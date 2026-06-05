import React from 'react';
import { Database } from 'lucide-react';

interface EmptyStateProps {
  message: string;
  icon?: React.ReactNode;
}

export default function EmptyState({ message, icon }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 opacity-30">
      {icon || <Database className="w-10 h-10 text-slate-500" />}
      <p className="text-[10px] font-mono uppercase tracking-widest text-slate-400">{message}</p>
    </div>
  );
}
