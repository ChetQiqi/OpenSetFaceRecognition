/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';

interface ProgressBarProps {
  /** 0–100. Omit for indeterminate animation. */
  value?: number;
  label?: string;
  size?: 'sm' | 'md';
}

export default function ProgressBar({ value, label, size = 'md' }: ProgressBarProps) {
  const isIndeterminate = value === undefined;
  const h = size === 'sm' ? 'h-1' : 'h-1.5';

  return (
    <div className="space-y-1.5">
      {(label || !isIndeterminate) && (
        <div className="flex justify-between text-[10px] font-mono">
          <span className="text-slate-500 uppercase font-bold tracking-wider">
            {label ?? 'Progress'}
          </span>
          {!isIndeterminate && (
            <span className="text-cyan-400 font-extrabold">{Math.round(value!)}%</span>
          )}
        </div>
      )}
      <div className={`w-full ${h} bg-slate-800 rounded-full overflow-hidden`}>
        {isIndeterminate ? (
          <div className="h-full w-1/2 bg-gradient-to-r from-transparent via-cyan-400 to-transparent rounded-full animate-[slide_1.2s_ease-in-out_infinite]" />
        ) : (
          <div
            className={`h-full rounded-full transition-all duration-500 ease-out ${
              (value ?? 0) >= 100 ? 'bg-emerald-500' : 'bg-gradient-to-r from-cyan-500 to-cyan-400'
            }`}
            style={{ width: `${Math.min(100, Math.max(0, value ?? 0))}%` }}
          />
        )}
      </div>
    </div>
  );
}
