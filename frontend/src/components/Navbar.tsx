'use client';

import React from 'react';
import { Leaf, Sparkles, History, FileText, Bot, HelpCircle, Activity } from 'lucide-react';
import { DiagnosticResult } from '@/types/diagnosis';

interface NavbarProps {
  onOpenHistory: () => void;
  onOpenChat: () => void;
  onOpenReport: () => void;
  onSelectSample: (sampleId: string) => void;
  currentResult: DiagnosticResult | null;
  historyCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  onOpenHistory,
  onOpenChat,
  onOpenReport,
  onSelectSample,
  currentResult,
  historyCount,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-emerald-950/60 bg-[#090f0c]/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo & Architecture Badge */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-950/80 border border-emerald-500/40 text-emerald-400 glow-emerald">
            <Leaf className="w-5 h-5" />
            <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-lg tracking-tight text-white flex items-center gap-1.5">
                Phytosense <span className="text-emerald-400 font-mono">AI</span>
              </h1>
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-emerald-950/80 text-emerald-300 border border-emerald-700/50">
                HXAI v2.4
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Intelligent Decision Support System for Tomato Pathology
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* HXAI Status Pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-800/40 text-xs font-mono text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
            <Activity className="w-3.5 h-3.5" />
            <span>4-LAYER HXAI ACTIVE</span>
          </div>

          {/* AI Agronomist Chat Button */}
          <button
            id="nav-chat-btn"
            onClick={onOpenChat}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/60 hover:bg-emerald-900/60 border border-emerald-600/40 hover:border-emerald-500 text-emerald-300 text-xs font-medium transition-all shadow-sm group"
          >
            <Bot className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
            <span className="hidden sm:inline">AI Agronomist</span>
          </button>

          {/* Clinical Report Button */}
          {currentResult && (
            <button
              id="nav-report-btn"
              onClick={onOpenReport}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-700 hover:border-slate-600 text-slate-200 text-xs font-medium transition-all"
            >
              <FileText className="w-4 h-4 text-emerald-400" />
              <span className="hidden sm:inline">Clinical Report</span>
            </button>
          )}

          {/* History Scan Button */}
          <button
            id="nav-history-btn"
            onClick={onOpenHistory}
            className="relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-700/80 hover:border-slate-600 text-slate-300 text-xs font-medium transition-all"
          >
            <History className="w-4 h-4 text-slate-400" />
            <span className="hidden sm:inline">Scans</span>
            {historyCount > 0 && (
              <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[10px] font-mono font-bold bg-emerald-600 text-white rounded-full">
                {historyCount}
              </span>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
