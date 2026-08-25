'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Leaf, History, Library, Bot, ShieldCheck, Activity, Plus, FileText } from 'lucide-react';
import { DiagnosticResult, SeverityTier } from '@/types/diagnosis';
import { BENCHMARK_SAMPLES } from '@/data/benchmarkDataset';

interface SidebarProps {
  recentScans: DiagnosticResult[];
  selectedScanId: string | null;
  onSelectScan: (scan: DiagnosticResult) => void;
  onNewScan: () => void;
  onOpenAgronomist: () => void;
  onOpenReport?: () => void;
  currentResult: DiagnosticResult | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  recentScans,
  selectedScanId,
  onSelectScan,
  onNewScan,
  onOpenAgronomist,
  onOpenReport,
  currentResult,
}) => {
  const getStatusDotColor = (severity: SeverityTier) => {
    switch (severity) {
      case 'Healthy':
        return 'bg-[#1CEB76] shadow-[#1CEB76]/50'; // Neon Vine
      case 'Mild':
        return 'bg-[#FDE12D] shadow-[#FDE12D]/50'; // Solar Pollen
      case 'Moderate':
        return 'bg-[#FFB020] shadow-[#FFB020]/50'; // Alert Amber
      case 'Severe':
        return 'bg-[#FF2A4D] shadow-[#FF2A4D]/50'; // Piercing Crimson
      default:
        return 'bg-[#828C9E] shadow-[#828C9E]/50';
    }
  };

  return (
    <aside
      id="sidebar-container"
      className="w-full lg:w-[25%] lg:min-w-[280px] lg:max-w-[340px] h-full bg-[#121215] border-r border-[#5B6987]/20 flex flex-col justify-between select-none z-30 shrink-0"
    >
      {/* Top Branding Section */}
      <div className="p-5 border-b border-[#5B6987]/15">
        <div className="flex items-center gap-3">
          <div className="relative w-9 h-9 rounded-xl bg-gradient-to-br from-[#2A7FFF]/20 to-[#121215] border border-[#2A7FFF]/40 flex items-center justify-center text-[#2A7FFF] shadow-md shadow-[#2A7FFF]/20">
            <Leaf className="w-5 h-5" />
            <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-[#1CEB76] animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h1 className="font-bold text-base tracking-tight text-[#E2E8F0] font-sans">
                Phytosense <span className="text-[#2A7FFF] font-mono">AI</span>
              </h1>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[#2A7FFF]/10 text-[#2A7FFF] border border-[#2A7FFF]/30">
                HXAI
              </span>
            </div>
            <p className="text-[11px] text-[#828C9E] font-medium tracking-tight">
              Explainable Decision Support
            </p>
          </div>
        </div>

        {/* Quick New Scan CTA */}
        <motion.button
          id="sidebar-new-scan-btn"
          whileTap={{ scale: 0.96 }}
          onClick={onNewScan}
          className="mt-4 w-full py-2 px-3 rounded-xl bg-[#2A7FFF] hover:bg-[#2A7FFF]/90 text-[#08080A] font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-[#2A7FFF]/25 transition-all"
        >
          <Plus className="w-4 h-4 stroke-[2.5]" />
          <span>New Tomato Leaf Scan</span>
        </motion.button>
      </div>

      {/* Middle Scrollable Section: Recent Scans History */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div>
          <div className="flex items-center justify-between px-1 mb-2.5">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-[#828C9E] uppercase tracking-wider">
              <History className="w-3.5 h-3.5 text-[#5B6987]" />
              <span>Recent Scans</span>
            </div>
            <span className="text-[10px] font-mono text-[#5B6987]">
              {recentScans.length} Cached
            </span>
          </div>

          {/* History List */}
          {recentScans.length === 0 ? (
            <div className="p-4 rounded-xl border border-dashed border-[#5B6987]/30 bg-[#08080A]/40 text-center">
              <p className="text-xs text-[#828C9E]">No scans recorded yet.</p>
              <p className="text-[10px] text-[#5B6987] mt-1">
                Upload a leaf image or run a demo cycle to populate history.
              </p>
            </div>
          ) : (
            <div className="space-y-1.5">
              {recentScans.map((scan) => {
                const isSelected = selectedScanId === scan.id;
                const dotColor = getStatusDotColor(scan.layer1.severity);

                return (
                  <motion.button
                    key={scan.id}
                    id={`recent-scan-${scan.id}`}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => onSelectScan(scan)}
                    className={`w-full p-2.5 rounded-xl text-left flex items-center gap-3 transition-all border ${
                      isSelected
                        ? 'bg-[rgba(255,255,255,0.08)] border-[#2A7FFF]/60 shadow-md shadow-[#2A7FFF]/10'
                        : 'bg-[#08080A]/60 hover:bg-[rgba(255,255,255,0.04)] border-[#5B6987]/20 hover:border-[#5B6987]/40'
                    }`}
                  >
                    {/* Small Thumbnail Placeholder */}
                    <div className="relative w-10 h-10 rounded-lg overflow-hidden bg-black border border-[#5B6987]/30 shrink-0">
                      <img
                        src={scan.imageUrl}
                        alt="Scan thumbnail"
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-cover"
                      />
                      {/* Tiny Colored Status Dot */}
                      <span
                        className={`absolute top-1 right-1 w-2 h-2 rounded-full shadow-sm ${dotColor}`}
                      />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-semibold text-[#E2E8F0] truncate">
                          {scan.layer1.primaryDisease}
                        </span>
                        <span className="text-[10px] font-mono text-[#828C9E] shrink-0">
                          {scan.layer1.confidenceScore.toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-[#828C9E] font-mono mt-0.5">
                        <span className="truncate italic">
                          {scan.layer1.scientificName}
                        </span>
                        <span
                          className={`font-semibold shrink-0 ${
                            scan.layer1.severity === 'Healthy'
                              ? 'text-[#1CEB76]'
                              : scan.layer1.severity === 'Severe'
                              ? 'text-[#FF2A4D]'
                              : scan.layer1.severity === 'Moderate'
                              ? 'text-[#FFB020]'
                              : 'text-[#FDE12D]'
                          }`}
                        >
                          {scan.layer1.severity}
                        </span>
                      </div>
                    </div>
                  </motion.button>
                );
              })}
            </div>
          )}
        </div>

        {/* Clinical Presets / Benchmark Data Quick Loader */}
        <div className="pt-2">
          <div className="flex items-center gap-1.5 px-1 mb-2 text-xs font-semibold text-[#828C9E] uppercase tracking-wider">
            <Library className="w-3.5 h-3.5 text-[#2A7FFF]" />
            <span>Preset Pathology Profiles</span>
          </div>

          <div className="grid grid-cols-2 gap-1.5">
            {BENCHMARK_SAMPLES.slice(0, 4).map((sample) => (
              <motion.button
                key={sample.id}
                whileTap={{ scale: 0.96 }}
                onClick={() => onSelectScan(sample.precomputedResult)}
                className="p-2 rounded-lg bg-[#08080A]/80 hover:bg-[rgba(255,255,255,0.06)] border border-[#5B6987]/20 hover:border-[#2A7FFF]/40 text-left transition-all group"
              >
                <span className="text-[11px] font-medium text-[#E2E8F0] group-hover:text-[#2A7FFF] truncate block transition-colors">
                  {sample.diseaseCategory}
                </span>
                <span className="text-[9px] font-mono text-[#828C9E] block">
                  {sample.severity} Tier
                </span>
              </motion.button>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Footer Section: AI Agronomist & Status */}
      <div className="p-4 border-t border-[#5B6987]/15 bg-[#121215] space-y-2">
        {currentResult && onOpenReport && (
          <motion.button
            whileTap={{ scale: 0.96 }}
            onClick={onOpenReport}
            className="w-full py-2 px-3 rounded-xl bg-[rgba(255,255,255,0.06)] hover:bg-[rgba(255,255,255,0.1)] border border-[#5B6987]/30 text-[#E2E8F0] text-xs font-medium flex items-center justify-center gap-2 transition-all"
          >
            <FileText className="w-4 h-4 text-[#2A7FFF]" />
            <span>Export Clinical Report</span>
          </motion.button>
        )}

        <motion.button
          whileTap={{ scale: 0.96 }}
          onClick={onOpenAgronomist}
          className="w-full py-2 px-3 rounded-xl bg-[#2A7FFF]/10 hover:bg-[#2A7FFF]/20 border border-[#2A7FFF]/40 text-[#2A7FFF] text-xs font-semibold flex items-center justify-center gap-2 transition-all shadow-sm"
        >
          <Bot className="w-4 h-4" />
          <span>Ask AI Agronomist</span>
        </motion.button>

        <div className="flex items-center justify-between text-[10px] font-mono text-[#828C9E] pt-1 px-1">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#1CEB76] animate-ping" />
            <span>HXAI Engine Online</span>
          </span>
          <span>v2.4</span>
        </div>
      </div>
    </aside>
  );
};
