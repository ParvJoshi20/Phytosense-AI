'use client';

import React from 'react';
import { Layer1DetectionData } from '@/types/diagnosis';
import { ShieldAlert, CheckCircle, AlertTriangle, Bug, Dna, Thermometer, Droplets, PieChart, Sparkles } from 'lucide-react';

interface Layer1DetectionProps {
  data: Layer1DetectionData;
}

export const Layer1Detection: React.FC<Layer1DetectionProps> = ({ data }) => {
  const isHealthy = data.severity === 'Healthy';
  const isSevere = data.severity === 'Severe';
  const isModerate = data.severity === 'Moderate';
  const isMild = data.severity === 'Mild';

  // Severity color maps
  const severityColors = {
    Healthy: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50 glow-emerald',
    Mild: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/50',
    Moderate: 'bg-amber-500/20 text-amber-300 border-amber-500/50 glow-amber',
    Severe: 'bg-rose-500/20 text-rose-300 border-rose-500/50 glow-rose',
  };

  return (
    <div className="rounded-2xl border border-emerald-950/80 bg-[#0d1611]/90 p-5 sm:p-6 shadow-xl space-y-6">
      {/* Header with Layer Tag */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-emerald-950/80 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-950 border border-emerald-600/40 text-emerald-400 font-mono text-xs font-bold">
            L1
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-white tracking-wide">
              Layer 1: Disease Detection & Severity Spectrum
            </h3>
            <p className="text-[11px] text-slate-400">
              Calibrated multi-class classification across 4-tier pathology spectrum
            </p>
          </div>
        </div>

        {/* Severity Badge */}
        <div className={`flex items-center gap-2 px-3 py-1 rounded-xl border font-mono text-xs font-bold ${severityColors[data.severity]}`}>
          {isHealthy ? (
            <CheckCircle className="w-4 h-4 text-emerald-400" />
          ) : isSevere ? (
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          )}
          <span>Severity: {data.severity.toUpperCase()}</span>
        </div>
      </div>

      {/* Primary Diagnosis Main Banner */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
        {/* Disease Title & Scientific Name */}
        <div className="md:col-span-7 space-y-2">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-emerald-950 text-emerald-400 border border-emerald-800/60">
              {data.pathogen.type} Pathology
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Foliar Canopy: {data.affectedCanopyEstimate}
            </span>
          </div>

          <h2 className="text-xl sm:text-2xl font-extrabold text-white tracking-tight">
            {data.primaryDisease}
          </h2>
          <p className="text-xs font-mono text-emerald-400/90 italic">
            {data.scientificName}
          </p>
        </div>

        {/* Confidence Dial / Metric Card */}
        <div className="md:col-span-5 flex items-center justify-between sm:justify-end gap-4 p-3.5 rounded-xl bg-black/40 border border-emerald-950">
          <div className="text-right">
            <span className="text-[10px] font-mono uppercase text-slate-400 block">
              Confidence Score
            </span>
            <span className="text-2xl sm:text-3xl font-extrabold font-mono text-emerald-400">
              {data.confidenceScore.toFixed(1)}%
            </span>
            <span className="text-[10px] text-emerald-500/80 block">
              Calibrated Bayesian Posterior
            </span>
          </div>

          {/* Mini Ring Meter */}
          <div className="relative w-14 h-14 shrink-0 flex items-center justify-center">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
              <path
                className="text-slate-800"
                strokeWidth="3.5"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className={isSevere ? 'text-rose-500' : isModerate ? 'text-amber-500' : 'text-emerald-400'}
                strokeDasharray={`${data.confidenceScore}, 100`}
                strokeWidth="3.5"
                strokeLinecap="round"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <Sparkles className="w-5 h-5 absolute text-emerald-400/80" />
          </div>
        </div>
      </div>

      {/* Pathogen Biological Profile Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="p-3 rounded-xl bg-black/30 border border-emerald-950/60 space-y-1">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Dna className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-medium">Pathogen Taxonomy</span>
          </div>
          <p className="text-xs font-semibold text-slate-200 truncate">
            {data.pathogen.commonName}
          </p>
          <p className="text-[10px] text-slate-400 font-mono truncate">
            {data.pathogen.family ? `Fam: ${data.pathogen.family}` : 'Domain: Solanaceae'}
          </p>
        </div>

        <div className="p-3 rounded-xl bg-black/30 border border-emerald-950/60 space-y-1">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Thermometer className="w-3.5 h-3.5 text-amber-400" />
            <span className="font-medium">Thermal Optimum</span>
          </div>
          <p className="text-xs font-semibold text-slate-200">
            {data.pathogen.optimalTempRange}
          </p>
          <p className="text-[10px] text-slate-400">Spore germination envelope</p>
        </div>

        <div className="p-3 rounded-xl bg-black/30 border border-emerald-950/60 space-y-1">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Droplets className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-medium">Moisture Risk Trigger</span>
          </div>
          <p className="text-xs font-semibold text-slate-200 line-clamp-1">
            {data.pathogen.humidityRiskThreshold}
          </p>
          <p className="text-[10px] text-slate-400">Microclimate vulnerability</p>
        </div>
      </div>

      {/* Calibrated Probability Distribution Bars */}
      <div className="space-y-2.5 pt-2">
        <div className="flex items-center justify-between text-xs text-slate-300">
          <div className="flex items-center gap-1.5">
            <PieChart className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-semibold">Candidate Pathogen Probabilities</span>
          </div>
          <span className="text-[10px] font-mono text-slate-400">Top Softmax Distributions</span>
        </div>

        <div className="space-y-2">
          {data.probabilities.map((item, idx) => (
            <div key={idx} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className={`font-medium ${item.isTopPrediction ? 'text-emerald-300 font-bold' : 'text-slate-300'}`}>
                  {item.disease} <span className="text-[10px] text-slate-500 italic font-mono">({item.scientificName})</span>
                </span>
                <span className="font-mono font-semibold text-slate-200">
                  {item.probability.toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    item.isTopPrediction
                      ? isSevere
                        ? 'bg-rose-500'
                        : isModerate
                        ? 'bg-amber-500'
                        : 'bg-emerald-400'
                      : 'bg-slate-700'
                  }`}
                  style={{ width: `${Math.max(item.probability, 2)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
