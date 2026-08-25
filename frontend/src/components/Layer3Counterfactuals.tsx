'use client';

import React from 'react';
import { Layer3ReasoningData } from '@/types/diagnosis';
import { Brain, HelpCircle, Check, X, ArrowRight, ShieldCheck, Cpu } from 'lucide-react';

interface Layer3CounterfactualsProps {
  data: Layer3ReasoningData;
}

export const Layer3Counterfactuals: React.FC<Layer3CounterfactualsProps> = ({ data }) => {
  return (
    <div className="rounded-2xl border border-emerald-950/80 bg-[#0d1611]/90 p-5 sm:p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-emerald-950/80 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-950 border border-emerald-600/40 text-emerald-400 font-mono text-xs font-bold">
            L3
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-white tracking-wide">
              Layer 3: Natural Language Reasoning & Counterfactuals
            </h3>
            <p className="text-[11px] text-slate-400">
              Explainable AI reasoning mapping latent representations to human-understandable symptoms
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/40 text-xs font-mono text-emerald-300">
          <Brain className="w-3.5 h-3.5 text-emerald-400" />
          <span>HXAI Reasoning Engine</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Section 1: "Why This Prediction?" Attribution */}
        <div className="lg:col-span-6 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-emerald-400 uppercase tracking-wider">
            <Check className="w-4 h-4 text-emerald-400" />
            <span>Why This Prediction? (Symptom Attribution)</span>
          </div>

          <div className="p-4 rounded-xl bg-black/40 border border-emerald-950 space-y-3.5">
            <div>
              <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">
                Primary Diagnostic Driver
              </span>
              <p className="text-xs sm:text-sm text-slate-200 font-medium leading-relaxed">
                {data.whyThisPrediction.primarySymptom}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-900/40 space-y-1.5">
              <div className="flex items-center gap-1.5 text-xs text-emerald-300 font-semibold">
                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                <span>Latent Space Feature Mapping</span>
              </div>
              <p className="text-xs text-slate-300 font-mono text-[11px] leading-relaxed">
                {data.whyThisPrediction.latentFeatureMapping}
              </p>
            </div>

            {/* Evidence Checklist */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-mono text-slate-400 block">
                Observed Morphological Evidence:
              </span>
              <ul className="space-y-1.5">
                {data.whyThisPrediction.morphologicalEvidence.map((evidence, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                    <span>{evidence}</span>
                  </li>
                ))}
              </ul>
            </div>

            {data.whyThisPrediction.canopyPattern && (
              <div className="pt-2 border-t border-emerald-950/60 text-[11px] text-slate-400 flex items-center gap-1.5">
                <span className="font-mono text-emerald-400">Canopy Location:</span>
                <span>{data.whyThisPrediction.canopyPattern}</span>
              </div>
            )}
          </div>
        </div>

        {/* Section 2: "Why Not Another Disease?" Counterfactuals */}
        <div className="lg:col-span-6 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-amber-400 uppercase tracking-wider">
            <HelpCircle className="w-4 h-4 text-amber-400" />
            <span>Why Not Lookalike Diseases? (Counterfactuals)</span>
          </div>

          <div className="space-y-3">
            {data.counterfactuals.map((cf, idx) => (
              <div
                key={idx}
                className="p-4 rounded-xl bg-black/40 border border-emerald-950 hover:border-emerald-800/60 transition-colors space-y-2.5"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white">
                      Not {cf.alternativeDisease}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400 italic">
                      ({cf.scientificName})
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
                    Probability: {cf.probability.toFixed(1)}%
                  </span>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">
                  {cf.whyNotReason}
                </p>

                {/* Missing Marker Callout */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                  <div className="p-2 rounded-lg bg-rose-950/20 border border-rose-900/30 text-[11px] space-y-0.5">
                    <span className="text-[10px] font-mono text-rose-400 flex items-center gap-1">
                      <X className="w-3 h-3 text-rose-400" /> Missing Marker
                    </span>
                    <p className="text-slate-300 font-medium truncate">{cf.missingMarker}</p>
                  </div>

                  <div className="p-2 rounded-lg bg-emerald-950/20 border border-emerald-900/30 text-[11px] space-y-0.5">
                    <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                      <Check className="w-3 h-3 text-emerald-400" /> Key Differentiator
                    </span>
                    <p className="text-slate-300 font-medium truncate">{cf.differentiatingFeature}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Model Certainty Assessment */}
          <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-300">
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            <span className="text-[11px]">{data.confidenceAssessment}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
