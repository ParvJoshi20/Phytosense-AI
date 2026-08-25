'use client';

import React, { useState } from 'react';
import { Layer4DecisionSupportData, TreatmentAction } from '@/types/diagnosis';
import { Stethoscope, AlertOctagon, CheckSquare, Square, Shield, Sprout, FlaskConical, Wind, Calendar, DollarSign } from 'lucide-react';

interface Layer4DecisionSupportProps {
  data: Layer4DecisionSupportData;
}

export const Layer4DecisionSupport: React.FC<Layer4DecisionSupportProps> = ({ data }) => {
  const [completedActions, setCompletedActions] = useState<Record<string, boolean>>({});

  const toggleAction = (id: string) => {
    setCompletedActions((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const plan = data.actionPlan;

  return (
    <div className="rounded-2xl border border-emerald-950/80 bg-[#0d1611]/90 p-5 sm:p-6 shadow-xl space-y-6">
      {/* Layer Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-emerald-950/80 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-950 border border-emerald-600/40 text-emerald-400 font-mono text-xs font-bold">
            L4
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-white tracking-wide">
              Layer 4: Contextualized Decision Support & Treatment Protocol
            </h3>
            <p className="text-[11px] text-slate-400">
              Evidence-based agronomic prescriptions broken into immediate, organic, and chemical protocols
            </p>
          </div>
        </div>

        {/* Economic Risk Badge */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-xl bg-black/40 border border-emerald-950 text-xs font-mono">
          <span className="text-slate-400">Yield Risk:</span>
          <span
            className={`font-bold ${
              data.economicRisk === 'Severe Yield Loss Warning'
                ? 'text-rose-400'
                : data.economicRisk === 'Moderate'
                ? 'text-amber-400'
                : 'text-emerald-400'
            }`}
          >
            {data.economicRisk}
          </span>
        </div>
      </div>

      {/* Weather Alert if present */}
      {data.weatherRiskWarning && (
        <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-amber-950/30 border border-amber-800/40 text-xs text-amber-300">
          <AlertOctagon className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold block mb-0.5">Microclimate Alert</span>
            <span>{data.weatherRiskWarning}</span>
          </div>
        </div>
      )}

      {/* 3 Main Action Modules */}
      <div className="space-y-6">
        {/* Module 1: Immediate Containment */}
        {plan.immediateContainment.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-bold text-rose-400 uppercase tracking-wider">
              <Shield className="w-4 h-4 text-rose-400" />
              <span>Phase 1: Immediate Cultural Containment (Within 24-48h)</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {plan.immediateContainment.map((action) => {
                const isDone = !!completedActions[action.id];
                return (
                  <div
                    key={action.id}
                    onClick={() => toggleAction(action.id)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer select-none space-y-2 ${
                      isDone
                        ? 'bg-emerald-950/20 border-emerald-800/40 opacity-70'
                        : 'bg-black/40 border-emerald-950 hover:border-emerald-800/70'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {isDone ? (
                          <CheckSquare className="w-4 h-4 text-emerald-400 shrink-0" />
                        ) : (
                          <Square className="w-4 h-4 text-slate-500 shrink-0" />
                        )}
                        <span className={`text-xs font-bold ${isDone ? 'line-through text-slate-400' : 'text-white'}`}>
                          {action.title}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800/40">
                        {action.timing}
                      </span>
                    </div>

                    <p className="text-xs text-slate-300 leading-relaxed pl-6">
                      {action.description}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Module 2: Organic & Biological Interventions */}
        {plan.organicBiological.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-400 uppercase tracking-wider">
              <Sprout className="w-4 h-4 text-emerald-400" />
              <span>Phase 2: Organic & Bio-Fungicide Inoculation (OMRI Listed)</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {plan.organicBiological.map((action) => {
                const isDone = !!completedActions[action.id];
                return (
                  <div
                    key={action.id}
                    onClick={() => toggleAction(action.id)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer select-none space-y-2 ${
                      isDone
                        ? 'bg-emerald-950/20 border-emerald-800/40 opacity-70'
                        : 'bg-black/40 border-emerald-950 hover:border-emerald-800/70'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {isDone ? (
                          <CheckSquare className="w-4 h-4 text-emerald-400 shrink-0" />
                        ) : (
                          <Square className="w-4 h-4 text-slate-500 shrink-0" />
                        )}
                        <span className={`text-xs font-bold ${isDone ? 'line-through text-slate-400' : 'text-white'}`}>
                          {action.title}
                        </span>
                      </div>
                      {action.safetyInterval && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/40">
                          {action.safetyInterval}
                        </span>
                      )}
                    </div>

                    <p className="text-xs text-slate-300 leading-relaxed pl-6">
                      {action.description}
                    </p>

                    {action.dosage && (
                      <div className="pl-6 pt-1 flex items-center gap-2 text-[11px] font-mono text-emerald-400">
                        <FlaskConical className="w-3.5 h-3.5" />
                        <span>Dosage: {action.dosage}</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Module 3: Targeted Chemical Treatments */}
        {plan.targetedChemical.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-xs font-bold text-amber-400 uppercase tracking-wider">
              <FlaskConical className="w-4 h-4 text-amber-400" />
              <span>Phase 3: Targeted Synthetic Chemistries & FRAC Rotation</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {plan.targetedChemical.map((action) => {
                const isDone = !!completedActions[action.id];
                return (
                  <div
                    key={action.id}
                    onClick={() => toggleAction(action.id)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer select-none space-y-2 ${
                      isDone
                        ? 'bg-emerald-950/20 border-emerald-800/40 opacity-70'
                        : 'bg-black/40 border-emerald-950 hover:border-emerald-800/70'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {isDone ? (
                          <CheckSquare className="w-4 h-4 text-emerald-400 shrink-0" />
                        ) : (
                          <Square className="w-4 h-4 text-slate-500 shrink-0" />
                        )}
                        <span className={`text-xs font-bold ${isDone ? 'line-through text-slate-400' : 'text-white'}`}>
                          {action.title}
                        </span>
                      </div>
                      {action.safetyInterval && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800/40">
                          {action.safetyInterval}
                        </span>
                      )}
                    </div>

                    <p className="text-xs text-slate-300 leading-relaxed pl-6">
                      {action.description}
                    </p>

                    {action.dosage && (
                      <div className="pl-6 pt-1 flex items-center gap-2 text-[11px] font-mono text-amber-400">
                        <FlaskConical className="w-3.5 h-3.5" />
                        <span>Dosage: {action.dosage}</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Environmental & Cultural Practices */}
        {plan.environmentalAdjustments.length > 0 && (
          <div className="p-4 rounded-xl bg-black/40 border border-emerald-950 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
              <Wind className="w-3.5 h-3.5 text-cyan-400" />
              <span>Cultural & Environmental Adjustments</span>
            </div>
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
              {plan.environmentalAdjustments.map((env, idx) => (
                <li key={idx} className="flex items-start gap-2 text-xs text-slate-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
                  <span>{env}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Prognosis & Monitoring Schedule Banner */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
          <div className="p-3.5 rounded-xl bg-emerald-950/30 border border-emerald-900/40 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-emerald-400 font-semibold">
              <Calendar className="w-3.5 h-3.5" />
              <span>Monitoring Interval</span>
            </div>
            <p className="text-xs text-slate-200">{plan.monitoringSchedule}</p>
          </div>

          <div className="p-3.5 rounded-xl bg-emerald-950/30 border border-emerald-900/40 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-emerald-400 font-semibold">
              <Stethoscope className="w-3.5 h-3.5" />
              <span>Agronomic Prognosis</span>
            </div>
            <p className="text-xs text-slate-200">{plan.prognosis}</p>
          </div>
        </div>
      </div>
    </div>
  );
};
