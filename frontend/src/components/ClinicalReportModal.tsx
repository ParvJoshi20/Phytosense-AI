'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { X, Printer, Download, FileText, CheckCircle, AlertTriangle, ShieldCheck, Leaf } from 'lucide-react';
import { DiagnosticResult } from '@/types/diagnosis';

interface ClinicalReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: DiagnosticResult;
}

export const ClinicalReportModal: React.FC<ClinicalReportModalProps> = ({
  isOpen,
  onClose,
  result,
}) => {
  if (!isOpen) return null;

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm overflow-y-auto">
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.96 }}
        className="w-full max-w-3xl my-8 rounded-3xl bg-[#121215] border border-[#5B6987]/30 shadow-2xl overflow-hidden flex flex-col text-left"
      >
        {/* Modal Toolbar */}
        <div className="p-4 bg-[#121215] border-b border-[#5B6987]/20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-[#2A7FFF]" />
            <h3 className="text-sm font-bold text-[#E2E8F0]">
              Phytosense AI Clinical Diagnostic Summary
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="px-3 py-1.5 rounded-lg bg-[#08080A] hover:bg-white/5 border border-[#5B6987]/40 text-xs font-mono text-[#E2E8F0] flex items-center gap-1.5 transition-all"
            >
              <Printer className="w-3.5 h-3.5 text-[#2A7FFF]" />
              <span>Print / PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-[#828C9E] hover:text-[#E2E8F0] hover:bg-white/5 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Printable Report Document Body */}
        <div className="p-6 sm:p-8 space-y-6 bg-[#08080A] text-[#E2E8F0] overflow-y-auto max-h-[75vh]">
          {/* Header */}
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#5B6987]/30 pb-5">
            <div>
              <div className="flex items-center gap-2">
                <Leaf className="w-5 h-5 text-[#2A7FFF]" />
                <span className="text-lg font-extrabold tracking-tight">
                  Phytosense AI
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#2A7FFF]/20 text-[#2A7FFF]">
                  HXAI Report v2.4
                </span>
              </div>
              <p className="text-xs text-[#828C9E] mt-0.5">
                Pathology Diagnostic & Decision Support Dossier
              </p>
            </div>
            <div className="text-right text-xs font-mono text-[#828C9E]">
              <p>ID: {result.id}</p>
              <p>Date: {new Date(result.timestamp).toLocaleDateString()}</p>
            </div>
          </div>

          {/* Section 1: Specimen & Detection */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 p-4 rounded-2xl bg-[#121215] border border-[#5B6987]/30">
            <div className="space-y-1">
              <span className="text-[10px] font-mono text-[#828C9E] uppercase">
                Primary Diagnosis (Layer 1)
              </span>
              <h2 className="text-xl font-bold text-[#E2E8F0]">
                {result.layer1.primaryDisease}
              </h2>
              <p className="text-xs font-mono text-[#FFB020] italic">
                {result.layer1.scientificName}
              </p>
              <div className="pt-2 flex items-center gap-2 text-xs">
                <span className="px-2 py-0.5 rounded bg-[#FFB020]/20 text-[#FFB020] font-mono font-bold">
                  Severity: {result.layer1.severity}
                </span>
                <span className="text-[#828C9E] font-mono">
                  Confidence: {result.layer1.confidenceScore.toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="space-y-1 text-xs">
              <span className="text-[10px] font-mono text-[#828C9E] uppercase">
                Pathogen Taxonomy
              </span>
              <p className="text-[#E2E8F0] font-medium">
                Type: {result.layer1.pathogen.type} ({result.layer1.pathogen.commonName})
              </p>
              <p className="text-[#828C9E]">
                Optimum Temp: {result.layer1.pathogen.optimalTempRange}
              </p>
              <p className="text-[#828C9E]">
                Moisture Risk: {result.layer1.pathogen.humidityRiskThreshold}
              </p>
            </div>
          </div>

          {/* Section 2: Visual XAI & Morphological Attribution */}
          <div className="space-y-2">
            <h4 className="text-xs font-mono font-bold text-[#2A7FFF] uppercase tracking-wider">
              Layer 2 & 3: Visual Saliency & Reasoning Attribution
            </h4>
            <div className="p-4 rounded-2xl bg-[#121215] border border-[#5B6987]/30 space-y-2 text-xs">
              <p className="text-[#E2E8F0]">
                <strong className="text-[#2A7FFF]">Primary Symptom:</strong>{' '}
                {result.layer3.whyThisPrediction.primarySymptom}
              </p>
              <p className="text-[#828C9E]">
                <strong className="text-[#2A7FFF]">Latent Tensor Mapping:</strong>{' '}
                {result.layer3.whyThisPrediction.latentFeatureMapping}
              </p>
              <div className="pt-2">
                <span className="font-semibold text-[#E2E8F0] block mb-1">
                  Morphological Evidence:
                </span>
                <ul className="list-disc list-inside space-y-1 text-[#828C9E]">
                  {result.layer3.whyThisPrediction.morphologicalEvidence.map((e, idx) => (
                    <li key={idx}>{e}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Section 3: Recommended Treatment Plan */}
          <div className="space-y-2">
            <h4 className="text-xs font-mono font-bold text-[#1CEB76] uppercase tracking-wider">
              Layer 4: Agronomic Treatment Protocol
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3.5 rounded-2xl bg-[#121215] border border-[#5B6987]/30 space-y-1.5">
                <span className="font-bold text-[#FFB020] block">Immediate Cultural Action:</span>
                <p className="text-[#828C9E]">
                  Prune infected lower foliage exhibiting target rings and switch to ground-level drip irrigation.
                </p>
              </div>

              <div className="p-3.5 rounded-2xl bg-[#121215] border border-[#5B6987]/30 space-y-1.5">
                <span className="font-bold text-[#1CEB76] block">Bio-Fungicide (OMRI):</span>
                <p className="text-[#828C9E]">
                  Copper Octanoate (1 fl oz/gal) or Bacillus subtilis foliar spray. Reapply every 7 days.
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
