'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Sidebar,
} from '@/components/Sidebar';
import {
  IdleDropzone,
} from '@/components/IdleDropzone';
import {
  ScanningView,
} from '@/components/ScanningView';
import {
  DiagnosisResultsView,
} from '@/components/DiagnosisResultsView';
import {
  CustomCursor,
} from '@/components/CustomCursor';
import {
  AgronomistModal,
} from '@/components/AgronomistModal';
import {
  ClinicalReportModal,
} from '@/components/ClinicalReportModal';
import { DiagnosticResult, BenchmarkSample } from '@/types/diagnosis';
import { BENCHMARK_SAMPLES } from '@/data/benchmarkDataset';
import { Play, Sparkles, RefreshCw, Layers, ShieldCheck, Activity } from 'lucide-react';

export default function Home() {
  // App UX States: 1 = IDLE, 2 = SCANNING, 3 = DIAGNOSIS
  const [appState, setAppState] = useState<'IDLE' | 'SCANNING' | 'DIAGNOSIS'>('IDLE');

  // Currently active specimen / diagnosis result
  const [currentImage, setCurrentImage] = useState<string>(BENCHMARK_SAMPLES[0].thumbnail);
  const [currentImageName, setCurrentImageName] = useState<string>('Early Blight Specimen');
  const [currentResult, setCurrentResult] = useState<DiagnosticResult | null>(
    BENCHMARK_SAMPLES[0].precomputedResult
  );

  // Scans history
  const [recentScans, setRecentScans] = useState<DiagnosticResult[]>([
    BENCHMARK_SAMPLES[0].precomputedResult,
    BENCHMARK_SAMPLES[1].precomputedResult,
    BENCHMARK_SAMPLES[2].precomputedResult,
    BENCHMARK_SAMPLES[3].precomputedResult,
  ]);

  const [selectedScanId, setSelectedScanId] = useState<string | null>(
    BENCHMARK_SAMPLES[0].precomputedResult.id
  );

  // Modals
  const [isAgronomistOpen, setIsAgronomistOpen] = useState(false);
  const [isReportOpen, setIsReportOpen] = useState(false);

  // Dynamic Theme Cursor & Ambient Light Color
  const getActiveThemeColor = () => {
    if (appState === 'IDLE') return '#2A7FFF'; // Scanner Blue
    if (appState === 'SCANNING') return '#2A7FFF'; // Scanner Blue
    if (!currentResult) return '#FFB020'; // Alert Amber default

    switch (currentResult.layer1.severity) {
      case 'Healthy':
        return '#1CEB76'; // Neon Vine
      case 'Mild':
        return '#FDE12D'; // Solar Pollen
      case 'Moderate':
        return '#FFB020'; // Alert Amber
      case 'Severe':
        return '#FF2A4D'; // Piercing Crimson
      default:
        return '#FFB020';
    }
  };

  // State Transition Handlers
  const handleImageSelected = (base64: string, name?: string) => {
    setCurrentImage(base64);
    setCurrentImageName(name || 'Uploaded Leaf Specimen');
    setAppState('SCANNING');

    // Trigger AI Diagnostic API or construct result
    fetch('/api/diagnose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageBase64: base64, imageName: name }),
    })
      .then((res) => res.json())
      .then((data: DiagnosticResult) => {
        setCurrentResult(data);
        setSelectedScanId(data.id);
        setRecentScans((prev) => [data, ...prev.filter((s) => s.id !== data.id)]);
      })
      .catch((err) => {
        console.warn('Diagnosis fallback:', err);
        // Fallback to Early Blight
        const eb = BENCHMARK_SAMPLES[0].precomputedResult;
        setCurrentResult(eb);
        setSelectedScanId(eb.id);
      });
  };

  const handleBenchmarkSelected = (sample: BenchmarkSample) => {
    setCurrentImage(sample.thumbnail);
    setCurrentImageName(sample.title);
    setCurrentResult(sample.precomputedResult);
    setSelectedScanId(sample.precomputedResult.id);
    setRecentScans((prev) => [
      sample.precomputedResult,
      ...prev.filter((s) => s.id !== sample.precomputedResult.id),
    ]);
    setAppState('SCANNING');
  };

  const handleSelectHistoryScan = (scan: DiagnosticResult) => {
    setCurrentResult(scan);
    setCurrentImage(scan.imageUrl);
    setCurrentImageName(scan.imageName || scan.layer1.primaryDisease);
    setSelectedScanId(scan.id);
    setAppState('DIAGNOSIS');
  };

  const handleNewScan = () => {
    setAppState('IDLE');
  };

  // "Demo Cycle" button handler to easily cycle through the 3 states
  const handleDemoCycle = () => {
    if (appState === 'IDLE') {
      const sample = BENCHMARK_SAMPLES[0]; // Early Blight (Moderate)
      setCurrentImage(sample.thumbnail);
      setCurrentImageName(sample.title);
      setCurrentResult(sample.precomputedResult);
      setSelectedScanId(sample.precomputedResult.id);
      setAppState('SCANNING');
    } else if (appState === 'SCANNING') {
      setAppState('DIAGNOSIS');
    } else {
      setAppState('IDLE');
    }
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#08080A] text-[#E2E8F0] font-sans flex flex-col select-none">
      {/* Custom Chiaroscuro Cursor & Ambient Light Source */}
      <CustomCursor statusColor={getActiveThemeColor()} />

      {/* Main 2-Column Full-Height Dashboard Container */}
      <div className="flex-1 w-full h-full flex flex-col lg:flex-row overflow-hidden">
        {/* Left Column (25% Width): Sidebar */}
        <Sidebar
          recentScans={recentScans}
          selectedScanId={selectedScanId}
          onSelectScan={handleSelectHistoryScan}
          onNewScan={handleNewScan}
          onOpenAgronomist={() => setIsAgronomistOpen(true)}
          onOpenReport={() => setIsReportOpen(true)}
          currentResult={currentResult}
        />

        {/* Main Canvas (75% Width): Core Interaction Area */}
        <main
          id="main-canvas"
          className="relative flex-1 h-full overflow-y-auto bg-[#08080A] p-4 sm:p-6 lg:p-8 flex flex-col justify-between"
        >
          {/* Subtle Background Radial & Grid Pattern */}
          <div className="absolute inset-0 bg-grid-void opacity-30 pointer-events-none" />

          {/* Top Bar Header with Demo Cycle Tester Button */}
          <div className="relative z-20 w-full flex items-center justify-between pb-4 border-b border-[#5B6987]/15">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-[#828C9E]">STATE:</span>
                <span
                  className={`text-xs font-mono font-bold px-2 py-0.5 rounded-full border ${
                    appState === 'IDLE'
                      ? 'bg-[#2A7FFF]/15 text-[#2A7FFF] border-[#2A7FFF]/40'
                      : appState === 'SCANNING'
                      ? 'bg-[#2A7FFF]/20 text-[#2A7FFF] border-[#2A7FFF]/60 animate-pulse'
                      : 'bg-[#FFB020]/15 text-[#FFB020] border-[#FFB020]/40'
                  }`}
                >
                  {appState === 'IDLE'
                    ? '1. IDLE (UPLOAD)'
                    : appState === 'SCANNING'
                    ? '2. SCANNING (LAYER 1 INFERENCE)'
                    : '3. DIAGNOSIS (4-LAYER HXAI)'}
                </span>
              </div>
            </div>

            {/* Quick Demo Cycle & Action Buttons */}
            <div className="flex items-center gap-2.5">
              <motion.button
                id="demo-cycle-btn"
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleDemoCycle}
                className="px-3.5 py-1.5 rounded-xl bg-[rgba(255,255,255,0.08)] hover:bg-[rgba(255,255,255,0.12)] border border-[#5B6987]/40 hover:border-[#2A7FFF] text-xs font-mono text-[#E2E8F0] flex items-center gap-2 transition-all shadow-sm"
              >
                <Play className="w-3.5 h-3.5 text-[#2A7FFF] fill-[#2A7FFF]" />
                <span>Demo Cycle (Next State)</span>
              </motion.button>
            </div>
          </div>

          {/* Core Interactive States Area */}
          <div className="relative z-10 flex-1 flex items-center justify-center py-6 w-full">
            <AnimatePresence mode="wait">
              {/* STATE 1: IDLE (Upload) */}
              {appState === 'IDLE' && (
                <motion.div
                  key="idle-state"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.35, type: 'spring', damping: 25 }}
                  className="w-full"
                >
                  <IdleDropzone
                    onImageSelected={handleImageSelected}
                    onBenchmarkSelected={handleBenchmarkSelected}
                  />
                </motion.div>
              )}

              {/* STATE 2: SCANNING (Processing - Layer 1) */}
              {appState === 'SCANNING' && (
                <motion.div
                  key="scanning-state"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.05 }}
                  transition={{ duration: 0.35, type: 'spring', damping: 25 }}
                  className="w-full"
                >
                  <ScanningView
                    imageUrl={currentImage}
                    imageName={currentImageName}
                    onComplete={() => setAppState('DIAGNOSIS')}
                  />
                </motion.div>
              )}

              {/* STATE 3: DIAGNOSIS & HXAI REASONING (The 4-Layer Architecture) */}
              {appState === 'DIAGNOSIS' && currentResult && (
                <motion.div
                  key="diagnosis-state"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.45, type: 'spring', damping: 25 }}
                  className="w-full"
                >
                  <DiagnosisResultsView
                    result={currentResult}
                    onSelectSimilarCase={(caseTitle) => {
                      console.log('Selected similar case:', caseTitle);
                    }}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Bottom Minimal Footer Strip */}
          <div className="relative z-10 w-full pt-4 border-t border-[#5B6987]/15 flex flex-wrap items-center justify-between text-[11px] font-mono text-[#828C9E]">
            <div className="flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-[#2A7FFF]" />
              <span>Phytosense Human-Centric Explainable AI (HXAI) System</span>
            </div>
            <div className="flex items-center gap-3">
              <span>Mendeley &amp; PlantVillage Grounded</span>
              <span>•</span>
              <span>EfficientNetV2-M + Grad-CAM++</span>
            </div>
          </div>
        </main>
      </div>

      {/* AI Agronomist Modal */}
      <AgronomistModal
        isOpen={isAgronomistOpen}
        onClose={() => setIsAgronomistOpen(false)}
        currentResult={currentResult}
      />

      {/* Clinical Dossier PDF Export Modal */}
      {currentResult && (
        <ClinicalReportModal
          isOpen={isReportOpen}
          onClose={() => setIsReportOpen(false)}
          result={currentResult}
        />
      )}
    </div>
  );
}
