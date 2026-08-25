'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  DiagnosticResult,
  MorphologicalMarker,
  SeverityTier,
} from '@/types/diagnosis';
import {
  Eye,
  EyeOff,
  Layers,
  Sparkles,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Info,
  Sliders,
  Sprout,
  FlaskConical,
  Scissors,
  CheckSquare,
  Square,
  Crosshair,
  ExternalLink,
  Database,
  ArrowRight,
} from 'lucide-react';

interface DiagnosisResultsViewProps {
  result: DiagnosticResult;
  onSelectSimilarCase?: (caseTitle: string) => void;
}

// Similar cases mock data grounded in Mendeley / PlantVillage dataset
const SIMILAR_CASES = [
  {
    id: 'case-eb-01',
    code: 'PLANTVILLAGE-EB#402',
    disease: 'Early Blight',
    similarity: 94.2,
    stage: 'Moderate (Canopy: 25%)',
    thumbnail:
      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100"><rect width="100" height="100" fill="%23121b14"/><circle cx="50" cy="50" r="38" fill="%2324542c"/><circle cx="45" cy="45" r="14" fill="%2347311d" stroke="%23c4a22b" stroke-width="2"/><circle cx="45" cy="45" r="8" fill="%23302013"/><circle cx="65" cy="65" r="10" fill="%2347311d" stroke="%23c4a22b" stroke-width="1.5"/></svg>',
    matchedFeature: 'Concentric ring edges in lower lamina',
  },
  {
    id: 'case-eb-02',
    code: 'MENDELEY-EB#189',
    disease: 'Early Blight',
    similarity: 88.7,
    stage: 'Mild-Moderate',
    thumbnail:
      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100"><rect width="100" height="100" fill="%23121b14"/><circle cx="50" cy="50" r="38" fill="%23214c28"/><circle cx="52" cy="48" r="16" fill="%23402b19" stroke="%23c4a22b" stroke-width="2"/><circle cx="52" cy="48" r="9" fill="%232e1e11"/></svg>',
    matchedFeature: 'Chlorotic halo phytotoxin margin',
  },
  {
    id: 'case-ts-03',
    code: 'PLANTDOC-TS#092',
    disease: 'Target Spot',
    similarity: 64.1,
    stage: 'Lookalike Differentiator',
    thumbnail:
      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100"><rect width="100" height="100" fill="%23121b14"/><circle cx="50" cy="50" r="38" fill="%231d4523"/><circle cx="40" cy="40" r="8" fill="%232b1f15"/><circle cx="60" cy="55" r="7" fill="%232b1f15"/><circle cx="35" cy="65" r="6" fill="%232b1f15"/></svg>',
    matchedFeature: 'Non-concentric pinpoint necrosis',
  },
];

export const DiagnosisResultsView: React.FC<DiagnosisResultsViewProps> = ({
  result,
  onSelectSimilarCase,
}) => {
  // Grad-CAM Heatmap state
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.7);
  const [colormap, setColormap] = useState<'jet' | 'turbo' | 'inferno' | 'viridis'>('jet');
  const [selectedMarker, setSelectedMarker] = useState<MorphologicalMarker | null>(
    result.layer2.markers[0] || null
  );

  // Accordion toggle states for Layer 3
  const [whyThisOpen, setWhyThisOpen] = useState(true);
  const [whyNotOpen, setWhyNotOpen] = useState(true);

  // Active tab for Layer 4 (Decision Support)
  const [decisionTab, setDecisionTab] = useState<'immediate' | 'organic' | 'chemical'>('immediate');
  const [completedItems, setCompletedItems] = useState<Record<string, boolean>>({});

  // Active modal or selected case
  const [inspectedCase, setInspectedCase] = useState<typeof SIMILAR_CASES[0] | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Interactive Card Glow tracking mouse coordinates
  const handleCardMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const card = e.currentTarget;
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    card.style.setProperty('--mouse-x', `${x}px`);
    card.style.setProperty('--mouse-y', `${y}px`);
  };

  const toggleCheck = (id: string) => {
    setCompletedItems((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Draw Grad-CAM heatmap overlay onto canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = (canvas.width = 600);
    const height = (canvas.height = 600);
    ctx.clearRect(0, 0, width, height);

    if (!showHeatmap || opacity <= 0.01) return;

    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    result.layer2.gradCamPoints.forEach((point) => {
      const px = (point.x / 100) * width;
      const py = (point.y / 100) * height;
      const radius = (point.radius / 100) * width * 1.5;

      const grad = offCtx.createRadialGradient(px, py, 0, px, py, radius);
      grad.addColorStop(0, `rgba(0, 0, 0, ${point.intensity})`);
      grad.addColorStop(0.5, `rgba(0, 0, 0, ${point.intensity * 0.5})`);
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

      offCtx.fillStyle = grad;
      offCtx.beginPath();
      offCtx.arc(px, py, radius, 0, Math.PI * 2);
      offCtx.fill();
    });

    const imgData = offCtx.getImageData(0, 0, width, height);
    const pixels = imgData.data;

    for (let i = 0; i < pixels.length; i += 4) {
      const alpha = pixels[i + 3] / 255;
      if (alpha > 0.02) {
        const val = Math.min(1, alpha * 1.35);
        const rgb = getColormapRGB(val, colormap);
        pixels[i] = rgb.r;
        pixels[i + 1] = rgb.g;
        pixels[i + 2] = rgb.b;
        pixels[i + 3] = Math.floor(alpha * opacity * 255);
      } else {
        pixels[i + 3] = 0;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [result, showHeatmap, opacity, colormap]);

  const getColormapRGB = (t: number, map: string) => {
    if (map === 'inferno') {
      if (t < 0.33) {
        const u = t / 0.33;
        return { r: Math.round(u * 120), g: 0, b: Math.round(u * 140) };
      } else if (t < 0.66) {
        const u = (t - 0.33) / 0.33;
        return { r: Math.round(120 + u * 135), g: Math.round(u * 100), b: Math.round(140 * (1 - u)) };
      } else {
        const u = (t - 0.66) / 0.34;
        return { r: 255, g: Math.round(100 + u * 155), b: Math.round(u * 150) };
      }
    } else if (map === 'turbo') {
      const r = Math.sin(t * Math.PI * 1.5) * 127 + 128;
      const g = Math.sin(t * Math.PI) * 127 + 128;
      const b = Math.cos(t * Math.PI * 1.5) * 127 + 128;
      return { r: Math.round(r), g: Math.round(g), b: Math.round(b) };
    } else if (map === 'viridis') {
      return {
        r: Math.round(70 + t * 180),
        g: Math.round(30 + t * 200),
        b: Math.round(180 * (1 - t * 0.8)),
      };
    } else {
      // Jet
      let r = 0, g = 0, b = 0;
      if (t < 0.125) {
        r = 0; g = 0; b = 0.5 + t * 4;
      } else if (t < 0.375) {
        r = 0; g = (t - 0.125) * 4; b = 1;
      } else if (t < 0.625) {
        r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4;
      } else if (t < 0.875) {
        r = 1; g = 1 - (t - 0.625) * 4; b = 0;
      } else {
        r = 1 - (t - 0.875) * 4 * 0.5; g = 0; b = 0;
      }
      return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
    }
  };

  const isModerate = result.layer1.severity === 'Moderate';
  const isSevere = result.layer1.severity === 'Severe';
  const isHealthy = result.layer1.severity === 'Healthy';

  const severityColor = isHealthy
    ? '#1CEB76'
    : isSevere
    ? '#FF2A4D'
    : isModerate
    ? '#FFB020'
    : '#FDE12D';

  return (
    <div className="relative w-full max-w-7xl mx-auto space-y-6">
      {/* Soft Radial Ambient Glow bleeding 10-15% into the #08080A Void */}
      <div
        className="absolute top-10 left-1/2 -translate-x-1/2 w-[700px] h-[550px] rounded-full blur-[120px] pointer-events-none -z-10 transition-colors duration-1000"
        style={{
          backgroundColor: isModerate ? 'rgba(255, 176, 32, 0.12)' : `${severityColor}1a`,
        }}
      />

      {/* 2-Column Main Canvas Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* ========================================================
            LEFT COLUMN (Visuals & Evidence - Layer 2 & Retrieval)
            ======================================================== */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, type: 'spring', damping: 25 }}
          className="lg:col-span-5 space-y-5"
        >
          {/* LAYER 2 (Visual XAI): Leaf Image + Grad-CAM Heatmap */}
          <div
            id="layer-2-visual-container"
            onMouseMove={handleCardMouseMove}
            className="group relative p-4 sm:p-5 rounded-3xl bg-[#121215] border border-[#5B6987]/30 shadow-2xl overflow-hidden space-y-4"
          >
            {/* Subtle cursor tracking radial light on card */}
            <div
              className="pointer-events-none absolute -inset-px opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-3xl"
              style={{
                background:
                  'radial-gradient(400px circle at var(--mouse-x, 100px) var(--mouse-y, 100px), rgba(255, 176, 32, 0.08), transparent 80%)',
              }}
            />

            {/* Header Badge */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#2A7FFF]/15 text-[#2A7FFF] border border-[#2A7FFF]/30 font-bold">
                  LAYER 2: VISUAL XAI
                </span>
                <span className="text-xs font-semibold text-[#E2E8F0]">
                  Grad-CAM++ Spatial Heatmap
                </span>
              </div>
              <span className="text-[10px] font-mono text-[#828C9E]">
                {result.layer2.targetLayer.split(' ')[0]}
              </span>
            </div>

            {/* Visual Leaf Specimen Frame with Canvas Overlay */}
            <div className="relative w-full aspect-square rounded-2xl overflow-hidden bg-black border border-[#5B6987]/30 group/leaf">
              {/* Raw Specimen Leaf Image */}
              <img
                src={result.imageUrl}
                alt="Tomato leaf diagnosis"
                referrerPolicy="no-referrer"
                className="absolute inset-0 w-full h-full object-cover select-none"
              />

              {/* Heatmap Overlay Canvas */}
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none mix-blend-screen select-none"
              />

              {/* Interactive Morphological Feature Pins */}
              {showHeatmap &&
                result.layer2.markers.map((marker, idx) => {
                  const isSelected = selectedMarker?.id === marker.id;
                  return (
                    <motion.button
                      key={marker.id}
                      id={`pin-marker-${marker.id}`}
                      whileHover={{ scale: 1.15 }}
                      whileTap={{ scale: 0.92 }}
                      onClick={() => setSelectedMarker(marker)}
                      style={{
                        left: `${marker.x}%`,
                        top: `${marker.y}%`,
                      }}
                      className={`absolute -translate-x-1/2 -translate-y-1/2 flex items-center justify-center ${
                        isSelected ? 'z-30 scale-110' : 'z-20'
                      }`}
                    >
                      <span
                        className={`absolute w-7 h-7 rounded-full ${
                          isSelected
                            ? 'bg-[#FFB020]/40 border-2 border-[#FFB020] animate-ping-slow'
                            : 'bg-[#2A7FFF]/30 border border-[#2A7FFF]'
                        }`}
                      />
                      <div
                        className={`w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-mono font-bold shadow-lg ${
                          isSelected
                            ? 'bg-[#FFB020] text-[#08080A] border border-white'
                            : 'bg-[#2A7FFF] text-[#08080A]'
                        }`}
                      >
                        {idx + 1}
                      </div>
                    </motion.button>
                  );
                })}

              {/* Legend bar */}
              {showHeatmap && (
                <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between px-3 py-1 rounded-lg bg-[#08080A]/85 backdrop-blur-md border border-white/10 text-[9px] font-mono text-[#828C9E]">
                  <span className="text-[#2A7FFF]">0.0 Low</span>
                  <div
                    className="flex-1 mx-2.5 h-1.5 rounded-full"
                    style={{
                      background:
                        colormap === 'jet'
                          ? 'linear-gradient(90deg, #00008f, #007fff, #00ffff, #7fff7f, #ffff00, #ff7f00, #ff0000)'
                          : colormap === 'turbo'
                          ? 'linear-gradient(90deg, #30123b, #4145ab, #4675ed, #39a2fc, #1bcfd4, #24eca6, #61fc6c, #a4fc3b, #d1e834, #f3c63a, #fe9b2d, #f36315, #d93806, #b11902, #7a0403)'
                          : colormap === 'viridis'
                          ? 'linear-gradient(90deg, #440154, #3b528b, #21918c, #5ec962, #fde725)'
                          : 'linear-gradient(90deg, #000004, #51127c, #b73779, #fc8961, #fec488)',
                    }}
                  />
                  <span className="text-[#FF2A4D]">1.0 Peak</span>
                </div>
              )}
            </div>

            {/* Frosted Glass Toggle Switch for Grad-CAM */}
            <div className="flex items-center justify-between p-3 rounded-2xl bg-[rgba(255,255,255,0.08)] backdrop-blur-md border border-white/10">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-[#2A7FFF]" />
                <span className="text-xs font-semibold text-[#E2E8F0]">
                  View Grad-CAM Activation Map
                </span>
              </div>

              <motion.button
                id="gradcam-toggle-switch"
                whileTap={{ scale: 0.94 }}
                onClick={() => setShowHeatmap(!showHeatmap)}
                className={`relative w-12 h-6 rounded-full transition-colors duration-300 p-0.5 border ${
                  showHeatmap
                    ? 'bg-[#2A7FFF] border-[#2A7FFF]/80 shadow-md shadow-[#2A7FFF]/30'
                    : 'bg-[#121215] border-[#5B6987]/40'
                }`}
              >
                <motion.div
                  className="w-4 h-4 rounded-full bg-white shadow"
                  animate={{ x: showHeatmap ? 24 : 0 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
              </motion.button>
            </div>

            {/* Heatmap Controls Strip (when active) */}
            {showHeatmap && (
              <div className="space-y-2 pt-1">
                <div className="flex items-center justify-between text-[11px] text-[#828C9E] font-mono">
                  <span>Opacity: {Math.round(opacity * 100)}%</span>
                  <div className="flex items-center gap-1">
                    {(['jet', 'turbo', 'inferno', 'viridis'] as const).map((map) => (
                      <button
                        key={map}
                        onClick={() => setColormap(map)}
                        className={`px-1.5 py-0.5 rounded text-[9px] font-mono capitalize transition-all border ${
                          colormap === map
                            ? 'bg-[#2A7FFF]/20 text-[#2A7FFF] border-[#2A7FFF]/60'
                            : 'bg-[#08080A] text-[#828C9E] border-[#5B6987]/30 hover:border-[#5B6987]'
                        }`}
                      >
                        {map}
                      </button>
                    ))}
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full h-1 bg-[#1e2028] rounded-lg appearance-none cursor-pointer accent-[#2A7FFF]"
                />
              </div>
            )}

            {/* Selected Morphological Feature Callout */}
            {selectedMarker && (
              <div className="p-3 rounded-xl bg-[#08080A]/80 border border-[#FFB020]/30 text-left space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-[#FFB020] flex items-center gap-1.5">
                    <Crosshair className="w-3.5 h-3.5" />
                    {selectedMarker.name}
                  </span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#FFB020]/15 text-[#FFB020]">
                    Saliency: {selectedMarker.significance}
                  </span>
                </div>
                <p className="text-[11px] text-[#828C9E] leading-relaxed">
                  {selectedMarker.description}
                </p>
              </div>
            )}
          </div>

          {/* EVIDENCE GROUNDING: Similar Case Retrieval Strip */}
          <div
            id="evidence-grounding-section"
            onMouseMove={handleCardMouseMove}
            className="group relative p-4 rounded-3xl bg-[#121215] border border-[#5B6987]/30 shadow-xl space-y-3"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-[#2A7FFF]" />
                <span className="text-xs font-bold text-[#E2E8F0] tracking-wide">
                  Similar Case Retrieval (k-NN Latent Match)
                </span>
              </div>
              <span className="text-[10px] font-mono text-[#828C9E]">
                Mendeley / PlantVillage
              </span>
            </div>

            <p className="text-[11px] text-[#828C9E]">
              Top cosine-similarity matches retrieved from our verified tomato leaf pathology repository:
            </p>

            {/* Horizontal Strip of 3 Thumbnail Items */}
            <div className="grid grid-cols-3 gap-2.5">
              {SIMILAR_CASES.map((item) => (
                <motion.button
                  key={item.id}
                  id={`similar-case-${item.id}`}
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.94 }}
                  onClick={() => {
                    setInspectedCase(item);
                    if (onSelectSimilarCase) onSelectSimilarCase(item.disease);
                  }}
                  className="p-2 rounded-xl bg-[#08080A] hover:bg-[rgba(255,255,255,0.06)] border border-[#5B6987]/25 hover:border-[#2A7FFF]/60 text-left transition-all flex flex-col group/case"
                >
                  <div className="relative w-full aspect-square rounded-lg overflow-hidden bg-black/60 mb-1.5 border border-[#5B6987]/20">
                    <img
                      src={item.thumbnail}
                      alt={item.code}
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover group-hover/case:scale-105 transition-transform"
                    />
                    <span className="absolute bottom-1 right-1 px-1 py-0.2 rounded text-[8px] font-mono font-bold bg-[#08080A]/90 text-[#1CEB76] border border-[#1CEB76]/40">
                      {item.similarity}%
                    </span>
                  </div>

                  <span className="text-[10px] font-mono font-bold text-[#E2E8F0] truncate group-hover/case:text-[#2A7FFF]">
                    {item.code.split('-')[1]}
                  </span>
                  <span className="text-[9px] text-[#828C9E] truncate">
                    {item.disease}
                  </span>
                </motion.button>
              ))}
            </div>

            {/* Case Details Drawer if clicked */}
            {inspectedCase && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="p-3 rounded-xl bg-[#08080A] border border-[#2A7FFF]/40 text-xs space-y-1 text-left"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-[#2A7FFF]">
                    {inspectedCase.code}
                  </span>
                  <span className="text-[10px] font-mono text-[#1CEB76]">
                    {inspectedCase.similarity}% Cosine Match
                  </span>
                </div>
                <p className="text-[11px] text-[#E2E8F0]">
                  {inspectedCase.matchedFeature}
                </p>
                <p className="text-[10px] text-[#828C9E] font-mono">
                  Pathology Stage: {inspectedCase.stage}
                </p>
              </motion.div>
            )}
          </div>
        </motion.div>

        {/* ========================================================
            RIGHT COLUMN (Reasoning & Action - The Results Panel)
            ======================================================== */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, type: 'spring', damping: 25, delay: 0.05 }}
          className="lg:col-span-7 space-y-5"
        >
          {/* LAYER 1 (Detection): Calibrated Gauge & Header */}
          <div
            id="layer-1-results-panel"
            onMouseMove={handleCardMouseMove}
            className="group relative p-5 sm:p-6 rounded-3xl bg-[#121215] border border-[#5B6987]/30 shadow-2xl overflow-hidden space-y-4"
          >
            {/* Subtle cursor tracking radial light on card */}
            <div
              className="pointer-events-none absolute -inset-px opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-3xl"
              style={{
                background:
                  'radial-gradient(500px circle at var(--mouse-x, 100px) var(--mouse-y, 100px), rgba(255, 176, 32, 0.09), transparent 80%)',
              }}
            />

            {/* Header Badge */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#5B6987]/20 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#FFB020]/15 text-[#FFB020] border border-[#FFB020]/30 font-bold">
                  LAYER 1: DETECTION
                </span>
                <span className="text-xs text-[#828C9E]">
                  Multi-Class Softmax Posterior
                </span>
              </div>

              {/* Severity Status Tag */}
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FFB020]/15 border border-[#FFB020]/40 text-[#FFB020] text-xs font-mono font-bold">
                <AlertTriangle className="w-3.5 h-3.5 text-[#FFB020]" />
                <span>MODERATE SEVERITY</span>
              </div>
            </div>

            {/* Main Header & Gauge Row */}
            <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 items-center">
              {/* Bold Header + Monospace Subtitle */}
              <div className="sm:col-span-7 space-y-1.5">
                <span className="text-[11px] font-mono text-[#828C9E] uppercase tracking-wider block">
                  Pathological Identification
                </span>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-[#E2E8F0] tracking-tight">
                  {result.layer1.primaryDisease}
                </h1>
                <p className="text-sm font-mono text-[#FFB020] italic">
                  {result.layer1.scientificName}
                </p>
                <p className="text-xs text-[#828C9E] pt-1">
                  Estimated Foliar Canopy Impact: <span className="text-[#E2E8F0] font-medium">{result.layer1.affectedCanopyEstimate}</span>
                </p>
              </div>

              {/* Sleek Gauge Showing Calibrated Confidence in Alert Amber */}
              <div className="sm:col-span-5 flex items-center justify-between sm:justify-end gap-3 p-3.5 rounded-2xl bg-[#08080A]/70 border border-[#5B6987]/30">
                <div className="text-right">
                  <span className="text-[10px] font-mono uppercase text-[#828C9E] block">
                    Confidence Gauge
                  </span>
                  <span className="text-2xl sm:text-3xl font-extrabold font-mono text-[#FFB020]">
                    92.4%
                  </span>
                  <span className="text-[10px] text-[#FFB020]/80 font-mono block">
                    Calibrated Bayesian
                  </span>
                </div>

                {/* Circular Gauge Meter */}
                <div className="relative w-14 h-14 shrink-0 flex items-center justify-center">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                    <path
                      className="text-[#1e2028]"
                      strokeWidth="3.5"
                      stroke="currentColor"
                      fill="none"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path
                      className="text-[#FFB020]"
                      strokeDasharray="92.4, 100"
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      stroke="currentColor"
                      fill="none"
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                  </svg>
                </div>
              </div>
            </div>
          </div>

          {/* LAYER 3 (HXAI Reasoning): Accordion / Stacked Card Layout */}
          <div
            id="layer-3-reasoning-panel"
            onMouseMove={handleCardMouseMove}
            className="group relative p-5 sm:p-6 rounded-3xl bg-[#121215] border border-[#5B6987]/30 shadow-2xl overflow-hidden space-y-4"
          >
            {/* Subtle cursor tracking radial light on card */}
            <div
              className="pointer-events-none absolute -inset-px opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-3xl"
              style={{
                background:
                  'radial-gradient(500px circle at var(--mouse-x, 100px) var(--mouse-y, 100px), rgba(42, 127, 255, 0.08), transparent 80%)',
              }}
            />

            {/* Header Badge */}
            <div className="flex items-center justify-between border-b border-[#5B6987]/20 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#2A7FFF]/15 text-[#2A7FFF] border border-[#2A7FFF]/30 font-bold">
                  LAYER 3: HXAI REASONING
                </span>
                <span className="text-xs font-semibold text-[#E2E8F0]">
                  Natural Language Attribution & Counterfactuals
                </span>
              </div>
            </div>

            <div className="space-y-3">
              {/* Accordion 1: "Why this prediction?" */}
              <div className="rounded-2xl bg-[#08080A]/80 border border-[#5B6987]/30 overflow-hidden">
                <motion.button
                  id="accordion-why-prediction-btn"
                  whileTap={{ scale: 0.99 }}
                  onClick={() => setWhyThisOpen(!whyThisOpen)}
                  className="w-full p-4 flex items-center justify-between text-left hover:bg-[rgba(255,255,255,0.03)] transition-colors"
                >
                  <div className="flex items-center gap-2.5">
                    <div className="w-2 h-2 rounded-full bg-[#1CEB76]" />
                    <span className="text-xs sm:text-sm font-bold text-[#E2E8F0]">
                      Why this prediction?
                    </span>
                  </div>
                  {whyThisOpen ? (
                    <ChevronUp className="w-4 h-4 text-[#828C9E]" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-[#828C9E]" />
                  )}
                </motion.button>

                <AnimatePresence>
                  {whyThisOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="px-4 pb-4 space-y-2.5 text-xs text-[#828C9E] border-t border-[#5B6987]/20 pt-3"
                    >
                      <div className="flex items-start gap-2 text-[#E2E8F0]">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#FFB020] mt-1.5 shrink-0" />
                        <span className="font-semibold text-xs">
                          Detected concentric, target-like rings on lower foliage.
                        </span>
                      </div>
                      <div className="pl-3.5 space-y-1.5 text-[11px] text-[#828C9E]">
                        <p>
                          • Characteristic Alternaria target-board pattern of alternating necrotic ridge lines.
                        </p>
                        <p>
                          • Surrounding diffuse chlorotic halo indicating active phytotoxin (Solanapyrone A) diffusion.
                        </p>
                        <p>
                          • Latent feature mapping confirms high-frequency concentric edge density in feature block 7.
                        </p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Accordion 2: "Why not another disease?" (Counterfactuals) */}
              <div className="rounded-2xl bg-[#08080A]/80 border border-[#5B6987]/30 overflow-hidden">
                <motion.button
                  id="accordion-why-not-btn"
                  whileTap={{ scale: 0.99 }}
                  onClick={() => setWhyNotOpen(!whyNotOpen)}
                  className="w-full p-4 flex items-center justify-between text-left hover:bg-[rgba(255,255,255,0.03)] transition-colors"
                >
                  <div className="flex items-center gap-2.5">
                    <div className="w-2 h-2 rounded-full bg-[#FFB020]" />
                    <span className="text-xs sm:text-sm font-bold text-[#E2E8F0]">
                      Why not another disease?
                    </span>
                  </div>
                  {whyNotOpen ? (
                    <ChevronUp className="w-4 h-4 text-[#828C9E]" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-[#828C9E]" />
                  )}
                </motion.button>

                <AnimatePresence>
                  {whyNotOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="px-4 pb-4 space-y-2.5 text-xs text-[#828C9E] border-t border-[#5B6987]/20 pt-3"
                    >
                      <div className="flex items-start gap-2 text-[#E2E8F0]">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#2A7FFF] mt-1.5 shrink-0" />
                        <span className="font-semibold text-xs">
                          Rejected Septoria Leaf Spot: Lesions lack characteristic gray centers.
                        </span>
                      </div>
                      <div className="pl-3.5 space-y-1.5 text-[11px] text-[#828C9E]">
                        <p>
                          • Septoria spots stay tiny (1.5-3mm) with embedded black speck pycnidia; this specimen exhibits large 12mm concentric lesions.
                        </p>
                        <p>
                          • Rejected Late Blight: Absence of water-soaked tissue collapse and zero white downy sporulation underneath the leaf.
                        </p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>

          {/* LAYER 4 (Decision Support): Tabbed Container using Frosted Glass */}
          <div
            id="layer-4-decision-panel"
            onMouseMove={handleCardMouseMove}
            className="group relative p-5 sm:p-6 rounded-3xl bg-[#121215] border border-[#5B6987]/30 shadow-2xl overflow-hidden space-y-4"
          >
            {/* Subtle cursor tracking radial light on card */}
            <div
              className="pointer-events-none absolute -inset-px opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-3xl"
              style={{
                background:
                  'radial-gradient(500px circle at var(--mouse-x, 100px) var(--mouse-y, 100px), rgba(28, 235, 118, 0.08), transparent 80%)',
              }}
            />

            {/* Header Badge */}
            <div className="flex items-center justify-between border-b border-[#5B6987]/20 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#1CEB76]/15 text-[#1CEB76] border border-[#1CEB76]/30 font-bold">
                  LAYER 4: DECISION SUPPORT
                </span>
                <span className="text-xs font-semibold text-[#E2E8F0]">
                  Agronomic Intervention Protocols
                </span>
              </div>
              <span className="text-[10px] font-mono text-[#828C9E]">
                Reapplication: 7 Days
              </span>
            </div>

            {/* Frosted Glass Tabs */}
            <div className="grid grid-cols-3 gap-2 p-1 rounded-2xl bg-[rgba(255,255,255,0.08)] backdrop-blur-md border border-white/10">
              <motion.button
                id="tab-immediate-btn"
                whileTap={{ scale: 0.96 }}
                onClick={() => setDecisionTab('immediate')}
                className={`py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                  decisionTab === 'immediate'
                    ? 'bg-[#FFB020] text-[#08080A] shadow-md shadow-[#FFB020]/20'
                    : 'text-[#828C9E] hover:text-[#E2E8F0] hover:bg-[rgba(255,255,255,0.04)]'
                }`}
              >
                <Scissors className="w-3.5 h-3.5" />
                <span>Immediate</span>
              </motion.button>

              <motion.button
                id="tab-organic-btn"
                whileTap={{ scale: 0.96 }}
                onClick={() => setDecisionTab('organic')}
                className={`py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                  decisionTab === 'organic'
                    ? 'bg-[#1CEB76] text-[#08080A] shadow-md shadow-[#1CEB76]/20'
                    : 'text-[#828C9E] hover:text-[#E2E8F0] hover:bg-[rgba(255,255,255,0.04)]'
                }`}
              >
                <Sprout className="w-3.5 h-3.5" />
                <span>Organic</span>
              </motion.button>

              <motion.button
                id="tab-chemical-btn"
                whileTap={{ scale: 0.96 }}
                onClick={() => setDecisionTab('chemical')}
                className={`py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                  decisionTab === 'chemical'
                    ? 'bg-[#2A7FFF] text-[#08080A] shadow-md shadow-[#2A7FFF]/20'
                    : 'text-[#828C9E] hover:text-[#E2E8F0] hover:bg-[rgba(255,255,255,0.04)]'
                }`}
              >
                <FlaskConical className="w-3.5 h-3.5" />
                <span>Chemical</span>
              </motion.button>
            </div>

            {/* Tab Content Cards with Interactive Checklists */}
            <div className="space-y-2.5 pt-1">
              {decisionTab === 'immediate' && (
                <div className="space-y-2">
                  <motion.div
                    whileTap={{ scale: 0.98 }}
                    onClick={() => toggleCheck('imm-1')}
                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3 ${
                      completedItems['imm-1']
                        ? 'bg-[#121215]/60 border-[#1CEB76]/40 opacity-70'
                        : 'bg-[#08080A]/80 border-[#5B6987]/30 hover:border-[#FFB020]/60'
                    }`}
                  >
                    {completedItems['imm-1'] ? (
                      <CheckSquare className="w-4 h-4 text-[#1CEB76] shrink-0 mt-0.5" />
                    ) : (
                      <Square className="w-4 h-4 text-[#5B6987] shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-0.5 text-left">
                      <p className={`text-xs font-bold ${completedItems['imm-1'] ? 'line-through text-[#828C9E]' : 'text-[#E2E8F0]'}`}>
                        Prune infected lower leaves.
                      </p>
                      <p className="text-[11px] text-[#828C9E]">
                        Prune all lower mature foliage with &gt;20% lesion coverage during dry weather. Sanitize shears with 70% alcohol.
                      </p>
                    </div>
                  </motion.div>

                  <motion.div
                    whileTap={{ scale: 0.98 }}
                    onClick={() => toggleCheck('imm-2')}
                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3 ${
                      completedItems['imm-2']
                        ? 'bg-[#121215]/60 border-[#1CEB76]/40 opacity-70'
                        : 'bg-[#08080A]/80 border-[#5B6987]/30 hover:border-[#FFB020]/60'
                    }`}
                  >
                    {completedItems['imm-2'] ? (
                      <CheckSquare className="w-4 h-4 text-[#1CEB76] shrink-0 mt-0.5" />
                    ) : (
                      <Square className="w-4 h-4 text-[#5B6987] shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-0.5 text-left">
                      <p className={`text-xs font-bold ${completedItems['imm-2'] ? 'line-through text-[#828C9E]' : 'text-[#E2E8F0]'}`}>
                        Shift immediately to ground-level drip irrigation.
                      </p>
                      <p className="text-[11px] text-[#828C9E]">
                        Halt overhead sprinklers to prevent water-splash dispersal of fungal conidia.
                      </p>
                    </div>
                  </motion.div>
                </div>
              )}

              {decisionTab === 'organic' && (
                <div className="space-y-2">
                  <motion.div
                    whileTap={{ scale: 0.98 }}
                    onClick={() => toggleCheck('org-1')}
                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3 ${
                      completedItems['org-1']
                        ? 'bg-[#121215]/60 border-[#1CEB76]/40 opacity-70'
                        : 'bg-[#08080A]/80 border-[#5B6987]/30 hover:border-[#1CEB76]/60'
                    }`}
                  >
                    {completedItems['org-1'] ? (
                      <CheckSquare className="w-4 h-4 text-[#1CEB76] shrink-0 mt-0.5" />
                    ) : (
                      <Square className="w-4 h-4 text-[#5B6987] shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-0.5 text-left">
                      <p className={`text-xs font-bold ${completedItems['org-1'] ? 'line-through text-[#828C9E]' : 'text-[#E2E8F0]'}`}>
                        Ensure adequate plant spacing for airflow.
                      </p>
                      <p className="text-[11px] text-[#828C9E]">
                        Maintain 24-36 inch spacing and stake vines to speed up foliage drying after dew.
                      </p>
                    </div>
                  </motion.div>

                  <motion.div
                    whileTap={{ scale: 0.98 }}
                    onClick={() => toggleCheck('org-2')}
                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3 ${
                      completedItems['org-2']
                        ? 'bg-[#121215]/60 border-[#1CEB76]/40 opacity-70'
                        : 'bg-[#08080A]/80 border-[#5B6987]/30 hover:border-[#1CEB76]/60'
                    }`}
                  >
                    {completedItems['org-2'] ? (
                      <CheckSquare className="w-4 h-4 text-[#1CEB76] shrink-0 mt-0.5" />
                    ) : (
                      <Square className="w-4 h-4 text-[#5B6987] shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-0.5 text-left">
                      <p className={`text-xs font-bold ${completedItems['org-2'] ? 'line-through text-[#828C9E]' : 'text-[#E2E8F0]'}`}>
                        Apply OMRI-listed Copper Octanoate or Bacillus subtilis.
                      </p>
                      <p className="text-[11px] text-[#828C9E]">
                        Dosage: 1.0 fl oz/gal (8 mL/L). Spray upper and lower leaf surfaces. 0-day Pre-Harvest Interval.
                      </p>
                    </div>
                  </motion.div>
                </div>
              )}

              {decisionTab === 'chemical' && (
                <div className="space-y-2">
                  <motion.div
                    whileTap={{ scale: 0.98 }}
                    onClick={() => toggleCheck('chem-1')}
                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3 ${
                      completedItems['chem-1']
                        ? 'bg-[#121215]/60 border-[#1CEB76]/40 opacity-70'
                        : 'bg-[#08080A]/80 border-[#5B6987]/30 hover:border-[#2A7FFF]/60'
                    }`}
                  >
                    {completedItems['chem-1'] ? (
                      <CheckSquare className="w-4 h-4 text-[#1CEB76] shrink-0 mt-0.5" />
                    ) : (
                      <Square className="w-4 h-4 text-[#5B6987] shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-0.5 text-left">
                      <p className={`text-xs font-bold ${completedItems['chem-1'] ? 'line-through text-[#828C9E]' : 'text-[#E2E8F0]'}`}>
                        Consider copper-based fungicide application.
                      </p>
                      <p className="text-[11px] text-[#828C9E]">
                        Apply broad-spectrum protectant barrier (Chlorothalonil, FRAC M05 or Fixed Copper) every 7-10 days.
                      </p>
                    </div>
                  </motion.div>

                  <motion.div
                    whileTap={{ scale: 0.98 }}
                    onClick={() => toggleCheck('chem-2')}
                    className={`p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3 ${
                      completedItems['chem-2']
                        ? 'bg-[#121215]/60 border-[#1CEB76]/40 opacity-70'
                        : 'bg-[#08080A]/80 border-[#5B6987]/30 hover:border-[#2A7FFF]/60'
                    }`}
                  >
                    {completedItems['chem-2'] ? (
                      <CheckSquare className="w-4 h-4 text-[#1CEB76] shrink-0 mt-0.5" />
                    ) : (
                      <Square className="w-4 h-4 text-[#5B6987] shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-0.5 text-left">
                      <p className={`text-xs font-bold ${completedItems['chem-2'] ? 'line-through text-[#828C9E]' : 'text-[#E2E8F0]'}`}>
                        Azoxystrobin + Difenoconazole (Quadris Top) Systemic Rotation.
                      </p>
                      <p className="text-[11px] text-[#828C9E]">
                        Dosage: 8.0 fl oz/acre (FRAC Group 11 + 3). PHI: 7 days. Arrests active internal hyphal growth.
                      </p>
                    </div>
                  </motion.div>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};
