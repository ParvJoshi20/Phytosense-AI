'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Layer2VisualExplainabilityData, MorphologicalMarker } from '@/types/diagnosis';
import { Eye, Layers, Sliders, Info, Zap, Crosshair, HelpCircle } from 'lucide-react';

interface Layer2GradCamProps {
  data: Layer2VisualExplainabilityData;
  imageUrl: string;
}

export const Layer2GradCam: React.FC<Layer2GradCamProps> = ({ data, imageUrl }) => {
  const [opacity, setOpacity] = useState<number>(0.65);
  const [colormap, setColormap] = useState<'jet' | 'turbo' | 'inferno' | 'viridis'>('jet');
  const [showMarkers, setShowMarkers] = useState<boolean>(true);
  const [selectedMarker, setSelectedMarker] = useState<MorphologicalMarker | null>(
    data.markers[0] || null
  );

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Re-draw Grad-CAM heatmap on canvas whenever opacity, colormap, or points change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = (canvas.width = 600);
    const height = (canvas.height = 600);

    // Clear previous
    ctx.clearRect(0, 0, width, height);

    if (opacity <= 0.01) return;

    // Create temporary offscreen buffer for density calculation
    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    // Render radial gradients for each Grad-CAM activation point
    data.gradCamPoints.forEach((point) => {
      const px = (point.x / 100) * width;
      const py = (point.y / 100) * height;
      const radius = (point.radius / 100) * width * 1.6;

      const radGrad = offCtx.createRadialGradient(px, py, 0, px, py, radius);
      radGrad.addColorStop(0, `rgba(0, 0, 0, ${point.intensity})`);
      radGrad.addColorStop(0.5, `rgba(0, 0, 0, ${point.intensity * 0.5})`);
      radGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');

      offCtx.fillStyle = radGrad;
      offCtx.beginPath();
      offCtx.arc(px, py, radius, 0, Math.PI * 2);
      offCtx.fill();
    });

    const imgData = offCtx.getImageData(0, 0, width, height);
    const pixels = imgData.data;

    // Colorize each pixel based on selected colormap
    for (let i = 0; i < pixels.length; i += 4) {
      const alpha = pixels[i + 3] / 255;
      if (alpha > 0.02) {
        const value = Math.min(1, alpha * 1.3);
        const rgb = getColormapRGB(value, colormap);

        pixels[i] = rgb.r;
        pixels[i + 1] = rgb.g;
        pixels[i + 2] = rgb.b;
        pixels[i + 3] = Math.floor(alpha * opacity * 255);
      } else {
        pixels[i + 3] = 0;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [data, opacity, colormap]);

  // Colormap RGB interpolation
  const getColormapRGB = (t: number, map: string) => {
    if (map === 'inferno') {
      // Inferno: black -> purple -> orange -> yellow
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
      // Turbo rainbow
      const r = Math.sin(t * Math.PI * 1.5) * 127 + 128;
      const g = Math.sin(t * Math.PI) * 127 + 128;
      const b = Math.cos(t * Math.PI * 1.5) * 127 + 128;
      return { r: Math.round(r), g: Math.round(g), b: Math.round(b) };
    } else if (map === 'viridis') {
      // Viridis: purple -> blue -> teal -> yellow
      return {
        r: Math.round(70 + t * 180),
        g: Math.round(30 + t * 200),
        b: Math.round(180 * (1 - t * 0.8)),
      };
    } else {
      // Standard Jet colormap: Blue -> Cyan -> Yellow -> Red
      let r = 0,
        g = 0,
        b = 0;
      if (t < 0.125) {
        r = 0;
        g = 0;
        b = 0.5 + t * 4;
      } else if (t < 0.375) {
        r = 0;
        g = (t - 0.125) * 4;
        b = 1;
      } else if (t < 0.625) {
        r = (t - 0.375) * 4;
        g = 1;
        b = 1 - (t - 0.375) * 4;
      } else if (t < 0.875) {
        r = 1;
        g = 1 - (t - 0.625) * 4;
        b = 0;
      } else {
        r = 1 - (t - 0.875) * 4 * 0.5;
        g = 0;
        b = 0;
      }
      return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
    }
  };

  return (
    <div className="rounded-2xl border border-emerald-950/80 bg-[#0d1611]/90 p-5 sm:p-6 shadow-xl space-y-6">
      {/* Layer Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-emerald-950/80 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-emerald-950 border border-emerald-600/40 text-emerald-400 font-mono text-xs font-bold">
            L2
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-white tracking-wide flex items-center gap-2">
              Layer 2: Visual Explainability (Grad-CAM++ Heatmap)
            </h3>
            <p className="text-[11px] text-slate-400">
              Interactive class-activation mapping highlighting morphological feature saliency
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-black/40 px-3 py-1 rounded-lg border border-emerald-950">
          <Layers className="w-3.5 h-3.5 text-emerald-400" />
          <span>Layer: {data.targetLayer}</span>
        </div>
      </div>

      {/* Main Interactive Stage */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Visual Heatmap Canvas Container */}
        <div className="lg:col-span-7 flex flex-col items-center">
          <div className="relative w-full aspect-square max-w-[480px] rounded-2xl overflow-hidden bg-black border-2 border-emerald-950 shadow-2xl group">
            {/* Base Specimen Leaf Image */}
            <img
              src={imageUrl}
              alt="Leaf specimen"
              referrerPolicy="no-referrer"
              className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none"
            />

            {/* Grad-CAM Heatmap Canvas Overlay */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none select-none mix-blend-screen"
            />

            {/* Interactive Morphological Feature Markers */}
            {showMarkers &&
              data.markers.map((marker, idx) => {
                const isSelected = selectedMarker?.id === marker.id;
                return (
                  <button
                    key={marker.id}
                    id={`hotspot-marker-${marker.id}`}
                    onClick={() => setSelectedMarker(marker)}
                    style={{
                      left: `${marker.x}%`,
                      top: `${marker.y}%`,
                    }}
                    className={`absolute -translate-x-1/2 -translate-y-1/2 flex items-center justify-center transition-transform ${
                      isSelected ? 'scale-125 z-30' : 'hover:scale-110 z-20'
                    }`}
                  >
                    {/* Pulsing ring */}
                    <span
                      className={`absolute w-8 h-8 rounded-full border-2 ${
                        isSelected
                          ? 'border-emerald-400 bg-emerald-400/30 animate-ping-slow'
                          : 'border-yellow-400/80 bg-yellow-400/20'
                      }`}
                    />
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-mono font-bold shadow-lg ${
                        isSelected
                          ? 'bg-emerald-400 text-emerald-950 border border-white'
                          : 'bg-yellow-400 text-yellow-950 border border-yellow-200'
                      }`}
                    >
                      {idx + 1}
                    </div>
                  </button>
                );
              })}

            {/* Color Gradient Scale Legend */}
            <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between px-3 py-1.5 rounded-xl bg-black/80 backdrop-blur-md border border-white/10 text-[10px] font-mono text-slate-300">
              <span className="text-blue-400 font-semibold">0.0 Low</span>
              <div
                className="flex-1 mx-3 h-2 rounded-full border border-white/20"
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
              <span className="text-rose-400 font-semibold">1.0 Peak</span>
            </div>
          </div>

          {/* Saliency Focus Summary */}
          <p className="mt-3 text-xs text-slate-400 text-center max-w-md italic">
            &quot;{data.heatmapDescription}&quot;
          </p>
        </div>

        {/* Controls & Morphological Marker Details */}
        <div className="lg:col-span-5 space-y-4">
          {/* Overlay Controls Panel */}
          <div className="p-4 rounded-xl bg-black/40 border border-emerald-950 space-y-3.5">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-200">
              <div className="flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-emerald-400" />
                <span>Heatmap Blend Controls</span>
              </div>
              <span className="font-mono text-emerald-400">
                {Math.round(opacity * 100)}% Opacity
              </span>
            </div>

            {/* Opacity Slider */}
            <div>
              <input
                id="heatmap-opacity-slider"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
              <div className="flex justify-between text-[9px] font-mono text-slate-500 mt-1">
                <span>0% (Raw Leaf)</span>
                <span>50%</span>
                <span>100% (Pure Activation)</span>
              </div>
            </div>

            {/* Colormap Selector */}
            <div className="space-y-1.5 pt-1">
              <span className="text-[11px] text-slate-400 block font-mono">
                Thermal Colormap
              </span>
              <div className="grid grid-cols-4 gap-1.5">
                {(['jet', 'turbo', 'inferno', 'viridis'] as const).map((map) => (
                  <button
                    key={map}
                    onClick={() => setColormap(map)}
                    className={`px-2 py-1 rounded-lg text-[11px] font-mono font-medium capitalize transition-all border ${
                      colormap === map
                        ? 'bg-emerald-950 text-emerald-300 border-emerald-500'
                        : 'bg-slate-900/80 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {map}
                  </button>
                ))}
              </div>
            </div>

            {/* Toggle Markers */}
            <div className="flex items-center justify-between pt-1 border-t border-emerald-950/60">
              <span className="text-xs text-slate-300">Feature Hotspots</span>
              <button
                id="toggle-markers-btn"
                onClick={() => setShowMarkers(!showMarkers)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-mono font-semibold transition-all border ${
                  showMarkers
                    ? 'bg-emerald-950 text-emerald-300 border-emerald-600/60'
                    : 'bg-slate-900 text-slate-500 border-slate-800'
                }`}
              >
                {showMarkers ? 'Markers ON' : 'Markers OFF'}
              </button>
            </div>
          </div>

          {/* Morphological Feature Inspector Card */}
          <div className="p-4 rounded-xl bg-black/40 border border-emerald-950 space-y-2.5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
                <Crosshair className="w-3.5 h-3.5 text-yellow-400" />
                <span>Morphological Feature Inspector</span>
              </div>
              <span className="text-[10px] font-mono text-slate-400">
                Click pins on leaf
              </span>
            </div>

            {selectedMarker ? (
              <div className="space-y-2 p-3 rounded-lg bg-emerald-950/30 border border-emerald-800/40 animate-fadeIn">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-emerald-300">
                    {selectedMarker.name}
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-900/60 text-emerald-200 border border-emerald-700/50">
                    Saliency: {selectedMarker.significance}
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {selectedMarker.description}
                </p>
                <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400 pt-1">
                  <span>Type: {selectedMarker.featureType.replace('_', ' ')}</span>
                  <span>•</span>
                  <span>Spatial Coord: ({Math.round(selectedMarker.x)}%, {Math.round(selectedMarker.y)}%)</span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic p-3 text-center">
                Click any numbered pin on the leaf image to inspect the morphological symptom marker.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
