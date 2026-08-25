'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, Layers, Sparkles } from 'lucide-react';

interface ScanningViewProps {
  imageUrl: string;
  imageName?: string;
  onComplete?: () => void;
}

export const ScanningView: React.FC<ScanningViewProps> = ({
  imageUrl,
  imageName = 'Uploaded Tomato Specimen',
  onComplete,
}) => {
  const [progress, setProgress] = useState(15);
  const [stepText, setStepText] = useState('Initializing spatial feature tensor...');

  useEffect(() => {
    const timer1 = setTimeout(() => {
      setProgress(42);
      setStepText('Extracting latent morphological features (EfficientNetV2-M)...');
    }, 600);

    const timer2 = setTimeout(() => {
      setProgress(78);
      setStepText('Computing Grad-CAM++ gradients and counterfactuals...');
    }, 1300);

    const timer3 = setTimeout(() => {
      setProgress(98);
      setStepText('Calibrating Bayesian confidence and assembling decision support...');
    }, 2000);

    const timer4 = setTimeout(() => {
      if (onComplete) {
        onComplete();
      }
    }, 2500);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, [onComplete]);

  return (
    <div className="relative w-full max-w-5xl mx-auto flex flex-col items-center justify-center p-4 sm:p-8">
      {/* Soft Radial Scanner Blue Ambient Glow behind image */}
      <div className="absolute w-[460px] sm:w-[580px] h-[460px] sm:h-[580px] rounded-full bg-[#2A7FFF]/15 blur-[90px] pointer-events-none -z-10 animate-pulse" />

      <div className="w-full grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
        {/* Left: High-resolution Leaf Specimen with Scanner Blue Laser Sweep */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4 }}
          className="md:col-span-6 flex flex-col items-center"
        >
          <div className="relative w-full aspect-square max-w-[380px] rounded-3xl overflow-hidden bg-[#121215] border-2 border-[#2A7FFF]/60 shadow-2xl shadow-[#2A7FFF]/25">
            {/* Specimen image uploaded by user */}
            <img
              src={imageUrl}
              alt={imageName}
              referrerPolicy="no-referrer"
              className="w-full h-full object-cover"
            />

            {/* Viewfinder Target HUD Elements */}
            <div className="absolute inset-0 border border-[#2A7FFF]/30 pointer-events-none flex flex-col justify-between p-4">
              <div className="flex justify-between items-start">
                <span className="text-[10px] font-mono text-[#2A7FFF] bg-[#08080A]/85 px-2 py-0.5 rounded border border-[#2A7FFF]/40">
                  FOV: 512x512 RES
                </span>
                <span className="flex items-center gap-1 text-[10px] font-mono text-[#2A7FFF] bg-[#08080A]/85 px-2 py-0.5 rounded border border-[#2A7FFF]/40">
                  <span className="w-2 h-2 rounded-full bg-[#2A7FFF] animate-ping" />
                  SCANNING
                </span>
              </div>
              <div className="flex justify-between items-end text-[9px] font-mono text-[#5B6987]">
                <span>LATENT: 1280-D</span>
                <span>SPECTRAL: L*A*B*</span>
              </div>
            </div>

            {/* Horizontal Scanner Blue Laser Line Sweeping Top to Bottom */}
            <div className="scanner-laser z-20" />
          </div>

          <p className="text-xs font-mono text-[#828C9E] mt-3 truncate max-w-xs text-center">
            {imageName}
          </p>
        </motion.div>

        {/* Right: 3 Pulsing Skeleton Loader Blocks */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="md:col-span-6 space-y-4"
        >
          {/* Skeleton Block 1: Layer 1 Detection Preview */}
          <div className="p-4 rounded-2xl bg-[#121215] border border-[#5B6987]/30 space-y-2.5 animate-pulse">
            <div className="flex items-center justify-between">
              <div className="h-4 w-32 bg-[#5B6987]/30 rounded" />
              <div className="h-4 w-12 bg-[#2A7FFF]/30 rounded-full" />
            </div>
            <div className="h-7 w-48 bg-[#5B6987]/20 rounded" />
            <div className="h-2 w-full bg-[#5B6987]/20 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#2A7FFF] rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Skeleton Block 2: Layer 2 & 3 Reasoning Preview */}
          <div className="p-4 rounded-2xl bg-[#121215] border border-[#5B6987]/30 space-y-2.5 animate-pulse">
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 rounded-md bg-[#2A7FFF]/20" />
              <div className="h-4 w-40 bg-[#5B6987]/30 rounded" />
            </div>
            <div className="space-y-1.5 pt-1">
              <div className="h-3 w-full bg-[#5B6987]/15 rounded" />
              <div className="h-3 w-5/6 bg-[#5B6987]/15 rounded" />
            </div>
          </div>

          {/* Skeleton Block 3: Layer 4 Decision Support Preview */}
          <div className="p-4 rounded-2xl bg-[#121215] border border-[#5B6987]/30 space-y-2.5 animate-pulse">
            <div className="flex items-center justify-between">
              <div className="h-4 w-36 bg-[#5B6987]/30 rounded" />
              <div className="h-3 w-16 bg-[#5B6987]/20 rounded" />
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div className="h-8 bg-[#5B6987]/20 rounded-lg" />
              <div className="h-8 bg-[#5B6987]/20 rounded-lg" />
              <div className="h-8 bg-[#5B6987]/20 rounded-lg" />
            </div>
          </div>

          {/* Text below loaders as required */}
          <div className="pt-2 text-center sm:text-left space-y-1">
            <div className="flex items-center gap-2 justify-center sm:justify-start text-xs font-mono text-[#2A7FFF]">
              <Cpu className="w-4 h-4 animate-spin" />
              <span className="font-semibold">{stepText}</span>
            </div>
            <p className="text-[11px] text-[#828C9E] font-mono">
              Extracting latent features... calibrating confidence...
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};
