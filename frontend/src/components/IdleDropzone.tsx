'use client';

import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, Camera, Database, Image as ImageIcon, AlertCircle } from 'lucide-react';
import { BENCHMARK_SAMPLES } from '@/data/benchmarkDataset';
import { BenchmarkSample } from '@/types/diagnosis';

interface IdleDropzoneProps {
  onImageSelected: (base64: string, name?: string) => void;
  onBenchmarkSelected: (sample: BenchmarkSample) => void;
}

export const IdleDropzone: React.FC<IdleDropzoneProps> = ({
  onImageSelected,
  onBenchmarkSelected,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const validExtensions = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];

  const validateAndProcessFile = (file: File) => {
    if (!validExtensions.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp)$/i)) {
      alert('Invalid file format. Please upload an image (.jpg, .jpeg, .png, .webp).');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      onImageSelected(base64, file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  // Camera handling
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraActive(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setCameraError('Camera unavailable or permission denied.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        stopCamera();
        onImageSelected(dataUrl, 'Live Camera Specimen');
      }
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col items-center justify-center space-y-6">
      {/* Central Drag-and-Drop Zone */}
      {cameraActive ? (
        <div className="relative w-full max-w-xl aspect-video rounded-3xl overflow-hidden bg-black border-2 border-[#2A7FFF] shadow-2xl shadow-[#2A7FFF]/20">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 border border-[#2A7FFF]/40 flex items-center justify-center pointer-events-none">
            <div className="w-56 h-56 border border-dashed border-[#2A7FFF] rounded-2xl flex items-center justify-center">
              <span className="text-[11px] font-mono text-[#2A7FFF] bg-[#08080A]/90 px-3 py-1 rounded-full">
                Align Leaf Symptom
              </span>
            </div>
          </div>
          <div className="absolute bottom-4 left-0 right-0 flex items-center justify-center gap-3">
            <motion.button
              whileTap={{ scale: 0.94 }}
              onClick={capturePhoto}
              className="px-6 py-2.5 rounded-full bg-[#2A7FFF] text-[#08080A] font-bold text-xs flex items-center gap-2 shadow-lg shadow-[#2A7FFF]/40"
            >
              <Camera className="w-4 h-4" />
              Capture Specimen
            </motion.button>
            <motion.button
              whileTap={{ scale: 0.94 }}
              onClick={stopCamera}
              className="px-4 py-2.5 rounded-full bg-[#121215] text-[#E2E8F0] border border-[#5B6987]/40 text-xs"
            >
              Cancel
            </motion.button>
          </div>
        </div>
      ) : (
        <motion.div
          id="massive-idle-dropzone"
          whileHover={{ scale: 0.98 }}
          transition={{ type: 'spring', stiffness: 350, damping: 25 }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative w-full min-h-[380px] sm:min-h-[420px] rounded-3xl border-2 border-dashed transition-all duration-300 p-8 sm:p-12 flex flex-col items-center justify-center text-center cursor-pointer select-none bg-[#121215]/80 overflow-hidden ${
            isDragOver
              ? 'border-[#2A7FFF] bg-[#2A7FFF]/10 shadow-2xl shadow-[#2A7FFF]/20 scale-[0.98]'
              : 'border-[#5B6987] hover:border-[#2A7FFF] animate-border-pulse'
          }`}
        >
          {/* Subtle Background Grid */}
          <div className="absolute inset-0 bg-grid-void opacity-40 pointer-events-none" />

          {/* Icon Circle */}
          <div className="relative z-10 mb-6">
            <div className="w-20 h-20 rounded-2xl bg-[#121215] border border-[#5B6987]/40 flex items-center justify-center text-[#2A7FFF] shadow-xl group-hover:scale-105 transition-transform">
              <Upload className="w-9 h-9 stroke-[1.75]" />
            </div>
            <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-[#2A7FFF] animate-ping" />
          </div>

          {/* Primary Text */}
          <h2 className="relative z-10 text-xl sm:text-2xl font-bold text-[#E2E8F0] tracking-tight mb-2">
            Tomato Leaf HXAI Diagnostic Station
          </h2>

          <p className="relative z-10 text-sm text-[#828C9E] max-w-md mb-6 leading-relaxed">
            Drag and drop a tomato leaf image for HXAI analysis.
          </p>

          {/* Upload Button & Action Controls */}
          <div
            className="relative z-10 flex flex-wrap items-center justify-center gap-3"
            onClick={(e) => e.stopPropagation()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={handleFileChange}
            />

            <motion.button
              id="idle-manual-upload-btn"
              whileTap={{ scale: 0.96 }}
              onClick={() => fileInputRef.current?.click()}
              className="px-5 py-2.5 rounded-xl bg-[#2A7FFF] hover:bg-[#2A7FFF]/90 text-[#08080A] font-semibold text-xs transition-all shadow-lg shadow-[#2A7FFF]/25 flex items-center gap-2"
            >
              <Upload className="w-4 h-4 stroke-[2.2]" />
              <span>Browse Image File</span>
            </motion.button>

            <motion.button
              id="idle-camera-btn"
              whileTap={{ scale: 0.96 }}
              onClick={startCamera}
              className="px-5 py-2.5 rounded-xl bg-[#121215] hover:bg-[rgba(255,255,255,0.08)] border border-[#5B6987]/40 hover:border-[#2A7FFF]/50 text-[#E2E8F0] text-xs font-medium transition-all flex items-center gap-2"
            >
              <Camera className="w-4 h-4 text-[#2A7FFF]" />
              <span>Use Camera</span>
            </motion.button>
          </div>

          {/* Accepted Formats pill */}
          <span className="relative z-10 text-[10px] font-mono text-[#5B6987] mt-5">
            Supported: .JPG, .JPEG, .PNG, .WEBP (Max 25MB)
          </span>

          {cameraError && (
            <div className="relative z-10 mt-3 flex items-center gap-1.5 text-xs text-[#FFB020] bg-[#FFB020]/10 px-3 py-1 rounded-lg border border-[#FFB020]/30">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>{cameraError}</span>
            </div>
          )}
        </motion.div>
      )}

      {/* Benchmark Datasets Quick Selector Strip */}
      <div className="w-full space-y-2.5">
        <div className="flex items-center justify-between px-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#828C9E] uppercase tracking-wider">
            <Database className="w-3.5 h-3.5 text-[#2A7FFF]" />
            <span>Or test with benchmark dataset cases (Mendeley / PlantVillage):</span>
          </div>
          <span className="text-[10px] font-mono text-[#5B6987]">1-Click Load</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {BENCHMARK_SAMPLES.slice(0, 4).map((sample) => (
            <motion.button
              key={sample.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => onBenchmarkSelected(sample)}
              className="p-3 rounded-2xl bg-[#121215] hover:bg-[rgba(255,255,255,0.06)] border border-[#5B6987]/25 hover:border-[#2A7FFF]/60 transition-all text-left flex flex-col group"
            >
              <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-black/50 mb-2 border border-[#5B6987]/20">
                <img
                  src={sample.thumbnail}
                  alt={sample.title}
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                />
                <span
                  className={`absolute top-1 right-1 px-1.5 py-0.5 rounded text-[8px] font-mono font-bold ${
                    sample.severity === 'Healthy'
                      ? 'bg-[#1CEB76] text-[#08080A]'
                      : sample.severity === 'Severe'
                      ? 'bg-[#FF2A4D] text-white'
                      : sample.severity === 'Moderate'
                      ? 'bg-[#FFB020] text-[#08080A]'
                      : 'bg-[#FDE12D] text-[#08080A]'
                  }`}
                >
                  {sample.severity}
                </span>
              </div>
              <span className="text-xs font-semibold text-[#E2E8F0] group-hover:text-[#2A7FFF] truncate transition-colors">
                {sample.diseaseCategory}
              </span>
              <span className="text-[10px] text-[#828C9E] line-clamp-1 italic font-mono">
                {sample.title.includes('(') ? sample.title.split('(')[1].replace(')', '') : sample.title}
              </span>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
};
