'use client';

import React, { useState, useRef, useCallback } from 'react';
import { Upload, Camera, Sparkles, RefreshCw, X, AlertCircle, CheckCircle2, ShieldAlert } from 'lucide-react';
import { BENCHMARK_SAMPLES } from '@/data/benchmarkDataset';
import { BenchmarkSample } from '@/types/diagnosis';

interface ImageDropzoneProps {
  onImageSelected: (base64: string, name?: string) => void;
  onBenchmarkSelected: (sample: BenchmarkSample) => void;
  isAnalyzing: boolean;
}

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({
  onImageSelected,
  onBenchmarkSelected,
  isAnalyzing,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // File Upload Handlers
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPEG, PNG, WEBP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result as string;
      setPreviewImage(base64);
      onImageSelected(base64, file.name);
    };
    reader.readAsDataURL(file);
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
      processFile(file);
    }
  };

  // Camera Handling
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
      setCameraError('Camera permission denied or camera device not available.');
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  const capturePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        setPreviewImage(dataUrl);
        stopCamera();
        onImageSelected(dataUrl, 'Camera Capture Specimen');
      }
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Primary Capture / Dropzone Box */}
      <div
        id="image-dropzone-container"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative overflow-hidden rounded-2xl border-2 transition-all duration-300 ${
          isDragOver
            ? 'border-emerald-400 bg-emerald-950/40 shadow-lg shadow-emerald-500/20'
            : 'border-dashed border-emerald-900/60 hover:border-emerald-600/60 bg-[#0d1611]/80'
        } p-6 sm:p-8 flex flex-col items-center justify-center min-h-[260px] text-center`}
      >
        {/* Subtle Background Grid */}
        <div className="absolute inset-0 bg-grid-pattern opacity-30 pointer-events-none" />

        {/* Live Laser Sweep during Analysis */}
        {isAnalyzing && <div className="scan-laser z-20" />}

        {/* Camera Feed Mode */}
        {cameraActive ? (
          <div className="relative w-full max-w-md aspect-video rounded-xl overflow-hidden bg-black border border-emerald-500/50 shadow-lg">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            {/* Camera Viewfinder Overlay */}
            <div className="absolute inset-0 border-2 border-emerald-500/30 pointer-events-none flex items-center justify-center">
              <div className="w-48 h-48 border border-emerald-400/60 rounded-xl flex items-center justify-center">
                <span className="text-[10px] font-mono text-emerald-300 bg-emerald-950/80 px-2 py-0.5 rounded">
                  Center Leaf in Frame
                </span>
              </div>
            </div>

            {/* Camera Controls */}
            <div className="absolute bottom-3 left-0 right-0 flex items-center justify-center gap-4">
              <button
                type="button"
                onClick={capturePhoto}
                className="px-5 py-2 rounded-full bg-emerald-500 hover:bg-emerald-400 text-emerald-950 font-bold text-xs flex items-center gap-2 shadow-lg shadow-emerald-500/30 transition-transform active:scale-95"
              >
                <Camera className="w-4 h-4" />
                Capture Leaf
              </button>
              <button
                type="button"
                onClick={stopCamera}
                className="p-2 rounded-full bg-slate-900/80 text-slate-300 hover:text-white border border-slate-700"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        ) : (
          /* Normal Upload / Drop State */
          <div className="relative z-10 flex flex-col items-center max-w-lg">
            <div className="relative mb-4">
              <div className="w-16 h-16 rounded-2xl bg-emerald-950/80 border border-emerald-500/30 flex items-center justify-center text-emerald-400 glow-emerald">
                {isAnalyzing ? (
                  <RefreshCw className="w-8 h-8 animate-spin text-emerald-400" />
                ) : (
                  <Upload className="w-8 h-8" />
                )}
              </div>
              {isAnalyzing && (
                <span className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 animate-ping" />
              )}
            </div>

            <h2 className="text-base sm:text-lg font-semibold text-white mb-1">
              {isAnalyzing ? (
                <span className="text-emerald-400 font-mono">
                  Executing 4-Layer HXAI Inference Pipeline...
                </span>
              ) : (
                'Upload Tomato Leaf Specimen'
              )}
            </h2>
            <p className="text-xs text-slate-400 mb-5 max-w-sm">
              Drag and drop an image of a symptomatic or healthy tomato leaf, snap a live photo, or test with benchmark samples below.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-3">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={handleFileChange}
              />
              <button
                id="btn-upload-file"
                type="button"
                disabled={isAnalyzing}
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-emerald-950 font-semibold text-xs transition-all shadow-md shadow-emerald-500/20 flex items-center gap-2"
              >
                <Upload className="w-4 h-4" />
                Browse File
              </button>

              <button
                id="btn-open-camera"
                type="button"
                disabled={isAnalyzing}
                onClick={startCamera}
                className="px-4 py-2 rounded-xl bg-slate-900/90 hover:bg-slate-800 disabled:opacity-50 border border-slate-700/80 hover:border-slate-600 text-slate-200 text-xs font-medium transition-all flex items-center gap-2"
              >
                <Camera className="w-4 h-4 text-emerald-400" />
                Live Camera
              </button>
            </div>

            {cameraError && (
              <div className="mt-3 flex items-center gap-2 text-[11px] text-amber-400 bg-amber-950/50 border border-amber-800/40 px-3 py-1 rounded-lg">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                <span>{cameraError}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Curated Benchmark Dataset Samples Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-semibold text-slate-200 tracking-wide uppercase">
              Benchmark Clinical Reference Samples (PlantVillage / PlantDoc)
            </span>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">1-Click Diagnosis</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
          {BENCHMARK_SAMPLES.map((sample) => {
            const isSeveritySevere = sample.severity === 'Severe';
            const isSeverityModerate = sample.severity === 'Moderate';
            const isSeverityHealthy = sample.severity === 'Healthy';

            return (
              <button
                key={sample.id}
                id={`sample-card-${sample.id}`}
                disabled={isAnalyzing}
                onClick={() => {
                  setPreviewImage(sample.thumbnail);
                  onBenchmarkSelected(sample);
                }}
                className="group relative flex flex-col items-start p-2.5 rounded-xl bg-[#0c140f] hover:bg-[#132219] border border-emerald-950 hover:border-emerald-500/50 transition-all text-left overflow-hidden hover:scale-[1.02]"
              >
                {/* Visual Thumbnail */}
                <div className="relative w-full aspect-square rounded-lg overflow-hidden bg-black/40 mb-2 border border-emerald-950/60 flex items-center justify-center">
                  <img
                    src={sample.thumbnail}
                    alt={sample.title}
                    referrerPolicy="no-referrer"
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  />
                  {/* Severity Badge */}
                  <span
                    className={`absolute top-1.5 right-1.5 px-1.5 py-0.5 rounded text-[9px] font-mono font-bold ${
                      isSeverityHealthy
                        ? 'bg-emerald-500/90 text-emerald-950'
                        : isSeveritySevere
                        ? 'bg-rose-500/90 text-white'
                        : isSeverityModerate
                        ? 'bg-amber-500/90 text-amber-950'
                        : 'bg-yellow-500/90 text-yellow-950'
                    }`}
                  >
                    {sample.severity}
                  </span>
                </div>

                <span className="text-xs font-semibold text-white truncate w-full group-hover:text-emerald-300 transition-colors">
                  {sample.diseaseCategory}
                </span>
                <span className="text-[10px] text-slate-400 line-clamp-1">
                  {sample.title.includes('(') ? sample.title.split('(')[1].replace(')', '') : sample.title}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
