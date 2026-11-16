'use client';

import React, { useState } from 'react';

interface ScanViewerProps {
  gradcamSlicePngB64?: string;
  seriesuid?: string;
  noduleCoord?: [number, number, number];
  diameterMm?: number;
  grade?: string;
  confidence?: number;
}

export default function ScanViewer({
  gradcamSlicePngB64,
  seriesuid,
  noduleCoord,
  diameterMm,
  grade,
  confidence,
}: ScanViewerProps) {
  const [showGradCam, setShowGradCam] = useState<boolean>(true);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  return (
    <div className="card-surface overflow-hidden flex flex-col bg-gray-900 border-gray-800 text-gray-200">
      {/* Top Radiological Toolbar */}
      <div className="bg-gray-950 px-4 py-2.5 border-b border-gray-800 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span className="font-mono text-gray-300">
            {seriesuid ? `AXIAL CT • ${seriesuid.slice(0, 24)}...` : 'AXIAL CT SLICE'}
          </span>
        </div>

        {/* Grad-CAM Toggle Control */}
        <div className="flex items-center gap-3">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <span className="text-[11px] text-gray-400">Grad-CAM Overlay</span>
            <input
              type="checkbox"
              checked={showGradCam}
              onChange={(e) => setShowGradCam(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-8 h-4 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-teal-600 relative"></div>
          </label>

          <div className="flex items-center gap-1 bg-gray-800 rounded px-1.5 py-0.5">
            <button
              onClick={() => setZoomLevel((z) => Math.max(1, z - 0.25))}
              className="px-1.5 hover:text-white text-gray-400"
              title="Zoom Out"
            >
              -
            </button>
            <span className="text-[10px] font-mono text-gray-300">{zoomLevel.toFixed(1)}x</span>
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.5, z + 0.25))}
              className="px-1.5 hover:text-white text-gray-400"
              title="Zoom In"
            >
              +
            </button>
          </div>
        </div>
      </div>

      {/* Main Viewport */}
      <div className="relative bg-black flex items-center justify-center min-h-[340px] sm:min-h-[420px] overflow-hidden p-4 select-none">
        {gradcamSlicePngB64 ? (
          <div
            className="transition-transform duration-150 ease-out flex items-center justify-center"
            style={{
              transform: `scale(${zoomLevel})`,
              filter: showGradCam ? 'none' : 'grayscale(100%) brightness(1.1) contrast(1.2)',
            }}
          >
            <img
              src={gradcamSlicePngB64}
              alt="Axial CT slice with Grad-CAM activation"
              className="max-h-[380px] w-auto object-contain rounded shadow-2xl border border-gray-800"
            />
          </div>
        ) : (
          <div className="text-gray-500 text-xs flex flex-col items-center gap-2">
            <svg className="w-10 h-10 text-gray-700 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span>No slice data loaded</span>
          </div>
        )}

        {/* Radiological Overlay HUD Badges */}
        <div className="absolute top-3 left-3 text-[10px] font-mono text-gray-400 space-y-0.5 bg-black/60 p-2 rounded border border-gray-800 backdrop-blur-sm pointer-events-none">
          <div>WL: -600 / WW: 1500 (Lung)</div>
          <div>ISOTROPIC: 1.0 x 1.0 x 1.0 mm</div>
          {noduleCoord && (
            <div>
              COORD: [{noduleCoord[0]}, {noduleCoord[1]}, {noduleCoord[2]}]
            </div>
          )}
        </div>

        {diameterMm !== undefined && diameterMm > 0 && (
          <div className="absolute bottom-3 right-3 text-[10px] font-mono text-teal-300 bg-teal-950/80 border border-teal-800 p-2 rounded backdrop-blur-sm pointer-events-none">
            <div>TARGET DIAMETER: {diameterMm.toFixed(1)} mm</div>
            {confidence && <div>DETECTION CONF: {(confidence * 100).toFixed(0)}%</div>}
            {showGradCam && <div className="text-[9px] text-gray-400">3D DENSENET GRAD-CAM</div>}
          </div>
        )}
      </div>

      {/* Footer Info Strip */}
      <div className="bg-gray-950 px-4 py-2 border-t border-gray-800 flex items-center justify-between text-[11px] text-gray-400">
        <span>Axial Multi-planar Reconstruction (MPR)</span>
        <span className="text-gray-500">
          {showGradCam ? 'Heatmap active (Jet colormap)' : 'Raw CT intensity'}
        </span>
      </div>
    </div>
  );
}
