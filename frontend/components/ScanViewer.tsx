'use client';

import React, { useState } from 'react';

interface ScanViewerProps {
  gradcamSlicePngB64?: string;
  seriesuid?: string;
  noduleCoord?: [number, number, number];
  diameterMm?: number;
  confidence?: number;
  sliceLabel?: string;
}

export default function ScanViewer({
  gradcamSlicePngB64,
  seriesuid,
  noduleCoord,
  diameterMm,
  confidence,
  sliceLabel,
}: ScanViewerProps) {
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const hasLesion = typeof diameterMm === 'number' && diameterMm > 0;

  return (
    <div className="card-surface overflow-hidden flex flex-col bg-slate-900 border-slate-800">
      {/* Toolbar */}
      <div className="bg-slate-950 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between gap-3 text-xs">
        <span className="font-medium text-slate-300 truncate">
          {sliceLabel || 'Axial view'}
          {seriesuid ? <span className="text-slate-500 font-normal"> · {seriesuid}</span> : null}
        </span>

        <div className="flex items-center gap-3 shrink-0">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <span className="text-[11px] text-slate-400">Highlight</span>
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
              className="sr-only peer"
            />
            <span className="relative w-8 h-4 rounded-full bg-slate-700 peer-checked:bg-teal-600 transition-colors after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:h-3 after:w-3 after:rounded-full after:bg-white after:transition-transform peer-checked:after:translate-x-4" />
          </label>

          <div className="flex items-center gap-1 bg-slate-800 rounded px-1.5 py-0.5">
            <button
              type="button"
              onClick={() => setZoomLevel((z) => Math.max(1, Number((z - 0.25).toFixed(2))))}
              className="px-1.5 text-slate-400 hover:text-white"
              aria-label="Zoom out"
            >
              −
            </button>
            <span className="text-[10px] font-mono text-slate-300 w-8 text-center">
              {zoomLevel.toFixed(2)}x
            </span>
            <button
              type="button"
              onClick={() => setZoomLevel((z) => Math.min(2.5, Number((z + 0.25).toFixed(2))))}
              className="px-1.5 text-slate-400 hover:text-white"
              aria-label="Zoom in"
            >
              +
            </button>
          </div>
        </div>
      </div>

      {/* Viewport */}
      <div className="relative bg-black flex items-center justify-center min-h-[320px] sm:min-h-[400px] overflow-hidden p-4 select-none">
        {gradcamSlicePngB64 ? (
          <div
            className="transition-transform duration-150 ease-out flex items-center justify-center"
            style={{
              transform: `scale(${zoomLevel})`,
              filter: showOverlay ? 'none' : 'grayscale(100%) brightness(1.05)',
            }}
          >
            <img
              src={gradcamSlicePngB64}
              alt="Axial CT slice with lesion highlight"
              className="max-h-[360px] w-auto object-contain rounded border border-slate-800"
            />
          </div>
        ) : (
          <div className="text-slate-500 text-xs flex flex-col items-center gap-2">
            <svg className="w-8 h-8 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
                d="M4 5h16v14H4zM4 12h16M12 5v14"
              />
            </svg>
            <span>No image available</span>
          </div>
        )}

        {hasLesion && (
          <div className="absolute bottom-3 right-3 text-[10px] font-mono text-teal-200 bg-slate-900/85 border border-teal-800/70 px-2 py-1.5 rounded space-y-0.5">
            <div>Lesion: {diameterMm!.toFixed(1)} mm</div>
            {typeof confidence === 'number' && <div>Confidence: {(confidence * 100).toFixed(0)}%</div>}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="bg-slate-950 px-4 py-2 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
        <span className="truncate">
          {noduleCoord
            ? `Location (mm): ${noduleCoord.map((c) => c.toFixed(1)).join(', ')}`
            : 'No focal lesion localised'}
        </span>
        <span className="text-slate-500">
          {showOverlay ? 'Highlight on' : 'Highlight off'}
        </span>
      </div>
    </div>
  );
}
