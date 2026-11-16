'use client';

import React, { useState } from 'react';
import { predictScan, uploadScanFile, PredictionResult, getReportDownloadUrl } from '../../lib/api';
import ScanViewer from '../../components/ScanViewer';
import GradeChip from '../../components/GradeChip';

export default function UploadScanPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [patientName, setPatientName] = useState<string>('Jane Doe');
  const [patientAge, setPatientAge] = useState<number>(64);
  const [packYears, setPackYears] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Drag and drop events
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (selectedFile) {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('patient_name', patientName);
        formData.append('patient_age', String(patientAge));

        const res = await uploadScanFile(formData);
        setPrediction(res);
      } else {
        // Fallback or demo run with a sample series
        const res = await predictScan('1.3.6.1.4.1.14519.5.2.1.6279.6001.1011', {
          name: patientName,
          age: patientAge,
          smoking_pack_years: packYears,
        });
        setPrediction(res);
      }
    } catch (err: any) {
      setError(err.message || 'Error executing nodule prediction pipeline.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDemo = async (seriesId: string, demoName: string) => {
    setError(null);
    setLoading(true);
    setPatientName(demoName);
    try {
      const res = await predictScan(seriesId, {
        name: demoName,
        age: 68,
        smoking_pack_years: 40,
      });
      setPrediction(res);
    } catch (err: any) {
      setError(err.message || 'Demo prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  const hasNodule = prediction?.top_nodule && prediction.top_nodule.diameter_mm > 0;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-textMain tracking-tight">
          3D CT Scan Upload & Inference
        </h1>
        <p className="text-xs text-gray-500 mt-1">
          Upload DICOM/MetaImage (.mhd + .raw) volumes for isotropic 3D nodule detection and ordinal malignancy grading.
        </p>
      </div>

      {/* Upload & Form Section */}
      <div className="card-surface p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Drag & Drop Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
              isDragging
                ? 'border-accent bg-teal-50/50'
                : 'border-border bg-gray-50/50 hover:bg-gray-50'
            }`}
            onClick={() => document.getElementById('file-upload-input')?.click()}
          >
            <input
              id="file-upload-input"
              type="file"
              accept=".mhd,.raw,.dcm,.zip"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  setSelectedFile(e.target.files[0]);
                }
              }}
            />
            <div className="flex flex-col items-center justify-center gap-2">
              <svg className="w-8 h-8 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <div className="text-xs font-semibold text-textMain">
                {selectedFile ? (
                  <span className="text-accent font-mono">{selectedFile.name}</span>
                ) : (
                  'Drag and drop your .mhd or .raw volume here, or click to browse'
                )}
              </div>
              <div className="text-[11px] text-gray-400">
                Supports LUNA16 MetaImage (.mhd/.raw) or DICOM series
              </div>
            </div>
          </div>

          {/* Patient Details & Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <div>
              <label className="block text-gray-600 font-medium mb-1">Patient Name</label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="w-full bg-surface border border-border rounded px-3 py-2 text-textMain focus:outline-none focus:border-accent"
                required
              />
            </div>
            <div>
              <label className="block text-gray-600 font-medium mb-1">Patient Age</label>
              <input
                type="number"
                value={patientAge}
                onChange={(e) => setPatientAge(Number(e.target.value))}
                className="w-full bg-surface border border-border rounded px-3 py-2 text-textMain focus:outline-none focus:border-accent"
                min="18"
                max="120"
                required
              />
            </div>
            <div>
              <label className="block text-gray-600 font-medium mb-1">Smoking Pack-Years</label>
              <input
                type="number"
                value={packYears}
                onChange={(e) => setPackYears(Number(e.target.value))}
                className="w-full bg-surface border border-border rounded px-3 py-2 text-textMain focus:outline-none focus:border-accent"
                min="0"
                max="120"
              />
            </div>
          </div>

          {/* Quick Demo Pre-load Pills */}
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-border text-xs">
            <span className="text-gray-400 text-[11px]">Instant Demo Samples:</span>
            <button
              type="button"
              onClick={() => handleQuickDemo('1.3.6.1.4.1.14519.5.2.1.6279.6001.1011', 'Walter White (High Risk)')}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-2.5 py-1 rounded text-[11px] font-mono transition-colors"
            >
              Grade IV (13.8mm Biopsy)
            </button>
            <button
              type="button"
              onClick={() => handleQuickDemo('1.3.6.1.4.1.14519.5.2.1.6279.6001.1008', 'Clara Oswald (Suspicious)')}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-2.5 py-1 rounded text-[11px] font-mono transition-colors"
            >
              Grade III (7.9mm 4A)
            </button>
            <button
              type="button"
              onClick={() => handleQuickDemo('1.3.6.1.4.1.14519.5.2.1.6279.6001.1001', 'Eleanor Vance (Clear)')}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-2.5 py-1 rounded text-[11px] font-mono transition-colors"
            >
              Grade 0 (No Nodule)
            </button>
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="bg-accent hover:bg-accent-hover disabled:bg-gray-300 text-white text-xs font-semibold px-6 py-2.5 rounded-md shadow-sm transition-colors flex items-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-1 h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                  </svg>
                  Processing 3D DenseNet-121...
                </>
              ) : (
                'Run AI Detection & Cancer Grading'
              )}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded text-xs text-rose-700">
          {error}
        </div>
      )}

      {/* Prediction Output Section (Identical to patient detail view) */}
      {prediction && (
        <div className="space-y-6 pt-4 border-t border-border">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-textMain tracking-tight">
                Prediction Results: {patientName}
              </h2>
              <p className="text-xs text-gray-400 font-mono">
                SeriesUID: {prediction.seriesuid}
              </p>
            </div>
            <a
              href={getReportDownloadUrl(prediction.seriesuid)}
              download
              className="inline-flex items-center gap-2 bg-surface border border-border hover:border-gray-400 text-textMain px-3 py-1.5 rounded-md text-xs font-medium shadow-sm transition-colors"
            >
              <svg className="w-3.5 h-3.5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download JSON Report
            </a>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: ScanViewer (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              <ScanViewer
                gradcamSlicePngB64={prediction.gradcam_slice_png_b64}
                seriesuid={prediction.seriesuid}
                noduleCoord={prediction.top_nodule?.coord}
                diameterMm={prediction.top_nodule?.diameter_mm || 0}
                grade={prediction.grade}
                confidence={prediction.confidence}
              />
            </div>

            {/* Right: Diagnostic Card (5 cols) */}
            <div className="lg:col-span-5 space-y-5">
              <div className="card-surface p-5 space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider block">
                      AI Cancer Grading
                    </span>
                    <h3 className="text-lg font-bold text-textMain tracking-tight">
                      {prediction.grade_label}
                    </h3>
                  </div>
                  <GradeChip
                    grade={prediction.grade}
                    labelSource={prediction.label_source}
                  />
                </div>

                <div className="grid grid-cols-3 gap-2 py-3 border-y border-border text-center">
                  <div>
                    <span className="text-[10px] text-gray-400 block uppercase">Lung-RADS</span>
                    <span className="text-xs font-semibold text-textMain font-mono">
                      {prediction.lung_rads}
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-gray-400 block uppercase">Confidence</span>
                    <span className="text-xs font-semibold text-textMain font-mono">
                      {(prediction.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-gray-400 block uppercase">Nodule Size</span>
                    <span className="text-xs font-semibold text-textMain font-mono">
                      {hasNodule ? `${prediction.top_nodule?.diameter_mm.toFixed(1)} mm` : '0 mm'}
                    </span>
                  </div>
                </div>

                {/* Ordinal Malignancy Meter */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500 font-medium">Ordinal Malignancy Score</span>
                    <span className="font-mono font-bold text-accent">
                      {prediction.malignancy_score} / 5
                    </span>
                  </div>
                  <div className="grid grid-cols-5 gap-1.5">
                    {[1, 2, 3, 4, 5].map((lvl) => (
                      <div
                        key={lvl}
                        className={`h-2 rounded-full ${
                          lvl <= prediction.malignancy_score
                            ? lvl <= 2
                              ? 'bg-emerald-500'
                              : lvl === 3
                              ? 'bg-amber-500'
                              : 'bg-rose-500'
                            : 'bg-gray-200'
                        }`}
                      />
                    ))}
                  </div>
                </div>

                {/* Recommendation */}
                <div className="bg-teal-50/70 border border-teal-200 rounded-md p-3.5 space-y-1">
                  <span className="text-[10px] font-mono text-teal-800 uppercase tracking-wide font-semibold">
                    Clinical Action Protocol
                  </span>
                  <p className="text-xs font-medium text-teal-950 capitalize">
                    {prediction.recommendation}
                  </p>
                </div>
              </div>

              {/* Detected Nodules List */}
              <div className="card-surface p-5 space-y-3">
                <h3 className="text-xs font-semibold text-textMain uppercase tracking-wider">
                  Detected Nodule Coordinates
                </h3>
                {prediction.nodules && prediction.nodules.length > 0 ? (
                  <div className="space-y-2">
                    {prediction.nodules.map((n, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between text-xs p-2 bg-gray-50 rounded border border-gray-100 font-mono"
                      >
                        <div>
                          <span className="text-gray-400 text-[10px] mr-2">#{idx + 1}</span>
                          <span>[{n.coord.map((c) => c.toFixed(1)).join(', ')}]</span>
                        </div>
                        <div className="text-right">
                          <span className="text-gray-600 font-medium mr-2">{n.diameter_mm.toFixed(1)} mm</span>
                          <span className="text-teal-700 font-semibold">{(n.prob * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-gray-400 italic">No nodules registered.</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
