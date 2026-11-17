'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { uploadScanFile, PredictionResult, getReportDownloadUrl } from '../../lib/api';
import ScanViewer from '../../components/ScanViewer';
import GradeChip from '../../components/GradeChip';

const SAMPLE_SCANS = [
  { id: '1.3.6.1.4.1.14519.5.2.1.6279.6001.1011', name: 'High-risk sample', detail: '13.8 mm · Grade IV' },
  { id: '1.3.6.1.4.1.14519.5.2.1.6279.6001.1008', name: 'Suspicious sample', detail: '7.9 mm · Grade III' },
  { id: '1.3.6.1.4.1.14519.5.2.1.6279.6001.1001', name: 'Clear sample', detail: 'No nodule · Grade 0' },
];

export default function UploadScanPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [patientName, setPatientName] = useState<string>('Jane Doe');
  const [patientAge, setPatientAge] = useState<number>(64);
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setPrediction(null);
    setLoading(true);

    try {
      if (!selectedFile) {
        setError('Choose a CT image or volume file first.');
        return;
      }
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('patient_name', patientName);
      formData.append('patient_age', String(patientAge));
      setPrediction(await uploadScanFile(formData));
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Please try another file.');
    } finally {
      setLoading(false);
    }
  };

  const handleSample = async (seriesId: string, name: string) => {
    setError(null);
    setLoading(true);
    setPatientName(name);
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seriesuid: seriesId,
          patient: { name, age: 68, smoking_pack_years: 40 },
        }),
      });
      if (!res.ok) throw new Error('Sample analysis failed.');
      setPrediction(await res.json());
    } catch (err: any) {
      setError(err.message || 'Sample analysis failed.');
    } finally {
      setLoading(false);
    }
  };

  const hasNodule = Boolean(prediction?.top_nodule && prediction.top_nodule.diameter_mm > 0);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-textMain">
          Analyze a CT scan
        </h1>
        <p className="text-xs text-gray-500 mt-1">
          Upload an axial CT image (PNG/JPG) or a MetaImage volume (.mhd/.raw) to review nodule size,
          severity grade and follow-up guidance.
        </p>
      </div>

      <div className="card-surface p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') document.getElementById('file-upload-input')?.click();
            }}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
              isDragging ? 'border-accent bg-teal-50/50' : 'border-border bg-gray-50/50 hover:bg-gray-50'
            }`}
            onClick={() => document.getElementById('file-upload-input')?.click()}
          >
            <input
              id="file-upload-input"
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp,.mhd,.raw,.dcm"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  setSelectedFile(file);
                  setPrediction(null);
                }
              }}
            />
            <div className="flex flex-col items-center justify-center gap-2">
              <svg className="w-8 h-8 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="1.5"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="text-xs font-medium text-textMain">
                {selectedFile ? (
                  <span className="text-accent font-mono">{selectedFile.name}</span>
                ) : (
                  'Drag a scan here, or click to browse'
                )}
              </p>
              <p className="text-[11px] text-gray-400">PNG, JPG, TIFF, or MetaImage (.mhd / .raw)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div>
              <label htmlFor="patient-name" className="block text-gray-600 font-medium mb-1">
                Patient name
              </label>
              <input
                id="patient-name"
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="w-full bg-surface border border-border rounded-md px-3 py-2 text-textMain focus:outline-none focus:border-accent"
                required
              />
            </div>
            <div>
              <label htmlFor="patient-age" className="block text-gray-600 font-medium mb-1">
                Age
              </label>
              <input
                id="patient-age"
                type="number"
                value={patientAge}
                onChange={(e) => setPatientAge(Number(e.target.value))}
                className="w-full bg-surface border border-border rounded-md px-3 py-2 text-textMain focus:outline-none focus:border-accent"
                min="18"
                max="120"
                required
              />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2 border-t border-border">
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="text-gray-400">Try a sample scan:</span>
              {SAMPLE_SCANS.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => handleSample(s.id, s.name)}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-2.5 py-1 rounded transition-colors"
                  title={s.detail}
                >
                  {s.name}
                </button>
              ))}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="bg-accent hover:bg-accent-hover disabled:bg-gray-300 text-white text-xs font-semibold px-5 py-2.5 rounded-md transition-colors flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Analyzing…
                </>
              ) : (
                'Run analysis'
              )}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-md text-xs text-rose-700">{error}</div>
      )}

      {prediction && (
        <div className="space-y-6 pt-4 border-t border-border">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold tracking-tight text-textMain">
                Results · {patientName}
              </h2>
              <p className="text-xs text-gray-400 font-mono">Series {prediction.seriesuid}</p>
            </div>
            <div className="flex items-center gap-2">
              <a
                href={getReportDownloadUrl(prediction.seriesuid)}
                download
                className="inline-flex items-center gap-2 bg-surface border border-border hover:border-gray-400 text-textMain px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
              >
                Download report
              </a>
              <Link
                href="/patients"
                className="text-xs font-medium text-accent hover:text-accent-hover transition-colors"
              >
                Patient directory →
              </Link>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7">
              <ScanViewer
                gradcamSlicePngB64={prediction.gradcam_slice_png_b64}
                seriesuid={prediction.seriesuid}
                noduleCoord={prediction.top_nodule?.coord}
                diameterMm={prediction.top_nodule?.diameter_mm || 0}
                confidence={prediction.confidence}
                sliceLabel="Analyzed slice"
              />
            </div>

            <div className="lg:col-span-5 space-y-5">
              <div className="card-surface p-5 space-y-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider block">
                      Assessment
                    </span>
                    <h3 className="text-lg font-semibold tracking-tight text-textMain">
                      {prediction.grade_label}
                    </h3>
                  </div>
                  <GradeChip grade={prediction.grade} />
                </div>

                <dl className="grid grid-cols-3 gap-2 py-3 border-y border-border text-center">
                  <div>
                    <dt className="text-[10px] text-gray-400 uppercase">Lung-RADS</dt>
                    <dd className="text-xs font-semibold text-textMain font-mono">{prediction.lung_rads}</dd>
                  </div>
                  <div>
                    <dt className="text-[10px] text-gray-400 uppercase">Confidence</dt>
                    <dd className="text-xs font-semibold text-textMain font-mono">
                      {(prediction.confidence * 100).toFixed(0)}%
                    </dd>
                  </div>
                  <div>
                    <dt className="text-[10px] text-gray-400 uppercase">Diameter</dt>
                    <dd className="text-xs font-semibold text-textMain font-mono">
                      {hasNodule ? `${prediction.top_nodule!.diameter_mm.toFixed(1)} mm` : '—'}
                    </dd>
                  </div>
                </dl>

                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500 font-medium">Malignancy rating</span>
                    <span className="font-mono font-semibold text-accent">
                      {prediction.malignancy_score} / 5
                    </span>
                  </div>
                  <div className="grid grid-cols-5 gap-1.5">
                    {[1, 2, 3, 4, 5].map((lvl) => (
                      <span
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

                <div className="bg-teal-50/70 border border-teal-200 rounded-md p-3.5 space-y-1">
                  <span className="text-[10px] font-medium text-teal-800 uppercase tracking-wide">
                    Recommended follow-up
                  </span>
                  <p className="text-xs font-medium text-teal-950 capitalize">{prediction.recommendation}</p>
                </div>
              </div>

              <div className="card-surface p-5 space-y-3">
                <h3 className="text-xs font-semibold text-textMain uppercase tracking-wider">
                  Detected candidates
                </h3>
                {prediction.nodules?.length ? (
                  <ul className="space-y-2">
                    {prediction.nodules.map((n, idx) => (
                      <li
                        key={idx}
                        className="flex items-center justify-between text-xs p-2 bg-gray-50 rounded border border-gray-100 font-mono"
                      >
                        <span className="text-gray-600">
                          <span className="text-gray-400 text-[10px] mr-2">#{idx + 1}</span>[
                          {n.coord.map((c) => c.toFixed(1)).join(', ')}]
                        </span>
                        <span className="text-right">
                          <span className="text-gray-700 font-medium mr-2">{n.diameter_mm.toFixed(1)} mm</span>
                          <span className="text-teal-700 font-semibold">{(n.prob * 100).toFixed(0)}%</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-xs text-gray-400 italic">No nodule candidates registered.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
