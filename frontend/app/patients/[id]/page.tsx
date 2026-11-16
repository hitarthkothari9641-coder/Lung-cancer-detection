'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { fetchPatients, predictScan, Patient, PredictionResult, getReportDownloadUrl } from '../../../lib/api';
import ScanViewer from '../../../components/ScanViewer';
import GradeChip from '../../../components/GradeChip';

export default function PatientDetailPage() {
  const params = useParams();
  const patientId = params?.id as string;

  const [patient, setPatient] = useState<Patient | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadPatient() {
      if (!patientId) return;
      try {
        setLoading(true);
        const patients = await fetchPatients();
        const found = patients.find((p) => p.id === patientId || p.seriesuid === patientId);

        if (found) {
          setPatient(found);
          // Run prediction to get Grad-CAM slice and candidate details
          const pred = await predictScan(found.seriesuid, found);
          setPrediction(pred);
        } else {
          setError(`Patient with ID "${patientId}" was not found in the cohort.`);
        }
      } catch (err: any) {
        setError(err.message || 'Error loading patient scan');
      } finally {
        setLoading(false);
      }
    }
    loadPatient();
  }, [patientId]);

  if (loading) {
    return (
      <div className="card-surface p-12 text-center text-xs text-gray-500">
        Loading 3D volumetric reconstruction and Grad-CAM overlays...
      </div>
    );
  }

  if (error || !patient) {
    return (
      <div className="card-surface p-8 text-center space-y-3">
        <div className="text-rose-600 font-semibold text-sm">{error || 'Patient not found'}</div>
        <Link href="/patients" className="text-xs text-accent hover:underline">
          ← Return to patient directory
        </Link>
      </div>
    );
  }

  const reportUrl = getReportDownloadUrl(patient.id);
  const hasNodule = patient.nodule_size_mm > 0;

  return (
    <div className="space-y-6">
      {/* Navigation Breadcrumb */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Link href="/patients" className="hover:text-accent transition-colors">
            Patients
          </Link>
          <span>/</span>
          <span className="text-textMain font-medium">{patient.name}</span>
          <span className="font-mono text-gray-400">({patient.id})</span>
        </div>

        <a
          href={reportUrl}
          download
          className="inline-flex items-center gap-2 bg-surface border border-border hover:border-gray-400 text-textMain px-3 py-1.5 rounded-md text-xs font-medium shadow-sm transition-colors"
        >
          <svg className="w-3.5 h-3.5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download Report (JSON)
        </a>
      </div>

      {/* Main Grid: Left = ScanViewer, Right = Clinical Evaluation Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: 3D CT + Grad-CAM Viewport (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <ScanViewer
            gradcamSlicePngB64={prediction?.gradcam_slice_png_b64}
            seriesuid={patient.seriesuid}
            noduleCoord={prediction?.top_nodule?.coord}
            diameterMm={patient.nodule_size_mm}
            grade={patient.model_grade}
            confidence={patient.confidence}
          />

          <div className="card-surface p-4 text-xs text-gray-500 space-y-1">
            <div className="font-semibold text-gray-700">Radiological Interpretation Note:</div>
            <div>
              {hasNodule
                ? `Focal solid attenuation measured at ${patient.nodule_size_mm.toFixed(1)} mm located in the ${patient.location}. 3D DenseNet-121 activation maps indicate hyper-intense feature focus corresponding to solid nodule margin.`
                : 'No focal solid pulmonary nodule identified above 3.0 mm. Lung parenchyma demonstrates clear bilateral aeration without architectural distortion.'}
            </div>
          </div>
        </div>

        {/* Right Column: Clinical Grading & Recommendations (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Diagnostic Assessment Card */}
          <div className="card-surface p-5 space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <span className="text-[10px] font-mono text-gray-400 uppercase tracking-wider block">
                  AI Cancer Grading
                </span>
                <h2 className="text-lg font-bold text-textMain tracking-tight">
                  {patient.grade_label}
                </h2>
              </div>
              <GradeChip
                grade={patient.model_grade}
                labelSource={prediction?.label_source || 'proxy'}
              />
            </div>

            {/* Metrics Ribbon */}
            <div className="grid grid-cols-3 gap-2 py-3 border-y border-border text-center">
              <div>
                <span className="text-[10px] text-gray-400 block uppercase">Lung-RADS</span>
                <span className="text-xs font-semibold text-textMain font-mono">
                  {patient.lung_rads}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 block uppercase">Confidence</span>
                <span className="text-xs font-semibold text-textMain font-mono">
                  {(patient.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 block uppercase">Nodule Size</span>
                <span className="text-xs font-semibold text-textMain font-mono">
                  {hasNodule ? `${patient.nodule_size_mm.toFixed(1)} mm` : '0 mm'}
                </span>
              </div>
            </div>

            {/* Ordinal Malignancy Score Meter */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-gray-500 font-medium">Ordinal Malignancy Score</span>
                <span className="font-mono font-bold text-accent">
                  {patient.malignancy_score} / 5
                </span>
              </div>
              <div className="grid grid-cols-5 gap-1.5">
                {[1, 2, 3, 4, 5].map((lvl) => (
                  <div
                    key={lvl}
                    className={`h-2 rounded-full ${
                      lvl <= patient.malignancy_score
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
              <div className="flex justify-between text-[10px] text-gray-400">
                <span>1 (Benign)</span>
                <span>3 (Indeterminate)</span>
                <span>5 (Malignant)</span>
              </div>
            </div>

            {/* Clinical Action Recommendation */}
            <div className="bg-teal-50/70 border border-teal-200 rounded-md p-3.5 space-y-1">
              <span className="text-[10px] font-mono text-teal-800 uppercase tracking-wide font-semibold">
                Clinical Recommendation (Lung-RADS Protocol)
              </span>
              <p className="text-xs font-medium text-teal-950 capitalize">
                {patient.recommendation}
              </p>
            </div>
          </div>

          {/* Patient Risk Profile Card */}
          <div className="card-surface p-5 space-y-3">
            <h3 className="text-xs font-semibold text-textMain uppercase tracking-wider">
              Patient Demographic & Risk Profile
            </h3>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <span className="text-gray-400 block text-[10px]">Age / Sex</span>
                <span className="font-medium text-gray-800">
                  {patient.age} yrs • {patient.sex}
                </span>
              </div>
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <span className="text-gray-400 block text-[10px]">Tobacco Exposure</span>
                <span className="font-medium text-gray-800">
                  {patient.smoking_pack_years} pack-years
                </span>
              </div>
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <span className="text-gray-400 block text-[10px]">Anatomical Lobe</span>
                <span className="font-medium text-gray-800">{patient.location}</span>
              </div>
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <span className="text-gray-400 block text-[10px]">Scan Acquisition</span>
                <span className="font-medium text-gray-800 font-mono">{patient.scan_date}</span>
              </div>
            </div>
          </div>

          {/* Detected Nodule Candidates */}
          <div className="card-surface p-5 space-y-3">
            <h3 className="text-xs font-semibold text-textMain uppercase tracking-wider">
              Detected Nodule Coordinates
            </h3>
            {prediction?.nodules && prediction.nodules.length > 0 ? (
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
  );
}
