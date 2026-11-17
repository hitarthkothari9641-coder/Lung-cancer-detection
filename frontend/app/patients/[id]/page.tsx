'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import {
  fetchPatients,
  predictScan,
  Patient,
  PredictionResult,
  getReportDownloadUrl,
} from '../../../lib/api';
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
          try {
            setPrediction(await predictScan(found.seriesuid, found as unknown as Record<string, unknown>));
          } catch {
            /* The record still renders even if the scan cannot be analysed. */
          }
        } else {
          setError(`No patient found for "${patientId}".`);
        }
      } catch (err: any) {
        setError(err.message || 'Unable to load this patient.');
      } finally {
        setLoading(false);
      }
    }
    loadPatient();
  }, [patientId]);

  if (loading) {
    return (
      <div className="card-surface p-12 text-center text-xs text-gray-500">
        Loading scan…
      </div>
    );
  }

  if (error || !patient) {
    return (
      <div className="card-surface p-8 text-center space-y-3">
        <p className="text-rose-600 font-medium text-sm">{error || 'Patient not found'}</p>
        <Link href="/patients" className="text-xs text-accent hover:underline">
          ← Back to patient directory
        </Link>
      </div>
    );
  }

  const hasNodule = patient.nodule_size_mm > 0;
  const reportUrl = getReportDownloadUrl(patient.id);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
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
          className="inline-flex items-center gap-2 bg-surface border border-border hover:border-gray-400 text-textMain px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
        >
          <svg className="w-3.5 h-3.5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
          Download report
        </a>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 space-y-4">
          <ScanViewer
            gradcamSlicePngB64={prediction?.gradcam_slice_png_b64}
            seriesuid={patient.seriesuid}
            noduleCoord={prediction?.top_nodule?.coord}
            diameterMm={patient.nodule_size_mm}
            confidence={patient.confidence}
            sliceLabel="Axial CT"
          />

          <div className="card-surface p-4 text-xs text-gray-500 leading-relaxed">
            <p className="font-medium text-gray-700 mb-1">Impression</p>
            <p>
              {hasNodule
                ? `Solid focal opacity measuring ${patient.nodule_size_mm.toFixed(
                    1,
                  )} mm in the ${patient.location.toLowerCase()}, highlighted on the axial slice.`
                : 'No focal solid nodule identified. Lung parenchyma appears clear bilaterally.'}
            </p>
          </div>
        </div>

        <div className="lg:col-span-5 space-y-5">
          <div className="card-surface p-5 space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider block">
                  Assessment
                </span>
                <h1 className="text-lg font-semibold tracking-tight text-textMain">
                  {patient.grade_label}
                </h1>
              </div>
              <GradeChip grade={patient.model_grade} />
            </div>

            <dl className="grid grid-cols-3 gap-2 py-3 border-y border-border text-center">
              <div>
                <dt className="text-[10px] text-gray-400 uppercase">Lung-RADS</dt>
                <dd className="text-xs font-semibold text-textMain font-mono">{patient.lung_rads}</dd>
              </div>
              <div>
                <dt className="text-[10px] text-gray-400 uppercase">Confidence</dt>
                <dd className="text-xs font-semibold text-textMain font-mono">
                  {(patient.confidence * 100).toFixed(0)}%
                </dd>
              </div>
              <div>
                <dt className="text-[10px] text-gray-400 uppercase">Diameter</dt>
                <dd className="text-xs font-semibold text-textMain font-mono">
                  {hasNodule ? `${patient.nodule_size_mm.toFixed(1)} mm` : '—'}
                </dd>
              </div>
            </dl>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-gray-500 font-medium">Malignancy rating</span>
                <span className="font-mono font-semibold text-accent">
                  {patient.malignancy_score} / 5
                </span>
              </div>
              <div className="grid grid-cols-5 gap-1.5">
                {[1, 2, 3, 4, 5].map((lvl) => (
                  <span
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
                <span>1 · benign</span>
                <span>3 · indeterminate</span>
                <span>5 · malignant</span>
              </div>
            </div>

            <div className="bg-teal-50/70 border border-teal-200 rounded-md p-3.5 space-y-1">
              <span className="text-[10px] font-medium text-teal-800 uppercase tracking-wide">
                Recommended follow-up
              </span>
              <p className="text-xs font-medium text-teal-950 capitalize">{patient.recommendation}</p>
            </div>
          </div>

          <div className="card-surface p-5 space-y-3">
            <h2 className="text-xs font-semibold text-textMain uppercase tracking-wider">Patient profile</h2>
            <dl className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <dt className="text-gray-400 block text-[10px]">Age / Sex</dt>
                <dd className="font-medium text-gray-800">
                  {patient.age} yrs · {patient.sex}
                </dd>
              </div>
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <dt className="text-gray-400 block text-[10px]">Smoking history</dt>
                <dd className="font-medium text-gray-800">{patient.smoking_pack_years} pack-years</dd>
              </div>
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <dt className="text-gray-400 block text-[10px]">Location</dt>
                <dd className="font-medium text-gray-800">{patient.location}</dd>
              </div>
              <div className="bg-gray-50 p-2.5 rounded border border-gray-100">
                <dt className="text-gray-400 block text-[10px]">Scan date</dt>
                <dd className="font-medium text-gray-800 font-mono">{patient.scan_date}</dd>
              </div>
            </dl>
          </div>

          <div className="card-surface p-5 space-y-3">
            <h2 className="text-xs font-semibold text-textMain uppercase tracking-wider">
              Detected candidates
            </h2>
            {prediction?.nodules && prediction.nodules.length > 0 ? (
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
  );
}
