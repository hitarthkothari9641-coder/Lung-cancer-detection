'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { fetchStats, fetchPatients, AggregateStats, Patient } from '../lib/api';
import StatCard from '../components/StatCard';
import GradeDistribution from '../components/GradeDistribution';
import PatientTable from '../components/PatientTable';

export default function DashboardPage() {
  const [stats, setStats] = useState<AggregateStats | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [statsData, patientsData] = await Promise.all([
          fetchStats().catch(() => null),
          fetchPatients().catch(() => []),
        ]);

        if (statsData) {
          setStats(statsData);
        } else if (patientsData.length > 0) {
          const total = patientsData.length;
          const detected = patientsData.filter((p) => p.nodule_size_mm > 0).length;
          const dist: Record<string, number> = { '0': 0, I: 0, II: 0, III: 0, IV: 0 };
          let highRisk = 0;
          let confSum = 0;
          patientsData.forEach((p) => {
            dist[p.model_grade] = (dist[p.model_grade] || 0) + 1;
            confSum += p.confidence;
            if (p.model_grade === 'III' || p.model_grade === 'IV') highRisk += 1;
          });

          setStats({
            total_patients: total,
            total_scans: total,
            nodules_detected: detected,
            grade_distribution: dist,
            mean_confidence: Number((confSum / total).toFixed(2)),
            high_risk_pct: Number(((highRisk / total) * 100).toFixed(1)),
            avg_nodules_per_scan: Number((detected / total).toFixed(2)),
            val_sensitivity: null,
            val_specificity: null,
            label_source: 'proxy',
            disclaimer: 'Research prototype. Not a medical device. Not for clinical diagnosis.',
          });
        }

        setPatients(patientsData);
      } catch (err: any) {
        setError(err.message || 'Unable to load cohort data.');
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-textMain">
            Screening overview
          </h1>
          <p className="text-xs text-gray-500 mt-1">
            Cohort-level nodule findings and Lung-RADS severity distribution.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/patients"
            className="card-surface px-3 py-1.5 text-xs font-medium text-gray-700 hover:text-accent transition-colors"
          >
            All patients ({patients.length})
          </Link>
          <Link
            href="/upload"
            className="bg-accent hover:bg-accent-hover text-white text-xs font-medium px-3.5 py-1.5 rounded-md transition-colors"
          >
            Analyze scan
          </Link>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-md text-xs text-rose-700">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Scans reviewed"
          value={stats ? stats.total_scans : loading ? '…' : '0'}
          subtext="1.0 mm isotropic axial volumes"
        />
        <StatCard
          title="Scans with a nodule"
          value={stats ? stats.nodules_detected : loading ? '…' : '0'}
          subtext={stats ? `${stats.avg_nodules_per_scan} findings per scan` : ''}
        />
        <StatCard
          title="High risk"
          value={stats ? `${stats.high_risk_pct}%` : loading ? '…' : '0%'}
          subtext="Grades III & IV (Lung-RADS 4A–4X)"
        />
        <StatCard
          title="Mean confidence"
          value={stats ? `${Math.round(stats.mean_confidence * 100)}%` : loading ? '…' : '—'}
          subtext={
            stats?.val_sensitivity
              ? `Sensitivity at 2 FP/scan: ${(stats.val_sensitivity * 100).toFixed(1)}%`
              : 'Derived from cohort records'
          }
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {stats ? (
            <GradeDistribution distribution={stats.grade_distribution} total={stats.total_patients} />
          ) : (
            <div className="card-surface p-6 h-64 flex items-center justify-center text-xs text-gray-400">
              {loading ? 'Loading distribution…' : 'No cohort data available.'}
            </div>
          )}
        </div>

        <div className="card-surface p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold tracking-tight text-textMain">Management guide</h2>
              <span className="text-[10px] font-mono bg-teal-50 text-teal-800 px-2 py-0.5 rounded border border-teal-200">
                Lung-RADS v1.1
              </span>
            </div>
            <p className="text-xs text-gray-500 mb-4 leading-relaxed">
              Each finding combines an ordinal malignancy rating (1–5) with the measured diameter
              to select the appropriate follow-up pathway.
            </p>
            <dl className="text-[11px] space-y-2 text-gray-600">
              <div className="flex items-center justify-between py-1 border-b border-gray-100">
                <dt className="font-medium text-slate-700">Grade 0</dt>
                <dd className="text-gray-500">No action</dd>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-gray-100">
                <dt className="font-medium text-emerald-700">Grade I</dt>
                <dd className="text-gray-500">Annual screening</dd>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-gray-100">
                <dt className="font-medium text-amber-700">Grade II</dt>
                <dd className="text-gray-500">6-month CT</dd>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-gray-100">
                <dt className="font-medium text-orange-700">Grade III</dt>
                <dd className="text-gray-500">3-month CT or PET</dd>
              </div>
              <div className="flex items-center justify-between py-1">
                <dt className="font-medium text-rose-700">Grade IV</dt>
                <dd className="text-gray-500">Biopsy / contrast CT</dd>
              </div>
            </dl>
          </div>
          <p className="mt-4 pt-3 border-t border-border text-[11px] text-gray-400">
            Thresholds follow the Lung-RADS v1.1 size criteria.
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold tracking-tight text-textMain">Recent scans</h2>
            <p className="text-xs text-gray-500">Latest reviewed patients in the screening queue</p>
          </div>
          <Link href="/patients" className="text-xs font-medium text-accent hover:text-accent-hover transition-colors">
            View all ({patients.length}) →
          </Link>
        </div>

        <PatientTable patients={patients} limit={6} showSearch={false} />
      </div>
    </div>
  );
}
