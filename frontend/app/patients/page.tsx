'use client';

import React, { useEffect, useState } from 'react';
import { fetchPatients, fetchStats, Patient, AggregateStats } from '../../lib/api';
import PatientTable from '../../components/PatientTable';

export default function PatientsDirectoryPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [stats, setStats] = useState<AggregateStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterGrade, setFilterGrade] = useState<string>('all');

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [pts, st] = await Promise.all([
          fetchPatients().catch(() => []),
          fetchStats().catch(() => null),
        ]);
        setPatients(pts);
        setStats(st);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const filteredPatients = filterGrade === 'all'
    ? patients
    : patients.filter((p) => p.model_grade === filterGrade);

  return (
    <div className="space-y-6">
      {/* Directory Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-textMain tracking-tight">
            Patient Screening Directory
          </h1>
          <p className="text-xs text-gray-500 mt-1">
            Search, sort, and inspect 3D multi-task nodule triage and cancer grading records.
          </p>
        </div>

        {/* Grade Filters */}
        <div className="flex items-center gap-1.5 bg-surface border border-border p-1 rounded-md text-xs">
          {['all', '0', 'I', 'II', 'III', 'IV'].map((g) => (
            <button
              key={g}
              onClick={() => setFilterGrade(g)}
              className={`px-2.5 py-1 rounded transition-colors ${
                filterGrade === g
                  ? 'bg-accent text-white font-medium'
                  : 'text-gray-600 hover:text-textMain hover:bg-gray-50'
              }`}
            >
              {g === 'all' ? 'All Grades' : `Grade ${g}`}
            </button>
          ))}
        </div>
      </div>

      {/* Main Table */}
      {loading ? (
        <div className="card-surface p-12 text-center text-xs text-gray-400">
          Loading patient directory...
        </div>
      ) : (
        <PatientTable
          patients={filteredPatients}
          labelSource={stats?.label_source || 'proxy'}
          showSearch={true}
        />
      )}
    </div>
  );
}
