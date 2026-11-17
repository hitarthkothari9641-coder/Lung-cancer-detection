'use client';

import React, { useEffect, useState } from 'react';
import { fetchPatients, Patient } from '../../lib/api';
import PatientTable from '../../components/PatientTable';

const GRADE_FILTERS = ['all', '0', 'I', 'II', 'III', 'IV'] as const;

export default function PatientsDirectoryPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [filterGrade, setFilterGrade] = useState<string>('all');

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setPatients(await fetchPatients());
      } catch (err: any) {
        setError(err.message || 'Unable to load the patient directory.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const filteredPatients =
    filterGrade === 'all' ? patients : patients.filter((p) => p.model_grade === filterGrade);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-textMain">
            Patient directory
          </h1>
          <p className="text-xs text-gray-500 mt-1">
            Search and review nodule findings for each screened patient.
          </p>
        </div>

        <div className="flex items-center gap-1 bg-surface border border-border p-1 rounded-md text-xs">
          {GRADE_FILTERS.map((g) => (
            <button
              key={g}
              type="button"
              onClick={() => setFilterGrade(g)}
              className={`px-2.5 py-1 rounded transition-colors ${
                filterGrade === g
                  ? 'bg-accent text-white font-medium'
                  : 'text-gray-600 hover:text-textMain hover:bg-gray-50'
              }`}
            >
              {g === 'all' ? 'All grades' : `Grade ${g}`}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-md text-xs text-rose-700">
          {error}
        </div>
      )}

      {loading ? (
        <div className="card-surface p-12 text-center text-xs text-gray-400">
          Loading patient directory…
        </div>
      ) : (
        <PatientTable patients={filteredPatients} showSearch={true} />
      )}
    </div>
  );
}
