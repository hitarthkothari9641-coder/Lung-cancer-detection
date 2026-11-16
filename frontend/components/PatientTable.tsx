'use client';

import React, { useState, useMemo } from 'react';
import Link from 'next/link';
import { Patient } from '../lib/api';
import GradeChip from './GradeChip';

interface PatientTableProps {
  patients: Patient[];
  labelSource?: string;
  limit?: number;
  showSearch?: boolean;
}

type SortField = 'name' | 'age' | 'smoking_pack_years' | 'nodule_size_mm' | 'model_grade_int';

export default function PatientTable({
  patients,
  labelSource = 'proxy',
  limit,
  showSearch = true,
}: PatientTableProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<SortField>('model_grade_int');
  const [sortAsc, setSortAsc] = useState<boolean>(false); // default descending (highest risk first)

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(field === 'name');
    }
  };

  const filteredAndSorted = useMemo(() => {
    let result = [...patients];

    if (searchTerm.trim()) {
      const q = searchTerm.toLowerCase();
      result = result.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.id.toLowerCase().includes(q) ||
          p.location.toLowerCase().includes(q) ||
          p.lung_rads.toLowerCase().includes(q)
      );
    }

    result.sort((a, b) => {
      let valA: any = a[sortField];
      let valB: any = b[sortField];

      if (typeof valA === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortAsc ? valA - valB : valB - valA;
    });

    if (limit && limit > 0) {
      result = result.slice(0, limit);
    }

    return result;
  }, [patients, searchTerm, sortField, sortAsc, limit]);

  return (
    <div className="card-surface overflow-hidden">
      {showSearch && (
        <div className="p-4 border-b border-border flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="relative w-full sm:w-72">
            <input
              type="text"
              placeholder="Search by name, ID, or lobe..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full text-xs bg-gray-50 border border-border rounded-md px-3 py-2 pl-8 focus:outline-none focus:border-accent text-textMain"
            />
            <svg
              className="w-4 h-4 text-gray-400 absolute left-2.5 top-2.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <div className="text-xs text-gray-500 font-mono">
            Showing {filteredAndSorted.length} of {patients.length} patients
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-gray-50/75 border-b border-border text-gray-500 font-medium select-none">
              <th
                onClick={() => handleSort('name')}
                className="py-3 px-4 cursor-pointer hover:text-textMain transition-colors"
              >
                Patient Name / ID {sortField === 'name' && (sortAsc ? '▲' : '▼')}
              </th>
              <th
                onClick={() => handleSort('age')}
                className="py-3 px-4 cursor-pointer hover:text-textMain transition-colors"
              >
                Age / Sex {sortField === 'age' && (sortAsc ? '▲' : '▼')}
              </th>
              <th
                onClick={() => handleSort('smoking_pack_years')}
                className="py-3 px-4 cursor-pointer hover:text-textMain transition-colors"
              >
                Smoking {sortField === 'smoking_pack_years' && (sortAsc ? '▲' : '▼')}
              </th>
              <th
                onClick={() => handleSort('nodule_size_mm')}
                className="py-3 px-4 cursor-pointer hover:text-textMain transition-colors"
              >
                Size (mm) {sortField === 'nodule_size_mm' && (sortAsc ? '▲' : '▼')}
              </th>
              <th
                onClick={() => handleSort('model_grade_int')}
                className="py-3 px-4 cursor-pointer hover:text-textMain transition-colors"
              >
                Grade {sortField === 'model_grade_int' && (sortAsc ? '▲' : '▼')}
              </th>
              <th className="py-3 px-4">Lung-RADS Category</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filteredAndSorted.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-gray-400">
                  No matching patients found.
                </td>
              </tr>
            ) : (
              filteredAndSorted.map((p) => (
                <tr
                  key={p.id}
                  className="hover:bg-teal-50/30 transition-colors group cursor-pointer"
                >
                  <td className="py-3.5 px-4 font-medium text-textMain">
                    <Link href={`/patients/${p.id}`} className="block">
                      <div className="font-semibold text-gray-900 group-hover:text-accent">
                        {p.name}
                      </div>
                      <div className="text-[11px] text-gray-400 font-mono">{p.id}</div>
                    </Link>
                  </td>
                  <td className="py-3.5 px-4 text-gray-600">
                    {p.age} yo / {p.sex}
                  </td>
                  <td className="py-3.5 px-4 text-gray-600">
                    {p.smoking_pack_years} pk-yrs
                  </td>
                  <td className="py-3.5 px-4 font-mono font-medium text-gray-700">
                    {p.nodule_size_mm > 0 ? `${p.nodule_size_mm.toFixed(1)} mm` : 'None'}
                  </td>
                  <td className="py-3.5 px-4">
                    <GradeChip grade={p.model_grade} labelSource={labelSource} />
                  </td>
                  <td className="py-3.5 px-4 text-gray-700 font-medium">
                    {p.lung_rads}
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <Link
                      href={`/patients/${p.id}`}
                      className="inline-flex items-center text-xs font-medium text-accent hover:text-accent-hover transition-colors"
                    >
                      View Scan →
                    </Link>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
