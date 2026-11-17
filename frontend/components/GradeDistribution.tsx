import React from 'react';

interface GradeDistributionProps {
  distribution: Record<string, number>;
  total?: number;
  className?: string;
}

const GRADE_METADATA = [
  { key: '0', label: 'No nodule', color: '#94A3B8' },
  { key: 'I', label: 'Grade I · benign', color: '#10B981' },
  { key: 'II', label: 'Grade II · probably benign', color: '#F59E0B' },
  { key: 'III', label: 'Grade III · suspicious', color: '#F97316' },
  { key: 'IV', label: 'Grade IV · highly suspicious', color: '#F43F5E' },
];

export default function GradeDistribution({
  distribution,
  total,
  className = '',
}: GradeDistributionProps) {
  const sum = total || Object.values(distribution).reduce((acc, v) => acc + v, 0) || 1;

  return (
    <div className={`card-surface p-5 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-textMain tracking-tight">Grade Distribution</h3>
          <p className="text-xs text-gray-500">Cohort triage risk stratification (Grades 0–IV)</p>
        </div>
        <span className="text-xs text-gray-500 font-mono">N = {sum}</span>
      </div>

      <div className="space-y-3">
        {GRADE_METADATA.map((meta) => {
          const count = distribution[meta.key] || 0;
          const percentage = Math.round((count / sum) * 100);

          return (
            <div key={meta.key} className="flex flex-col gap-1">
              <div className="flex justify-between items-center text-xs">
                <span className="font-medium text-gray-700">{meta.label}</span>
                <span className="text-gray-500 font-mono">
                  {count} ({percentage}%)
                </span>
              </div>
              {/* Pure SVG Bar */}
              <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
                <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
                  <rect
                    x="0"
                    y="0"
                    width={`${percentage}%`}
                    height="100%"
                    fill={meta.color}
                    rx="4"
                  />
                </svg>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
