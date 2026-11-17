import React from 'react';

interface GradeChipProps {
  grade: string | number;
  className?: string;
}

const GRADE_STYLES: Record<string, { label: string; classes: string }> = {
  '0': { label: 'Grade 0', classes: 'bg-slate-100 text-slate-700 ring-slate-200' },
  '1': { label: 'Grade I', classes: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
  I: { label: 'Grade I', classes: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
  '2': { label: 'Grade II', classes: 'bg-amber-50 text-amber-700 ring-amber-200' },
  II: { label: 'Grade II', classes: 'bg-amber-50 text-amber-700 ring-amber-200' },
  '3': { label: 'Grade III', classes: 'bg-orange-50 text-orange-700 ring-orange-200' },
  III: { label: 'Grade III', classes: 'bg-orange-50 text-orange-700 ring-orange-200' },
  '4': { label: 'Grade IV', classes: 'bg-rose-50 text-rose-700 ring-rose-200' },
  IV: { label: 'Grade IV', classes: 'bg-rose-50 text-rose-700 ring-rose-200' },
};

export default function GradeChip({ grade, className = '' }: GradeChipProps) {
  const key = String(grade).toUpperCase().replace('GRADE', '').trim();
  const style = GRADE_STYLES[key] ?? GRADE_STYLES['0'];

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${style.classes} ${className}`}
    >
      {style.label}
    </span>
  );
}
