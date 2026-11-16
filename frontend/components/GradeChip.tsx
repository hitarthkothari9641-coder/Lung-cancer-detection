import React from 'react';

interface GradeChipProps {
  grade: string | number;
  labelSource?: string;
  showProxyNote?: boolean;
  className?: string;
}

export default function GradeChip({
  grade,
  labelSource = 'proxy',
  showProxyNote = true,
  className = '',
}: GradeChipProps) {
  const gStr = String(grade).toUpperCase();

  // Color mapping based on design specification
  let colorClasses = 'bg-slate-100 text-slate-600';
  let romanText = gStr;

  if (gStr === '0' || gStr === 'GRADE 0') {
    colorClasses = 'bg-slate-100 text-slate-600';
    romanText = 'Grade 0';
  } else if (gStr === 'I' || gStr === '1' || gStr === 'GRADE I') {
    colorClasses = 'bg-emerald-100 text-emerald-700';
    romanText = 'Grade I';
  } else if (gStr === 'II' || gStr === '2' || gStr === 'GRADE II') {
    colorClasses = 'bg-amber-100 text-amber-700';
    romanText = 'Grade II';
  } else if (gStr === 'III' || gStr === '3' || gStr === 'GRADE III') {
    colorClasses = 'bg-orange-100 text-orange-700';
    romanText = 'Grade III';
  } else if (gStr === 'IV' || gStr === '4' || gStr === 'GRADE IV') {
    colorClasses = 'bg-rose-100 text-rose-700';
    romanText = 'Grade IV';
  }

  const isProxy = labelSource.toLowerCase() === 'proxy';

  return (
    <div className={`inline-flex items-center gap-1.5 ${className}`}>
      <span
        className={`inline-flex items-center rounded-full text-xs font-medium px-2.5 py-0.5 tracking-tight ${colorClasses}`}
      >
        {romanText}
      </span>
      {isProxy && showProxyNote && (
        <span
          title="Label derived from diameter size bucket (<6mm, 6-8, 8-15, 15-20, >20) in absence of LIDC-IDRI XML."
          className="text-[10px] text-gray-400 bg-gray-100 border border-gray-200 px-1.5 py-0.2 rounded font-normal"
        >
          proxy labels
        </span>
      )}
    </div>
  );
}
