import React from 'react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtext?: string;
  badge?: string;
  badgeColor?: 'default' | 'teal' | 'amber' | 'rose';
}

export default function StatCard({
  title,
  value,
  subtext,
  badge,
  badgeColor = 'default',
}: StatCardProps) {
  const badgeClasses = {
    default: 'bg-gray-100 text-gray-600',
    teal: 'bg-teal-50 text-teal-700 border border-teal-200',
    amber: 'bg-amber-50 text-amber-700 border border-amber-200',
    rose: 'bg-rose-50 text-rose-700 border border-rose-200',
  }[badgeColor];

  return (
    <div className="card-surface p-5 flex flex-col justify-between">
      <div className="flex items-center justify-between text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
        <span>{title}</span>
        {badge && (
          <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${badgeClasses}`}>
            {badge}
          </span>
        )}
      </div>
      <div className="text-2xl sm:text-3xl font-semibold text-textMain tracking-tight">
        {value}
      </div>
      {subtext && (
        <div className="text-xs text-gray-500 mt-2 font-normal">
          {subtext}
        </div>
      )}
    </div>
  );
}
