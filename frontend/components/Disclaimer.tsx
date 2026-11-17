import React from 'react';

interface DisclaimerProps {
  className?: string;
}

export default function Disclaimer({ className = '' }: DisclaimerProps) {
  return (
    <footer className={`border-t border-border bg-surface py-4 px-6 ${className}`}>
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-500">
        <div className="flex items-center gap-2 text-center sm:text-left">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
          <span>
            <span className="font-medium text-gray-700">Research use only.</span> Not a medical
            device and not intended for clinical diagnosis.
          </span>
        </div>
        <span className="text-gray-400">Evaluated retrospectively on LUNA16 / LIDC-IDRI cohorts.</span>
      </div>
    </footer>
  );
}
