import React from 'react';

interface DisclaimerProps {
  className?: string;
}

export default function Disclaimer({ className = '' }: DisclaimerProps) {
  return (
    <footer className={`border-t border-border bg-surface py-4 px-6 text-center ${className}`}>
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between text-xs text-gray-500 gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-amber-500"></span>
          <span className="font-medium text-gray-700">Notice:</span>
          <span>Research prototype. Not a medical device. Not for clinical diagnosis.</span>
        </div>
        <div className="text-gray-400">
          Strictly for retrospective research and benchmarking on LUNA16 & LIDC cohorts.
        </div>
      </div>
    </footer>
  );
}
