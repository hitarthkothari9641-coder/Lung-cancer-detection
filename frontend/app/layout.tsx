import React from 'react';
import Link from 'next/link';
import './globals.css';
import Disclaimer from '../components/Disclaimer';

export const metadata = {
  title: 'Lung Nodule Screening',
  description: 'Pulmonary nodule detection and Lung-RADS severity grading for research cohorts.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-background text-textMain min-h-screen flex flex-col antialiased">
        <header className="bg-surface border-b border-border sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-8">
              <Link href="/" className="flex items-center gap-2.5">
                <span className="w-8 h-8 rounded-lg bg-accent text-white flex items-center justify-center font-semibold text-sm">
                  LN
                </span>
                <span className="font-semibold text-sm tracking-tight text-textMain">
                  Lung Nodule Screening
                </span>
              </Link>

              <nav className="hidden md:flex items-center gap-1 text-xs font-medium text-gray-600">
                <Link href="/" className="px-3 py-1.5 rounded-md hover:text-accent hover:bg-gray-50 transition-colors">
                  Overview
                </Link>
                <Link href="/patients" className="px-3 py-1.5 rounded-md hover:text-accent hover:bg-gray-50 transition-colors">
                  Patients
                </Link>
                <Link href="/upload" className="px-3 py-1.5 rounded-md hover:text-accent hover:bg-gray-50 transition-colors">
                  New Scan
                </Link>
              </nav>
            </div>

            <Link
              href="/upload"
              className="bg-accent hover:bg-accent-hover text-white text-xs font-medium px-3.5 py-1.5 rounded-md transition-colors"
            >
              Analyze a scan
            </Link>
          </div>
        </header>

        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">{children}</main>

        <Disclaimer />
      </body>
    </html>
  );
}
