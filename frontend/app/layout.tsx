import React from 'react';
import Link from 'next/link';
import './globals.css';
import Disclaimer from '../components/Disclaimer';

export const metadata = {
  title: '3D Lung Nodule Detector & Cancer Grader',
  description: 'Offline 3D Deep Learning pipeline for pulmonary nodule detection and Lung-RADS cancer grading.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-background text-textMain min-h-screen flex flex-col antialiased">
        {/* Minimalist Top Navigation */}
        <header className="bg-surface border-b border-border sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link href="/" className="flex items-center gap-2.5">
                <span className="w-8 h-8 rounded-lg bg-accent text-white flex items-center justify-center font-bold text-sm shadow-sm">
                  LN
                </span>
                <div>
                  <span className="font-semibold text-sm tracking-tight text-textMain block">
                    Lung Nodule Grader
                  </span>
                  <span className="text-[10px] text-gray-500 font-mono block -mt-0.5">
                    3D DenseNet • LUNA16
                  </span>
                </div>
              </Link>

              <nav className="hidden md:flex items-center gap-1 text-xs font-medium text-gray-600">
                <Link
                  href="/"
                  className="px-3 py-1.5 rounded-md hover:text-accent hover:bg-gray-50 transition-colors"
                >
                  Dashboard
                </Link>
                <Link
                  href="/patients"
                  className="px-3 py-1.5 rounded-md hover:text-accent hover:bg-gray-50 transition-colors"
                >
                  Patients
                </Link>
                <Link
                  href="/upload"
                  className="px-3 py-1.5 rounded-md hover:text-accent hover:bg-gray-50 transition-colors"
                >
                  Upload Scan
                </Link>
              </nav>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-xs text-gray-500 bg-gray-50 border border-border px-2.5 py-1 rounded-full">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="font-mono text-[11px]">OFFLINE RUNTIME</span>
              </div>
              <Link
                href="/upload"
                className="bg-accent hover:bg-accent-hover text-white text-xs font-medium px-3.5 py-1.5 rounded-md shadow-sm transition-colors hidden sm:inline-block"
              >
                + New Prediction
              </Link>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">
          {children}
        </main>

        {/* Global Mandatory Regulatory Disclaimer */}
        <Disclaimer />
      </body>
    </html>
  );
}
