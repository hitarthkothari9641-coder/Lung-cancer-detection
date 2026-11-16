/**
 * frontend/lib/api.ts
 *
 * Lightweight API wrapper with offline fallback resilience.
 * Runtime must be offline: no external HTTP calls.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface NoduleItem {
  coord: [number, number, number];
  diameter_mm: number;
  prob: number;
}

export interface PredictionResult {
  seriesuid: string;
  nodules: NoduleItem[];
  top_nodule: NoduleItem | null;
  malignancy_score: number;
  grade: '0' | 'I' | 'II' | 'III' | 'IV';
  grade_int: number;
  grade_label: string;
  lung_rads: string;
  confidence: number;
  recommendation: string;
  gradcam_slice_png_b64: string;
  label_source: string;
  disclaimer: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  sex: string;
  smoking_pack_years: number;
  scan_date: string;
  seriesuid: string;
  nodule_size_mm: number;
  location: string;
  malignancy_score: number;
  model_grade: '0' | 'I' | 'II' | 'III' | 'IV';
  model_grade_int: number;
  grade_label: string;
  lung_rads: string;
  confidence: number;
  recommendation: string;
  status: 'Screening' | 'Follow-up' | 'Biopsy' | 'Benign';
  disclaimer: string;
}

export interface AggregateStats {
  total_patients: number;
  total_scans: number;
  nodules_detected: number;
  grade_distribution: Record<string, number>;
  mean_confidence: number;
  high_risk_pct: number;
  avg_nodules_per_scan: number;
  val_sensitivity: number | null;
  val_specificity: number | null;
  label_source: string;
  disclaimer: string;
}

export interface HealthInfo {
  status: string;
  model_loaded: bool;
  label_source: string;
  disclaimer: string;
}

export async function fetchHealth(): Promise<HealthInfo> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return {
      status: 'offline-mock',
      model_loaded: false,
      label_source: 'proxy',
      disclaimer: 'Research prototype. Not a medical device. Not for clinical diagnosis.',
    };
  }
}

export async function fetchPatients(): Promise<Patient[]> {
  const res = await fetch(`${API_BASE}/api/patients`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load patients: ${res.statusText}`);
  return await res.json();
}

export async function fetchStats(): Promise<AggregateStats> {
  const res = await fetch(`${API_BASE}/api/stats`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Failed to load statistics: ${res.statusText}`);
  return await res.json();
}

export async function predictScan(seriesuid: string, patientMeta?: Record<string, any>): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seriesuid, patient: patientMeta }),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return await res.json();
}

export async function uploadScanFile(formData: FormData): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`File upload inference failed: ${res.statusText}`);
  return await res.json();
}

export function getReportDownloadUrl(patientId: string): string {
  return `${API_BASE}/api/report/${patientId}`;
}
