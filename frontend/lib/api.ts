/**
 * frontend/lib/api.ts
 *
 * Typed client for the FastAPI backend.
 *
 * Requests default to same-origin `/api/*`, which `next.config.js` rewrites to
 * the Python service. That works for local runs, Docker, and hosted previews
 * without CORS configuration. Set NEXT_PUBLIC_API_URL only to point the browser
 * at a different host entirely.
 */

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || '').replace(/\/$/, '');

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
  has_nodule?: boolean;
  patient?: Record<string, unknown>;
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
  model_loaded: boolean;
  label_source: string;
  disclaimer: string;
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res;
}

export async function fetchHealth(): Promise<HealthInfo> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    return {
      status: 'unavailable',
      model_loaded: false,
      label_source: 'proxy',
      disclaimer: 'Research prototype. Not a medical device. Not for clinical diagnosis.',
    };
  }
}

export async function fetchPatients(): Promise<Patient[]> {
  const res = await request('/api/patients', { cache: 'no-store' });
  return res.json();
}

export async function fetchStats(): Promise<AggregateStats> {
  const res = await request('/api/stats', { cache: 'no-store' });
  return res.json();
}

export async function predictScan(
  seriesuid: string,
  patientMeta?: Record<string, unknown>,
): Promise<PredictionResult> {
  const res = await request('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seriesuid, patient: patientMeta }),
  });
  return res.json();
}

export async function uploadScanFile(formData: FormData): Promise<PredictionResult> {
  const res = await request('/api/upload', { method: 'POST', body: formData });
  return res.json();
}

export function getReportDownloadUrl(patientId: string): string {
  return `${API_BASE}/api/report/${encodeURIComponent(patientId)}`;
}
