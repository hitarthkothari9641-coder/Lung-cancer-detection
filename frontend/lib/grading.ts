/**
 * frontend/lib/grading.ts
 *
 * Exact frontend mirror of backend/app/grading.py.
 * Ensures client-side rendering and rule evaluations are 100% synchronized with backend.
 *
 * DISCLAIMER:
 * Research prototype. Not a medical device. Not for clinical diagnosis.
 */

export type GradeRoman = '0' | 'I' | 'II' | 'III' | 'IV';

export interface GradeInfo {
  grade: number;
  roman: GradeRoman;
  label: string;
  lungRads: string;
  recommendation: string;
}

export const GRADE_DETAILS: Record<number, GradeInfo> = {
  0: {
    grade: 0,
    roman: '0',
    label: 'No nodule',
    lungRads: 'Lung-RADS 1',
    recommendation: 'no action',
  },
  1: {
    grade: 1,
    roman: 'I',
    label: 'Benign/Very low',
    lungRads: 'Lung-RADS 1–2',
    recommendation: 'annual screening',
  },
  2: {
    grade: 2,
    roman: 'II',
    label: 'Probably benign',
    lungRads: 'Lung-RADS 3',
    recommendation: '6-month follow-up CT',
  },
  3: {
    grade: 3,
    roman: 'III',
    label: 'Suspicious',
    lungRads: 'Lung-RADS 4A',
    recommendation: '3-month CT or PET/CT',
  },
  4: {
    grade: 4,
    roman: 'IV',
    label: 'Highly suspicious',
    lungRads: 'Lung-RADS 4B/4X',
    recommendation: 'contrast CT, PET/CT, biopsy',
  },
};

/**
 * Converts model predicted malignancy score (1..5) and nodule diameter (mm)
 * into a clinical grade (0-4), Lung-RADS category, and actionable recommendation.
 */
export function malignancyToGrade(score: number, diameterMm: number): GradeInfo {
  if (score <= 0 || diameterMm <= 0.0) {
    return GRADE_DETAILS[0];
  }

  const clampedScore = Math.max(1, Math.min(5, Math.floor(score)));
  const dia = Number(diameterMm);

  let grade = 0;

  if (dia < 6.0) {
    grade = clampedScore <= 2 ? 1 : 2;
  } else if (dia >= 6.0 && dia < 8.0) {
    if (clampedScore <= 2) {
      grade = 1;
    } else if (clampedScore === 3) {
      grade = 2;
    } else {
      grade = 3;
    }
  } else if (dia >= 8.0 && dia < 15.0) {
    if (clampedScore <= 2) {
      grade = 2;
    } else if (clampedScore <= 4) {
      grade = 3;
    } else {
      grade = 4;
    }
  } else {
    // dia >= 15.0
    grade = clampedScore <= 2 ? 3 : 4;
  }

  return GRADE_DETAILS[grade];
}
