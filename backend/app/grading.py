"""
grading.py - Clinical Rule Overlay and Lung-RADS Classification.

Single source of truth for converting model outputs (malignancy score 1-5, nodule diameter mm)
into clinical Lung-RADS categories and actionable oncologic grades (Grade 0 to Grade IV).

DISCLAIMER:
Research prototype. Not a medical device. Not for clinical diagnosis.
"""

from typing import Tuple, Dict, Any

GRADE_ROMAN_MAP: Dict[int, str] = {
    0: "0",
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
}

ROMAN_TO_GRADE: Dict[str, int] = {
    "0": 0,
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
}

GRADE_DETAILS: Dict[int, Dict[str, str]] = {
    0: {
        "label": "No nodule",
        "lung_rads": "Lung-RADS 1",
        "recommendation": "no action",
        "roman": "0",
    },
    1: {
        "label": "Benign/Very low",
        "lung_rads": "Lung-RADS 1–2",
        "recommendation": "annual screening",
        "roman": "I",
    },
    2: {
        "label": "Probably benign",
        "lung_rads": "Lung-RADS 3",
        "recommendation": "6-month follow-up CT",
        "roman": "II",
    },
    3: {
        "label": "Suspicious",
        "lung_rads": "Lung-RADS 4A",
        "recommendation": "3-month CT or PET/CT",
        "roman": "III",
    },
    4: {
        "label": "Highly suspicious",
        "lung_rads": "Lung-RADS 4B/4X",
        "recommendation": "contrast CT, PET/CT, biopsy",
        "roman": "IV",
    },
}


def malignancy_to_grade(score: int, diameter_mm: float) -> Tuple[int, str, str, str]:
    """
    Converts model predicted malignancy score (ordinal 1..5) and nodule diameter (mm)
    into a standardized clinical grade (0-4), label, Lung-RADS category, and recommendation.

    Args:
        score: Ordinal malignancy rating (1 to 5). 0 indicates absence of nodule.
        diameter_mm: Maximum axial diameter of the nodule in millimeters.

    Returns:
        tuple of (grade: int, label: str, lung_rads: str, recommendation: str)
        where grade is an integer from 0 to 4.
    """
    # No nodule detected or non-positive measurements
    if score <= 0 or diameter_mm <= 0.0:
        details = GRADE_DETAILS[0]
        return 0, details["label"], details["lung_rads"], details["recommendation"]

    # Clamp score to 1..5 range
    clamped_score = max(1, min(5, int(score)))
    dia = float(diameter_mm)

    if dia < 6.0:
        if clamped_score <= 2:
            grade = 1
        else:
            grade = 2
    elif 6.0 <= dia < 8.0:
        if clamped_score <= 2:
            grade = 1
        elif clamped_score == 3:
            grade = 2
        else:
            grade = 3
    elif 8.0 <= dia < 15.0:
        if clamped_score <= 2:
            grade = 2
        elif clamped_score <= 4:
            grade = 3
        else:
            grade = 4
    else:  # dia >= 15.0
        if clamped_score <= 2:
            grade = 3
        else:
            grade = 4

    details = GRADE_DETAILS[grade]
    return grade, details["label"], details["lung_rads"], details["recommendation"]


def get_grade_info(grade: int) -> Dict[str, str]:
    """Retrieve metadata and clinical definitions for a given grade (0-4)."""
    grade_clamped = max(0, min(4, int(grade)))
    return GRADE_DETAILS[grade_clamped]
