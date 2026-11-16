"""
scripts/make_sample_data.py
Generates deterministic, clinically consistent sample patients JSON for demonstration and testing.
"""

import json
import os
import sys

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.grading import malignancy_to_grade, GRADE_ROMAN_MAP

SAMPLE_PATIENTS_RAW = [
    {
        "id": "PAT-1001",
        "name": "Eleanor Vance",
        "age": 62,
        "sex": "Female",
        "smoking_pack_years": 0,
        "scan_date": "2026-09-14",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1001",
        "nodule_size_mm": 0.0,
        "location": "None",
        "malignancy_score": 0,
        "confidence": 0.96,
        "status": "Screening",
    },
    {
        "id": "PAT-1002",
        "name": "David Miller",
        "age": 54,
        "sex": "Male",
        "smoking_pack_years": 12,
        "scan_date": "2026-09-18",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1002",
        "nodule_size_mm": 0.0,
        "location": "None",
        "malignancy_score": 0,
        "confidence": 0.94,
        "status": "Screening",
    },
    {
        "id": "PAT-1003",
        "name": "Arthur Pendelton",
        "age": 71,
        "sex": "Male",
        "smoking_pack_years": 25,
        "scan_date": "2026-09-22",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1003",
        "nodule_size_mm": 4.2,
        "location": "Right Upper Lobe",
        "malignancy_score": 1,
        "confidence": 0.88,
        "status": "Benign",
    },
    {
        "id": "PAT-1004",
        "name": "Beatrice Thorne",
        "age": 59,
        "sex": "Female",
        "smoking_pack_years": 15,
        "scan_date": "2026-10-01",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1004",
        "nodule_size_mm": 5.1,
        "location": "Left Lower Lobe",
        "malignancy_score": 2,
        "confidence": 0.84,
        "status": "Benign",
    },
    {
        "id": "PAT-1005",
        "name": "Marcus Aurelius Chen",
        "age": 66,
        "sex": "Male",
        "smoking_pack_years": 30,
        "scan_date": "2026-10-05",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1005",
        "nodule_size_mm": 6.8,
        "location": "Right Middle Lobe",
        "malignancy_score": 2,
        "confidence": 0.79,
        "status": "Screening",
    },
    {
        "id": "PAT-1006",
        "name": "Sarah Jenkins",
        "age": 48,
        "sex": "Female",
        "smoking_pack_years": 8,
        "scan_date": "2026-10-10",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1006",
        "nodule_size_mm": 5.8,
        "location": "Left Upper Lobe",
        "malignancy_score": 3,
        "confidence": 0.72,
        "status": "Follow-up",
    },
    {
        "id": "PAT-1007",
        "name": "Robert Sterling",
        "age": 68,
        "sex": "Male",
        "smoking_pack_years": 42,
        "scan_date": "2026-10-14",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1007",
        "nodule_size_mm": 7.4,
        "location": "Right Lower Lobe",
        "malignancy_score": 3,
        "confidence": 0.76,
        "status": "Follow-up",
    },
    {
        "id": "PAT-1008",
        "name": "Clara Oswald",
        "age": 52,
        "sex": "Female",
        "smoking_pack_years": 20,
        "scan_date": "2026-10-19",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1008",
        "nodule_size_mm": 7.9,
        "location": "Right Upper Lobe",
        "malignancy_score": 4,
        "confidence": 0.81,
        "status": "Follow-up",
    },
    {
        "id": "PAT-1009",
        "name": "Jonathan Archer",
        "age": 74,
        "sex": "Male",
        "smoking_pack_years": 50,
        "scan_date": "2026-10-24",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1009",
        "nodule_size_mm": 11.2,
        "location": "Left Upper Lobe",
        "malignancy_score": 3,
        "confidence": 0.85,
        "status": "Follow-up",
    },
    {
        "id": "PAT-1010",
        "name": "Evelyn Reed",
        "age": 60,
        "sex": "Female",
        "smoking_pack_years": 35,
        "scan_date": "2026-10-29",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1010",
        "nodule_size_mm": 16.5,
        "location": "Left Lower Lobe",
        "malignancy_score": 2,
        "confidence": 0.83,
        "status": "Follow-up",
    },
    {
        "id": "PAT-1011",
        "name": "Walter White",
        "age": 78,
        "sex": "Male",
        "smoking_pack_years": 58,
        "scan_date": "2026-11-02",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1011",
        "nodule_size_mm": 13.8,
        "location": "Right Upper Lobe",
        "malignancy_score": 5,
        "confidence": 0.92,
        "status": "Biopsy",
    },
    {
        "id": "PAT-1012",
        "name": "Helena Rostova",
        "age": 65,
        "sex": "Female",
        "smoking_pack_years": 45,
        "scan_date": "2026-11-08",
        "seriesuid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.1012",
        "nodule_size_mm": 22.4,
        "location": "Right Lower Lobe",
        "malignancy_score": 4,
        "confidence": 0.95,
        "status": "Biopsy",
    },
]


def generate_sample_patients():
    """Generates 12 fictional patients strictly verified by grading.py."""
    patients = []
    grade_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for item in SAMPLE_PATIENTS_RAW:
        score = item["malignancy_score"]
        dia = item["nodule_size_mm"]
        grade, label, lung_rads, rec = malignancy_to_grade(score, dia)

        grade_counts[grade] += 1
        roman = GRADE_ROMAN_MAP[grade]

        patient_record = {
            "id": item["id"],
            "name": item["name"],
            "age": item["age"],
            "sex": item["sex"],
            "smoking_pack_years": item["smoking_pack_years"],
            "scan_date": item["scan_date"],
            "seriesuid": item["seriesuid"],
            "nodule_size_mm": dia,
            "location": item["location"],
            "malignancy_score": score,
            "model_grade": roman,
            "model_grade_int": grade,
            "grade_label": label,
            "lung_rads": lung_rads,
            "confidence": round(item["confidence"], 2),
            "recommendation": rec,
            "status": item["status"],
            "disclaimer": "Research prototype. Not a medical device. Not for clinical diagnosis.",
        }
        patients.append(patient_record)

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sample_patients.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)

    print(f"Generated {len(patients)} sample patients at: {out_path}")
    print(f"Grade distribution: {grade_counts}")


if __name__ == "__main__":
    generate_sample_patients()
