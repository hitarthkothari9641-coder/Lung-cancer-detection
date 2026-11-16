"""
test_grading.py - Comprehensive Unit Tests for Clinical Rule Overlay and Lung-RADS Mapping.
"""

import pytest
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.grading import malignancy_to_grade, GRADE_ROMAN_MAP, GRADE_DETAILS


def test_grade_zero_no_nodule():
    """Verify absence of nodule produces Grade 0 and 'no action'."""
    grade, label, lung_rads, rec = malignancy_to_grade(score=0, diameter_mm=0.0)
    assert grade == 0
    assert label == "No nodule"
    assert "Lung-RADS 1" in lung_rads
    assert rec == "no action"

    # Negative diameter or score <= 0 should also return Grade 0
    g2, _, _, _ = malignancy_to_grade(score=-1, diameter_mm=5.0)
    assert g2 == 0
    g3, _, _, _ = malignancy_to_grade(score=2, diameter_mm=0.0)
    assert g3 == 0


def test_small_nodules_under_6mm():
    """Verify nodules < 6 mm: score <=2 -> Grade I; score >=3 -> Grade II."""
    # Low score
    g1, label1, rads1, rec1 = malignancy_to_grade(score=1, diameter_mm=4.5)
    assert g1 == 1
    assert label1 == "Benign/Very low"
    assert rec1 == "annual screening"

    g2, label2, _, _ = malignancy_to_grade(score=2, diameter_mm=5.9)
    assert g2 == 1
    assert label2 == "Benign/Very low"

    # Suspicious small nodule
    g3, label3, rads3, rec3 = malignancy_to_grade(score=3, diameter_mm=5.0)
    assert g3 == 2
    assert label3 == "Probably benign"
    assert "Lung-RADS 3" in rads3
    assert rec3 == "6-month follow-up CT"

    g4, label4, _, _ = malignancy_to_grade(score=5, diameter_mm=3.2)
    assert g4 == 2


def test_intermediate_nodules_6_to_8mm():
    """Verify nodules 6-8 mm: score <=2 -> Grade I; score=3 -> Grade II; score >=4 -> Grade III."""
    g1, _, _, _ = malignancy_to_grade(score=2, diameter_mm=6.0)
    assert g1 == 1

    g2, label2, rads2, rec2 = malignancy_to_grade(score=3, diameter_mm=7.5)
    assert g2 == 2
    assert "Lung-RADS 3" in rads2
    assert rec2 == "6-month follow-up CT"

    g3, label3, rads3, rec3 = malignancy_to_grade(score=4, diameter_mm=6.8)
    assert g3 == 3
    assert label3 == "Suspicious"
    assert "Lung-RADS 4A" in rads3
    assert rec3 == "3-month CT or PET/CT"

    g4, _, _, _ = malignancy_to_grade(score=5, diameter_mm=7.9)
    assert g4 == 3


def test_large_nodules_8_to_15mm():
    """Verify nodules 8-15 mm: score <=2 -> Grade II; score 3-4 -> Grade III; score=5 -> Grade IV."""
    g1, _, _, _ = malignancy_to_grade(score=2, diameter_mm=9.0)
    assert g1 == 2

    g2, label2, rads2, rec2 = malignancy_to_grade(score=3, diameter_mm=12.0)
    assert g2 == 3
    assert "Lung-RADS 4A" in rads2

    g3, _, _, _ = malignancy_to_grade(score=4, diameter_mm=14.9)
    assert g3 == 3

    g4, label4, rads4, rec4 = malignancy_to_grade(score=5, diameter_mm=13.5)
    assert g4 == 4
    assert label4 == "Highly suspicious"
    assert "Lung-RADS 4B/4X" in rads4
    assert rec4 == "contrast CT, PET/CT, biopsy"


def test_very_large_nodules_above_15mm():
    """Verify nodules >= 15 mm: score <=2 -> Grade III; score >=3 -> Grade IV."""
    g1, label1, rads1, rec1 = malignancy_to_grade(score=2, diameter_mm=18.0)
    assert g1 == 3
    assert "Lung-RADS 4A" in rads1

    g2, label2, rads2, rec2 = malignancy_to_grade(score=3, diameter_mm=16.0)
    assert g2 == 4
    assert label2 == "Highly suspicious"
    assert "Lung-RADS 4B/4X" in rads2

    g3, _, _, _ = malignancy_to_grade(score=5, diameter_mm=25.0)
    assert g3 == 4


def test_spot_check_twenty_pairs():
    """Spot check 20 distinct (score, diameter) combinations."""
    test_cases = [
        (0, 0.0, 0),
        (1, 2.5, 1),
        (2, 4.8, 1),
        (3, 3.5, 2),
        (4, 5.2, 2),
        (1, 6.2, 1),
        (2, 7.1, 1),
        (3, 6.9, 2),
        (4, 7.8, 3),
        (5, 7.0, 3),
        (1, 8.5, 2),
        (2, 10.0, 2),
        (3, 11.2, 3),
        (4, 13.0, 3),
        (5, 14.5, 4),
        (1, 16.0, 3),
        (2, 20.0, 3),
        (3, 15.5, 4),
        (4, 22.0, 4),
        (5, 28.0, 4),
    ]

    for score, dia, expected_grade in test_cases:
        actual_grade, _, _, _ = malignancy_to_grade(score, dia)
        assert actual_grade == expected_grade, f"Mismatch for score={score}, dia={dia}: got {actual_grade}, expected {expected_grade}"
