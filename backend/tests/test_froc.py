"""
test_froc.py - Unit Tests for FROC Sensitivity Calculation and QWK Metric.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "training")))

from evaluate import compute_froc, compute_qwk, compute_calibration_ece


def test_froc_monotonic_sensitivity():
    """Verify FROC sensitivity increases or remains constant as allowed FP rate increases."""
    # Synthetic ground truth: 10 positives, 90 negatives
    np.random.seed(42)
    y_true = np.array([1] * 10 + [0] * 90)
    # Give higher scores to positives on average
    y_scores = np.concatenate([
        np.random.normal(0.8, 0.1, 10),
        np.random.normal(0.2, 0.15, 90),
    ])

    froc_res = compute_froc(y_true, y_scores, num_scans=10, fp_rates=[1.0, 2.0, 4.0, 8.0])
    sens = froc_res["sensitivities"]

    s1 = sens["1.0"]
    s2 = sens["2.0"]
    s4 = sens["4.0"]
    s8 = sens["8.0"]

    assert 0.0 <= s1 <= s2 <= s4 <= s8 <= 1.0
    assert froc_res["average_sensitivity"] == round(float(np.mean([s1, s2, s4, s8])), 4)


def test_froc_empty_or_zero_positives():
    """Verify FROC calculation handles zero positives gracefully without division by zero."""
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4])
    froc_res = compute_froc(y_true, y_scores, num_scans=2)
    assert froc_res["average_sensitivity"] == 0.0
    for v in froc_res["sensitivities"].values():
        assert v == 0.0


def test_qwk_perfect_and_imperfect():
    """Verify Quadratic Weighted Kappa yields 1.0 on exact agreement."""
    y_true = np.array([1, 2, 3, 4, 5, 2, 3, 4])
    y_pred_perfect = np.array([1, 2, 3, 4, 5, 2, 3, 4])
    qwk_perfect = compute_qwk(y_true, y_pred_perfect)
    assert qwk_perfect == 1.0

    # Slight off-by-one errors should give high but <1.0 kappa
    y_pred_slight = np.array([1, 2, 4, 4, 5, 2, 3, 3])
    qwk_slight = compute_qwk(y_true, y_pred_slight)
    assert 0.70 < qwk_slight < 1.0


def test_calibration_ece_bounds():
    """Verify ECE is bounded in [0, 1]."""
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.3, 0.15])
    ece, bin_data = compute_calibration_ece(y_true, y_probs, num_bins=10)
    assert 0.0 <= ece <= 1.0
    assert len(bin_data) == 10
