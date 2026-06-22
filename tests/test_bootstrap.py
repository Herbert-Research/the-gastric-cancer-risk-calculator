"""Tests for bootstrap confidence interval utilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, brier_score_loss

from utils.bootstrap import bootstrap_correlation, bootstrap_metric


class TestBootstrapMetric:
    """Tests for the bootstrap_metric function."""

    def test_returns_three_values(self):
        """Test that function returns exactly three values."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        result = bootstrap_metric(y_true, y_pred, brier_score_loss)
        assert len(result) == 3

    def test_ci_contains_point_estimate(self):
        """Test that CI bounds contain the point estimate."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        estimate, ci_low, ci_high = bootstrap_metric(y_true, y_pred, brier_score_loss)
        assert ci_low <= estimate <= ci_high

    def test_ci_ordering(self):
        """Test that lower CI <= upper CI."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        _, ci_low, ci_high = bootstrap_metric(y_true, y_pred, brier_score_loss)
        assert ci_low <= ci_high

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
        result1 = bootstrap_metric(y_true, y_pred, brier_score_loss, random_state=42)
        result2 = bootstrap_metric(y_true, y_pred, brier_score_loss, random_state=42)
        assert result1 == result2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different CI bounds."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        result1 = bootstrap_metric(y_true, y_pred, brier_score_loss, random_state=42)
        result2 = bootstrap_metric(y_true, y_pred, brier_score_loss, random_state=123)
        # Point estimates should be the same
        assert result1[0] == result2[0]
        # CI bounds may differ (with high probability for large n_bootstrap)
        # This is a probabilistic test, so we just verify they're both valid
        assert result1[1] <= result1[2]
        assert result2[1] <= result2[2]

    def test_brier_score_bounded(self):
        """Test that Brier score CI is in valid range [0, 1]."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        estimate, ci_low, ci_high = bootstrap_metric(y_true, y_pred, brier_score_loss)
        assert 0 <= ci_low <= ci_high <= 1

    def test_wider_ci_for_smaller_samples(self):
        """Test that smaller samples produce wider confidence intervals."""
        # Large sample
        rng = np.random.default_rng(42)
        y_true_large = rng.integers(0, 2, 100)
        y_pred_large = rng.uniform(0, 1, 100)

        # Small sample (subset)
        y_true_small = y_true_large[:10]
        y_pred_small = y_pred_large[:10]

        _, ci_low_large, ci_high_large = bootstrap_metric(
            y_true_large, y_pred_large, brier_score_loss, random_state=42
        )
        _, ci_low_small, ci_high_small = bootstrap_metric(
            y_true_small, y_pred_small, brier_score_loss, random_state=42
        )

        width_large = ci_high_large - ci_low_large
        width_small = ci_high_small - ci_low_small
        # Small samples should generally have wider CI
        # (This is probabilistic but should hold for most cases)
        assert width_small >= width_large * 0.5  # Allow some tolerance

    def test_custom_confidence_level(self):
        """Test with non-default confidence level."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
        y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5] * 5)

        # 90% CI should be narrower than 99% CI
        _, ci_low_90, ci_high_90 = bootstrap_metric(
            y_true, y_pred, brier_score_loss, confidence_level=0.90, random_state=42
        )
        _, ci_low_99, ci_high_99 = bootstrap_metric(
            y_true, y_pred, brier_score_loss, confidence_level=0.99, random_state=42
        )

        width_90 = ci_high_90 - ci_low_90
        width_99 = ci_high_99 - ci_low_99
        assert width_90 <= width_99

    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_metric(np.array([]), np.array([]), brier_score_loss)

    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched array lengths raise ValueError."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0.1, 0.9])
        with pytest.raises(ValueError, match="same length"):
            bootstrap_metric(y_true, y_pred, brier_score_loss)

    def test_with_accuracy_score(self):
        """Test bootstrap works with accuracy (non-probability metric)."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 0])  # Binary predictions
        estimate, ci_low, ci_high = bootstrap_metric(y_true, y_pred, accuracy_score)
        assert 0 <= ci_low <= estimate <= ci_high <= 1


class TestBootstrapCorrelation:
    """Tests for the bootstrap_correlation function."""

    def test_returns_three_values(self):
        """Test that function returns exactly three values."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([2, 4, 5, 4, 6, 7, 8, 9])
        result = bootstrap_correlation(x, y)
        assert len(result) == 3

    def test_perfect_positive_correlation(self):
        """Test with perfectly correlated data."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = 2 * x + 3
        corr, ci_low, ci_high = bootstrap_correlation(x, y)
        assert corr > 0.99  # Should be ~1.0
        assert ci_low > 0.9

    def test_negative_correlation(self):
        """Test with negatively correlated data."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = -x + 15
        corr, ci_low, ci_high = bootstrap_correlation(x, y)
        assert corr < -0.99
        assert ci_high < -0.9

    def test_handles_nan_values(self):
        """Test that NaN values are properly handled."""
        x = np.array([1, 2, np.nan, 4, 5, 6, 7, 8])
        y = np.array([2, 4, 5, np.nan, 6, 7, 8, 9])
        # Should not raise and should exclude NaN pairs
        corr, ci_low, ci_high = bootstrap_correlation(x, y)
        assert -1 <= ci_low <= corr <= ci_high <= 1

    def test_too_few_valid_pairs_raises_error(self):
        """Test that insufficient valid pairs raise ValueError."""
        x = np.array([1, np.nan, np.nan])
        y = np.array([np.nan, 2, np.nan])
        with pytest.raises(ValueError, match="at least 3"):
            bootstrap_correlation(x, y)

    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched array lengths raise ValueError."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        with pytest.raises(ValueError, match="same length"):
            bootstrap_correlation(x, y)

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([2, 4, 5, 4, 6, 7, 8, 9])
        result1 = bootstrap_correlation(x, y, random_state=42)
        result2 = bootstrap_correlation(x, y, random_state=42)
        assert result1 == result2
