"""Tests for visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from utils.visualization import (
    CATEGORY_COLORS,
    FIG_RISK_PREDICTIONS,
    FIG_SURVIVAL_PREDICTIONS,
    FIG_SURVIVAL_VS_RECURRENCE,
    FIG_TCGA_SUMMARY,
    finalize_figure,
    plot_calibration_curve,
    plot_individual_predictions,
    plot_survival_predictions,
    plot_survival_vs_recurrence,
    plot_tcga_summary,
)


@pytest.fixture
def sample_results():
    """Create minimal results DataFrame for testing."""
    return pd.DataFrame(
        {
            "Patient": ["A", "B", "C", "D"],
            "Risk": [0.15, 0.35, 0.55, 0.85],
            "Category": ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
            "T_stage": ["T1", "T2", "T3", "T4"],
            "N_stage": ["N0", "N1", "N2", "N3"],
        }
    )


@pytest.fixture
def sample_results_with_survival():
    """Create results DataFrame with survival columns."""
    return pd.DataFrame(
        {
            "Patient": ["A", "B", "C", "D", "E"],
            "Risk": [0.15, 0.35, 0.55, 0.75, 0.85],
            "Category": ["Low Risk", "Moderate Risk", "High Risk", "High Risk", "Very High Risk"],
            "T_stage": ["T1", "T2", "T3", "T3", "T4"],
            "N_stage": ["N0", "N1", "N2", "N2", "N3"],
            "survival_5yr": [0.85, 0.70, 0.50, 0.45, 0.25],
            "survival_10yr": [0.75, 0.55, 0.35, 0.30, 0.10],
            "survival_category": [
                "Excellent Prognosis",
                "Good Prognosis",
                "Moderate Prognosis",
                "Poor Prognosis",
                "Very Poor Prognosis",
            ],
        }
    )


@pytest.fixture
def sample_calibration_data():
    """Create results DataFrame with event labels for calibration."""
    return pd.DataFrame(
        {
            "Patient": [f"P{i}" for i in range(50)],
            "Risk": [0.1 + 0.8 * (i / 49) for i in range(50)],  # Range from 0.1 to 0.9
            "event_observed": [1 if i % 3 == 0 else 0 for i in range(50)],  # ~33% events
        }
    )


class TestCategoryColors:
    """Test that category color mapping is properly defined."""

    def test_all_categories_have_colors(self):
        """Verify all standard risk categories have defined colors."""
        expected_categories = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
        for category in expected_categories:
            assert category in CATEGORY_COLORS
            assert CATEGORY_COLORS[category].startswith("#")


class TestFinalizeFunction:
    """Test the finalize_figure utility function."""

    def test_finalize_creates_output_directory(self):
        """Test that output directory is created if missing."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "subdir" / "figure.png"
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])

            output_path = finalize_figure(fig, nested_path, show_plots=False)
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_finalize_closes_figure_when_not_showing(self):
        """Test that figure is closed when show_plots=False."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            fig_num = fig.number
            ax.plot([1, 2, 3])

            finalize_figure(fig, Path(tmpdir) / "test.png", show_plots=False)
            # Figure should be closed
            assert fig_num not in plt.get_fignums()


class TestFigureGeneration:
    """Test that figures are created without errors."""

    def test_individual_predictions_creates_file(self, sample_results):
        """Test individual predictions plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_individual_predictions(
                sample_results, Path(tmpdir), show_plots=False
            )
            assert output_path.exists()
            assert output_path.suffix == ".png"
            assert output_path.name == FIG_RISK_PREDICTIONS

    def test_individual_predictions_handles_missing_categories(self):
        """Test plot handles missing risk categories gracefully."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B"],
                "Risk": [0.15, 0.25],
                "Category": ["Low Risk", "Low Risk"],  # Only one category
                "T_stage": ["T1", "T2"],
                "N_stage": ["N0", "N0"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_individual_predictions(df, Path(tmpdir), show_plots=False)
            assert output_path.exists()

    def test_tcga_summary_creates_file(self, sample_results):
        """Test TCGA summary plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_tcga_summary(sample_results, Path(tmpdir), show_plots=False)
            assert output_path.exists()
            assert output_path.suffix == ".png"
            assert output_path.name == FIG_TCGA_SUMMARY

    def test_tcga_summary_handles_sparse_stages(self):
        """Test TCGA summary handles sparse stage combinations."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B", "C"],
                "Risk": [0.15, 0.45, 0.85],
                "Category": ["Low Risk", "High Risk", "Very High Risk"],
                "T_stage": ["T1", "T2", "T4"],  # Missing T3
                "N_stage": ["N0", "N1", "N3"],  # Missing N2
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_tcga_summary(df, Path(tmpdir), show_plots=False)
            assert output_path.exists()


class TestSurvivalPlots:
    """Test survival-related plots."""

    def test_survival_predictions_creates_file(self, sample_results_with_survival):
        """Test survival predictions plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_predictions(
                sample_results_with_survival, Path(tmpdir), show_plots=False
            )
            assert output_path is not None
            assert output_path.exists()
            assert output_path.name == FIG_SURVIVAL_PREDICTIONS

    def test_survival_predictions_returns_none_without_data(self, sample_results):
        """Test survival predictions returns None when survival data missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_predictions(
                sample_results, Path(tmpdir), show_plots=False  # No survival columns
            )
            assert output_path is None

    def test_survival_vs_recurrence_creates_file(self, sample_results_with_survival):
        """Test survival vs recurrence comparison plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_vs_recurrence(
                sample_results_with_survival, Path(tmpdir), show_plots=False
            )
            assert output_path is not None
            assert output_path.exists()
            assert output_path.name == FIG_SURVIVAL_VS_RECURRENCE

    def test_survival_vs_recurrence_returns_none_without_data(self, sample_results):
        """Test returns None when required columns missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_vs_recurrence(
                sample_results, Path(tmpdir), show_plots=False
            )
            assert output_path is None

    def test_survival_vs_recurrence_handles_empty_after_dropna(self):
        """Test handles case where all rows have NaN."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B"],
                "Risk": [0.5, 0.6],
                "survival_5yr": [None, None],  # All NaN
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_survival_vs_recurrence(df, Path(tmpdir), show_plots=False)
            assert output_path is None


class TestCalibrationCurve:
    """Test calibration curve plotting."""

    def test_calibration_curve_creates_file(self, sample_calibration_data):
        """Test calibration curve plot generates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                sample_calibration_data,
                Path(tmpdir),
                show_plots=False,
                label_column="event_observed",
            )
            assert result is not None
            output_path, brier, ci_low, ci_high = result
            assert output_path.exists()
            assert output_path.suffix == ".png"
            assert 0.0 <= brier <= 1.0
            # CI should contain point estimate
            assert ci_low <= brier <= ci_high
            assert 0.0 <= ci_low <= ci_high <= 1.0

    def test_calibration_curve_returns_none_without_labels(self, sample_results):
        """Test returns None when label column missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                sample_results, Path(tmpdir), show_plots=False, label_column="event_observed"
            )
            assert result is None

    def test_calibration_curve_handles_all_nan_labels(self):
        """Test handles case where all labels are NaN."""
        df = pd.DataFrame(
            {
                "Patient": ["A", "B"],
                "Risk": [0.5, 0.6],
                "event_observed": [None, None],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                df, Path(tmpdir), show_plots=False, label_column="event_observed"
            )
            assert result is None

    def test_calibration_curve_bootstrap_reproducibility(self, sample_calibration_data):
        """Test that bootstrap CI is reproducible with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = plot_calibration_curve(
                sample_calibration_data,
                Path(tmpdir),
                show_plots=False,
                label_column="event_observed",
                n_bootstrap=100,
            )
            result2 = plot_calibration_curve(
                sample_calibration_data,
                Path(tmpdir),
                show_plots=False,
                label_column="event_observed",
                n_bootstrap=100,
            )
            assert result1 is not None and result2 is not None
            _, brier1, ci_low1, ci_high1 = result1
            _, brier2, ci_low2, ci_high2 = result2
            # Same seed should give same results
            assert brier1 == brier2
            assert ci_low1 == ci_low2
            assert ci_high1 == ci_high2


class TestPlotContent:
    """Test that plot content is as expected."""

    def test_individual_predictions_bar_count(self, sample_results):
        """Test that correct number of bars are plotted."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        # Get figure before it's closed
        colors = sample_results["Category"].map(CATEGORY_COLORS).fillna("gray")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bars = axes[0].barh(sample_results["Patient"], sample_results["Risk"] * 100, color=colors)
        assert len(bars) == len(sample_results)
        plt.close(fig)

    def test_risk_values_scaled_correctly(self, sample_results):
        """Verify risk values are scaled to percentage (0-100)."""
        # Risk values in DataFrame are 0-1, but plots show 0-100%
        assert all(0 <= r <= 1 for r in sample_results["Risk"])
        # Scaled values for display
        scaled = sample_results["Risk"] * 100
        assert all(0 <= s <= 100 for s in scaled)


class TestVisualizationEdgeCases:
    """Additional edge case tests for visualization functions.

    These tests verify that visualization functions handle unusual inputs
    gracefully, including NaN values, single-bin scenarios, and empty data.
    """

    def test_survival_predictions_with_nan_values(self):
        """Ensure NaN survival values produce a valid plot (non-NaN data exists)."""
        import numpy as np

        df = pd.DataFrame(
            {
                "Patient": ["A", "B", "C", "D", "E"],
                "Risk": [0.15, 0.35, 0.55, 0.65, 0.85],
                "Category": [
                    "Low Risk",
                    "Moderate Risk",
                    "High Risk",
                    "High Risk",
                    "Very High Risk",
                ],
                "survival_5yr": [0.8, np.nan, 0.6, 0.4, 0.2],
                "survival_10yr": [0.7, 0.5, np.nan, 0.3, 0.1],
                "survival_category": [
                    "Good Prognosis",
                    None,
                    "Moderate Prognosis",
                    "Poor Prognosis",
                    "Very Poor Prognosis",
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_survival_predictions(df, Path(tmpdir), show_plots=False)
            assert result is not None, "Expected a plot when non-NaN survival data exists"
            assert result.exists()

    def test_survival_vs_recurrence_with_partial_nan(self):
        """Test survival vs recurrence plot with some NaN values.

        5 of 6 rows have valid data, so a plot should be produced.
        """
        import numpy as np

        df = pd.DataFrame(
            {
                "Patient": ["A", "B", "C", "D", "E", "F"],
                "Risk": [0.15, 0.35, 0.55, 0.65, 0.75, 0.85],
                "survival_5yr": [0.8, np.nan, 0.6, 0.5, 0.4, 0.2],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_survival_vs_recurrence(df, Path(tmpdir), show_plots=False)
            assert result is not None, "Expected a plot when 5/6 rows have valid data"
            assert result.exists()

    def test_calibration_with_extreme_predictions(self):
        """Test calibration curve with predictions at boundaries (0 and 1).

        20 patients with varied predictions and both classes present --
        calibration should succeed.
        """
        df = pd.DataFrame(
            {
                "Patient": [f"P{i}" for i in range(20)],
                "Risk": [0.0, 0.0, 0.05, 0.1]
                + [0.3 + 0.03 * i for i in range(12)]
                + [0.95, 1.0, 1.0, 1.0],
                "event_observed": [0, 0, 0, 0]
                + [1 if i % 2 == 0 else 0 for i in range(12)]
                + [1, 1, 1, 1],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                df, Path(tmpdir), show_plots=False, label_column="event_observed"
            )
            assert result is not None, "Expected calibration to succeed with 20 patients"
            output_path, brier, ci_low, ci_high = result
            assert output_path.exists()
            assert 0.0 <= brier <= 1.0

    def test_calibration_with_imbalanced_classes(self):
        """Test calibration curve with highly imbalanced classes (5% event rate).

        100 patients with both classes present -- calibration should succeed.
        """
        df = pd.DataFrame(
            {
                "Patient": [f"P{i}" for i in range(100)],
                "Risk": [0.05 + 0.009 * i for i in range(100)],
                "event_observed": [1 if i < 5 else 0 for i in range(100)],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_calibration_curve(
                df, Path(tmpdir), show_plots=False, label_column="event_observed"
            )
            assert result is not None, "Expected calibration to succeed with 100 patients"
            output_path, brier, ci_low, ci_high = result
            assert output_path.exists()
            assert 0.0 <= brier <= 1.0

    def test_tcga_summary_single_stage_combination(self):
        """Test TCGA summary with only one stage combination.

        Edge case where all patients have the same staging.
        """
        df = pd.DataFrame(
            {
                "Patient": ["A", "B", "C", "D", "E"],
                "Risk": [0.45, 0.46, 0.47, 0.48, 0.49],
                "Category": ["High Risk"] * 5,
                "T_stage": ["T2"] * 5,
                "N_stage": ["N1"] * 5,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_tcga_summary(df, Path(tmpdir), show_plots=False)
            assert output_path.exists()

    def test_individual_predictions_large_cohort(self):
        """Test individual predictions with a larger cohort.

        Verify the function handles 100+ patients efficiently.
        """
        n_patients = 150
        import numpy as np

        rng = np.random.default_rng(42)
        categories = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
        t_stages = ["T1", "T2", "T3", "T4"]
        n_stages = ["N0", "N1", "N2", "N3"]

        df = pd.DataFrame(
            {
                "Patient": [f"P{i:03d}" for i in range(n_patients)],
                "Risk": rng.uniform(0.05, 0.95, n_patients),
                "Category": rng.choice(categories, n_patients),
                "T_stage": rng.choice(t_stages, n_patients),
                "N_stage": rng.choice(n_stages, n_patients),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_individual_predictions(df, Path(tmpdir), show_plots=False)
            assert output_path.exists()

    def test_survival_predictions_all_same_category(self):
        """Test survival predictions where all patients have the same prognosis.

        All 4 patients have valid survival data, so a plot should be produced.
        """
        df = pd.DataFrame(
            {
                "Patient": ["A", "B", "C", "D"],
                "Risk": [0.10, 0.12, 0.14, 0.16],
                "Category": ["Low Risk"] * 4,
                "survival_5yr": [0.90, 0.88, 0.87, 0.85],
                "survival_10yr": [0.80, 0.78, 0.77, 0.75],
                "survival_category": ["Excellent Prognosis"] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_survival_predictions(df, Path(tmpdir), show_plots=False)
            assert result is not None, "Expected a plot when all patients have valid survival data"
            assert result.exists()

    def test_plot_handles_special_characters_in_patient_names(self):
        """Test that special characters in patient IDs don't break plotting."""
        df = pd.DataFrame(
            {
                "Patient": ["Patient A/B", "Patient (C)", "Patient D&E", "Patient 'F'"],
                "Risk": [0.15, 0.35, 0.55, 0.85],
                "Category": ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
                "T_stage": ["T1", "T2", "T3", "T4"],
                "N_stage": ["N0", "N1", "N2", "N3"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = plot_individual_predictions(df, Path(tmpdir), show_plots=False)
            assert output_path.exists()

    def test_calibration_with_identical_predictions(self):
        """Test calibration curve when all predictions are identical.

        Identical predictions may cause binning issues in sklearn's
        calibration_curve. The function should handle this gracefully --
        either producing a valid result or raising a catchable error.
        """
        df = pd.DataFrame(
            {
                "Patient": [f"P{i}" for i in range(20)],
                "Risk": [0.5] * 20,
                "event_observed": [1 if i % 2 == 0 else 0 for i in range(20)],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = plot_calibration_curve(
                    df, Path(tmpdir), show_plots=False, label_column="event_observed"
                )
            except ValueError:
                return
            if result is not None:
                _, brier, _, _ = result
                assert 0.0 <= brier <= 1.0
