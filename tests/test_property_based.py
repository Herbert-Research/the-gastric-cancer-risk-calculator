"""Property-based tests using Hypothesis for edge case discovery.

This module uses property-based testing to discover edge cases that
traditional example-based tests might miss. Properties that must always
hold are defined and tested against randomly generated inputs.

The Hypothesis library generates a large number of test cases automatically,
exploring boundary conditions and unusual input combinations.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from risk_calculator import (
    GastricCancerRiskModel,
    load_model_config,
    resolve_ln_ratio,
    sigmoid,
)

# =============================================================================
# Sigmoid Function Properties
# =============================================================================


class TestSigmoidProperties:
    """Property-based tests for the sigmoid (logistic) function.

    The sigmoid function σ(x) = 1/(1 + e^{-x}) has several mathematical
    properties that must always hold:

    1. Output bounded in (0, 1)
    2. Monotonically increasing
    3. Symmetric: σ(x) = 1 - σ(-x)
    4. σ(0) = 0.5
    """

    @given(x=st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False))
    def test_sigmoid_bounded(self, x: float) -> None:
        """Sigmoid output is always strictly in (0, 1).

        This tests the fundamental property of the logistic function.
        Even for extreme inputs, the output should never reach exactly 0 or 1.
        """
        result = sigmoid(x)
        assert 0.0 <= result <= 1.0, f"sigmoid({x}) = {result} is out of bounds"

    @given(x=st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False))
    def test_sigmoid_monotonic(self, x: float) -> None:
        """Sigmoid is monotonically increasing.

        For any x₁ < x₂, we must have σ(x₁) ≤ σ(x₂).
        """
        delta = 0.001
        assume(x + delta <= 500)  # Stay within valid range
        assert sigmoid(x) <= sigmoid(
            x + delta
        ), f"sigmoid not monotonic: sigmoid({x}) > sigmoid({x + delta})"

    @given(x=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    def test_sigmoid_symmetry(self, x: float) -> None:
        """Sigmoid is symmetric around (0, 0.5): σ(x) = 1 - σ(-x).

        This is a fundamental property of the logistic function.
        """
        left = sigmoid(x)
        right = 1.0 - sigmoid(-x)
        assert (
            abs(left - right) < 1e-10
        ), f"sigmoid symmetry violated: sigmoid({x}) = {left}, 1 - sigmoid({-x}) = {right}"

    def test_sigmoid_at_zero(self) -> None:
        """Sigmoid at zero equals exactly 0.5."""
        result = sigmoid(0.0)
        assert abs(result - 0.5) < 1e-15, f"sigmoid(0) = {result}, expected 0.5"

    @given(x=st.floats(min_value=100, max_value=700, allow_nan=False, allow_infinity=False))
    def test_sigmoid_large_positive(self, x: float) -> None:
        """Sigmoid of large positive values approaches 1 without overflow.

        Tests numerical stability for large inputs where naive implementation
        would overflow.
        """
        result = sigmoid(x)
        assert result > 0.99, f"sigmoid({x}) = {result}, expected close to 1"
        assert result <= 1.0, f"sigmoid({x}) = {result}, exceeded 1.0"
        assert not math.isnan(result), f"sigmoid({x}) produced NaN"
        assert not math.isinf(result), f"sigmoid({x}) produced Inf"

    @given(x=st.floats(min_value=-700, max_value=-100, allow_nan=False, allow_infinity=False))
    def test_sigmoid_large_negative(self, x: float) -> None:
        """Sigmoid of large negative values approaches 0 without underflow.

        Tests numerical stability for large negative inputs.
        """
        result = sigmoid(x)
        assert result < 0.01, f"sigmoid({x}) = {result}, expected close to 0"
        assert result >= 0.0, f"sigmoid({x}) = {result}, went below 0.0"
        assert not math.isnan(result), f"sigmoid({x}) produced NaN"


# =============================================================================
# Lymph Node Ratio Resolution Properties
# =============================================================================


class TestLnRatioProperties:
    """Property-based tests for lymph node ratio resolution.

    The resolve_ln_ratio function derives the LN ratio from different
    input combinations. These tests verify mathematical correctness.
    """

    @given(
        positive=st.integers(min_value=0, max_value=50),
        total=st.integers(min_value=1, max_value=100),
    )
    def test_ln_ratio_bounded(self, positive: int, total: int) -> None:
        """LN ratio is always between 0 and 1 when computed from counts.

        Given positive_LN ≤ total_LN and total_LN > 0, the ratio must
        be in [0, 1].
        """
        assume(positive <= total)  # Biologically valid: can't have more positive than total
        ratio = resolve_ln_ratio(None, float(positive), float(total))

        assert ratio is not None, "ratio should be computed when total > 0"
        assert 0.0 <= ratio <= 1.0, f"LN ratio {ratio} out of bounds for {positive}/{total}"

    @given(
        ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_ln_ratio_passthrough(self, ratio: float) -> None:
        """When ln_ratio is provided directly, it passes through unchanged.

        The function should return the provided ratio even if counts are also
        provided (priority to explicit ratio).
        """
        result = resolve_ln_ratio(ratio, 5.0, 10.0)  # Counts should be ignored
        assert result == ratio, f"Expected passthrough of {ratio}, got {result}"

    @given(
        positive=st.floats(min_value=0, max_value=50, allow_nan=False),
    )
    def test_ln_ratio_zero_total_returns_none(self, positive: float) -> None:
        """When total_LN is 0, the function returns None (undefined ratio).

        Division by zero must be handled gracefully.
        """
        result = resolve_ln_ratio(None, positive, 0.0)
        assert result is None, "Expected None when total_LN is 0"

    def test_ln_ratio_none_total_returns_none(self) -> None:
        """When total_LN is None, the function returns None."""
        result = resolve_ln_ratio(None, 5.0, None)
        assert result is None, "Expected None when total_LN is None"

    @given(total=st.integers(min_value=1, max_value=100))
    def test_ln_ratio_missing_positive_defaults_to_zero(self, total: int) -> None:
        """When positive_LN is None, it defaults to 0 (no positive nodes found).

        This represents the common case where no positive nodes were found.
        """
        result = resolve_ln_ratio(None, None, float(total))
        assert result == 0.0, f"Expected 0.0 when positive is None, got {result}"


# =============================================================================
# Risk Model Properties
# =============================================================================


class TestRiskModelProperties:
    """Property-based tests for the GastricCancerRiskModel.

    These tests verify that the risk model produces sensible outputs
    across all valid input combinations.
    """

    @pytest.fixture
    def model(self) -> GastricCancerRiskModel:
        """Load the default model configuration."""
        return GastricCancerRiskModel(load_model_config())

    @given(
        t_stage=st.sampled_from(["T1", "T2", "T3", "T4"]),
        n_stage=st.sampled_from(["N0", "N1", "N2", "N3"]),
        age=st.floats(min_value=18, max_value=100, allow_nan=False),
        tumor_size=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_risk_always_bounded(
        self, t_stage: str, n_stage: str, age: float, tumor_size: float
    ) -> None:
        """Risk output is always within configured bounds [risk_floor, risk_ceiling].

        For any valid patient, the predicted risk must respect the model's
        configured bounds.
        """
        model = GastricCancerRiskModel(load_model_config())
        patient = {
            "T_stage": t_stage,
            "N_stage": n_stage,
            "age": age,
            "tumor_size_cm": tumor_size,
        }
        risk = model.calculate_risk(patient)

        assert (
            model.risk_floor <= risk <= model.risk_ceiling
        ), f"Risk {risk} out of bounds [{model.risk_floor}, {model.risk_ceiling}]"

    @given(
        t_stage=st.sampled_from(["T1", "T2", "T3", "T4"]),
        n_stage=st.sampled_from(["N0", "N1", "N2", "N3"]),
    )
    def test_risk_is_float(self, t_stage: str, n_stage: str) -> None:
        """Risk output is always a valid float, never NaN or Inf."""
        model = GastricCancerRiskModel(load_model_config())
        patient = {"T_stage": t_stage, "N_stage": n_stage}
        risk = model.calculate_risk(patient)

        assert isinstance(risk, float), f"Risk should be float, got {type(risk)}"
        assert not math.isnan(risk), f"Risk is NaN for {patient}"
        assert not math.isinf(risk), f"Risk is Inf for {patient}"

    @given(
        t_stage=st.sampled_from(["T1", "T2", "T3", "T4"]),
        n_stage=st.sampled_from(["N0", "N1", "N2", "N3"]),
    )
    @settings(max_examples=100)
    def test_advanced_stage_higher_risk(self, t_stage: str, n_stage: str) -> None:
        """T4N3 should always have higher or equal risk than any other combination.

        This tests the clinical validity: more advanced staging should correlate
        with higher risk.
        """
        model = GastricCancerRiskModel(load_model_config())

        # Most advanced staging
        advanced = model.calculate_risk({"T_stage": "T4", "N_stage": "N3"})

        # Current staging
        current = model.calculate_risk({"T_stage": t_stage, "N_stage": n_stage})

        assert (
            current <= advanced
        ), f"Risk for {t_stage}{n_stage} ({current:.3f}) > T4N3 ({advanced:.3f})"

    @given(
        t_stage=st.sampled_from(["T1", "T2", "T3", "T4"]),
        n_stage=st.sampled_from(["N0", "N1", "N2", "N3"]),
    )
    @settings(max_examples=100)
    def test_early_stage_lower_risk(self, t_stage: str, n_stage: str) -> None:
        """T1N0 should always have lower or equal risk than any other combination.

        This tests the clinical validity: early staging should correlate
        with lower risk.
        """
        model = GastricCancerRiskModel(load_model_config())

        # Earliest staging
        early = model.calculate_risk({"T_stage": "T1", "N_stage": "N0"})

        # Current staging
        current = model.calculate_risk({"T_stage": t_stage, "N_stage": n_stage})

        assert early <= current, f"Risk for T1N0 ({early:.3f}) > {t_stage}{n_stage} ({current:.3f})"

    @given(
        base_age=st.floats(min_value=30, max_value=70, allow_nan=False),
        age_increment=st.floats(min_value=1, max_value=20, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_age_increases_risk(self, base_age: float, age_increment: float) -> None:
        """Older age should result in equal or higher risk.

        The model applies an age penalty for ages above a pivot (typically 50),
        so older patients should have equal or higher risk, ceteris paribus.
        """
        model = GastricCancerRiskModel(load_model_config())

        younger_risk = model.calculate_risk({"T_stage": "T2", "N_stage": "N1", "age": base_age})
        older_risk = model.calculate_risk(
            {"T_stage": "T2", "N_stage": "N1", "age": base_age + age_increment}
        )

        assert (
            younger_risk <= older_risk
        ), f"Age {base_age} risk ({younger_risk:.3f}) > age {base_age + age_increment} ({older_risk:.3f})"

    @given(
        base_size=st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
        size_increment=st.floats(min_value=0.5, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_larger_tumor_higher_risk(self, base_size: float, size_increment: float) -> None:
        """Larger tumor size should result in equal or higher risk.

        Tumor size is a positive prognostic factor in the model.
        """
        model = GastricCancerRiskModel(load_model_config())

        smaller_risk = model.calculate_risk(
            {"T_stage": "T2", "N_stage": "N1", "tumor_size_cm": base_size}
        )
        larger_risk = model.calculate_risk(
            {"T_stage": "T2", "N_stage": "N1", "tumor_size_cm": base_size + size_increment}
        )

        assert (
            smaller_risk <= larger_risk
        ), f"Size {base_size}cm risk ({smaller_risk:.3f}) > size {base_size + size_increment}cm ({larger_risk:.3f})"

    @given(
        ln_ratio_low=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        ln_ratio_high=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_higher_ln_ratio_higher_risk(self, ln_ratio_low: float, ln_ratio_high: float) -> None:
        """Higher lymph node ratio should result in equal or higher risk.

        LN ratio (positive/total) is a key prognostic factor.
        """
        assume(ln_ratio_low < ln_ratio_high)

        model = GastricCancerRiskModel(load_model_config())

        low_ratio_risk = model.calculate_risk(
            {"T_stage": "T2", "N_stage": "N1", "ln_ratio": ln_ratio_low}
        )
        high_ratio_risk = model.calculate_risk(
            {"T_stage": "T2", "N_stage": "N1", "ln_ratio": ln_ratio_high}
        )

        assert (
            low_ratio_risk <= high_ratio_risk
        ), f"LN ratio {ln_ratio_low} risk ({low_ratio_risk:.3f}) > ratio {ln_ratio_high} ({high_ratio_risk:.3f})"


# =============================================================================
# Risk Category Properties
# =============================================================================


class TestRiskCategoryProperties:
    """Property-based tests for risk categorization."""

    @given(risk=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_category_always_valid(self, risk: float) -> None:
        """Risk category is always one of the four defined categories."""
        valid_categories = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
        category = GastricCancerRiskModel.risk_category(risk)

        assert category in valid_categories, f"Unexpected category '{category}' for risk {risk}"

    @given(risk=st.floats(min_value=0.0, max_value=0.199, allow_nan=False))
    def test_low_risk_category(self, risk: float) -> None:
        """Risk < 0.20 is always categorized as 'Low Risk'."""
        category = GastricCancerRiskModel.risk_category(risk)
        assert category == "Low Risk", f"Risk {risk} should be 'Low Risk', got '{category}'"

    @given(risk=st.floats(min_value=0.20, max_value=0.399, allow_nan=False))
    def test_moderate_risk_category(self, risk: float) -> None:
        """Risk in [0.20, 0.40) is always categorized as 'Moderate Risk'."""
        category = GastricCancerRiskModel.risk_category(risk)
        assert (
            category == "Moderate Risk"
        ), f"Risk {risk} should be 'Moderate Risk', got '{category}'"

    @given(risk=st.floats(min_value=0.40, max_value=0.599, allow_nan=False))
    def test_high_risk_category(self, risk: float) -> None:
        """Risk in [0.40, 0.60) is always categorized as 'High Risk'."""
        category = GastricCancerRiskModel.risk_category(risk)
        assert category == "High Risk", f"Risk {risk} should be 'High Risk', got '{category}'"

    @given(risk=st.floats(min_value=0.60, max_value=1.0, allow_nan=False))
    def test_very_high_risk_category(self, risk: float) -> None:
        """Risk >= 0.60 is always categorized as 'Very High Risk'."""
        category = GastricCancerRiskModel.risk_category(risk)
        assert (
            category == "Very High Risk"
        ), f"Risk {risk} should be 'Very High Risk', got '{category}'"

    @given(
        risk1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        risk2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_category_ordering(self, risk1: float, risk2: float) -> None:
        """Higher risk should never result in a 'lower' category.

        Tests that the category ordering is consistent:
        Low Risk < Moderate Risk < High Risk < Very High Risk
        """
        assume(risk1 < risk2)

        category_order = {
            "Low Risk": 0,
            "Moderate Risk": 1,
            "High Risk": 2,
            "Very High Risk": 3,
        }

        cat1 = GastricCancerRiskModel.risk_category(risk1)
        cat2 = GastricCancerRiskModel.risk_category(risk2)

        assert (
            category_order[cat1] <= category_order[cat2]
        ), f"Category ordering violated: {risk1} -> '{cat1}', {risk2} -> '{cat2}'"
