"""Regression tests pinning the example outputs published in the README.

These tests freeze the exact recurrence-risk percentages documented in the
README "Example Output" / "Quick Verification" sections. If the model
coefficients, sigmoid, or scoring path change, these tests fail loudly so the
README can never silently drift from the code that produced it.
"""

from __future__ import annotations

import pytest

from risk_calculator import (
    GastricCancerRiskModel,
    load_model_config,
    run_example_patients,
)

# (patient label, expected 5-year recurrence risk %) as printed in the README.
EXPECTED_RECURRENCE_PCT = {
    "Patient A - Early Stage": 12.8,
    "Patient B - Moderate Stage": 64.1,
    "Patient C - Advanced Stage": 94.3,
    "Patient D - Very Advanced": 95.0,
}

EXPECTED_CATEGORY = {
    "Patient A - Early Stage": "Low Risk",
    "Patient B - Moderate Stage": "Very High Risk",
    "Patient C - Advanced Stage": "Very High Risk",
    "Patient D - Very Advanced": "Very High Risk",
}


@pytest.fixture
def model() -> GastricCancerRiskModel:
    return GastricCancerRiskModel(load_model_config())


def test_readme_example_risks_match(model: GastricCancerRiskModel):
    """The four documented patients reproduce the README percentages exactly."""
    results = run_example_patients(model, survival_model=None)

    for _, row in results.iterrows():
        label = row["Patient"]
        expected_pct = EXPECTED_RECURRENCE_PCT[label]
        actual_pct = round(row["Risk"] * 100.0, 1)
        assert actual_pct == pytest.approx(expected_pct, abs=0.05), (
            f"{label}: README documents {expected_pct}% recurrence risk but the "
            f"model now produces {actual_pct}%. Update the README or the model."
        )
        assert row["Category"] == EXPECTED_CATEGORY[label]
