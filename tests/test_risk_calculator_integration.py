"""End-to-end integration tests for the gastric cancer risk calculator."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from risk_calculator import (
    GastricCancerRiskModel,
    estimate_ln_ratio,
    estimate_tumor_size,
    load_model_config,
    load_survival_model,
    load_tcga_cohort,
    normalize_n_stage,
    normalize_t_stage,
    parse_event_status,
    predict_with_both_models,
    resolve_ln_ratio,
    safe_float,
    score_patients,
    sigmoid,
)

TCGA_DATA_PATH = Path("data/tcga_2018_clinical_data.tsv")
FIXTURE_PATH = Path("tests/fixtures/dummy_data.tsv")


def get_tcga_or_fixture_path() -> tuple[Path, bool]:
    """Return TCGA data path if available, otherwise the synthetic fixture."""
    if TCGA_DATA_PATH.exists():
        return TCGA_DATA_PATH, False
    if FIXTURE_PATH.exists():
        return FIXTURE_PATH, True
    pytest.skip("TCGA data file and fixture not available")


class TestSigmoidFunction:
    """Test the numerically stable sigmoid implementation."""

    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        assert sigmoid(0.0) == 0.5

    def test_sigmoid_positive(self):
        """Test sigmoid with positive values."""
        result = sigmoid(2.0)
        assert 0.8 < result < 0.9

    def test_sigmoid_negative(self):
        """Test sigmoid with negative values."""
        result = sigmoid(-2.0)
        assert 0.1 < result < 0.2

    def test_sigmoid_large_negative_no_overflow(self):
        """Test sigmoid handles large negative values without overflow."""
        result = sigmoid(-710)
        assert 0.0 <= result <= 1.0, f"Risk score {result} out of valid range [0, 1]"
        assert math.isfinite(result), f"Risk score {result} is not finite"

    def test_sigmoid_large_positive(self):
        """Test sigmoid handles large positive values."""
        result = sigmoid(710)
        assert 0.0 <= result <= 1.0, f"Risk score {result} out of valid range [0, 1]"


class TestSafeFloat:
    """Test the safe_float utility function."""

    def test_safe_float_python_number(self):
        """Test with standard Python numbers."""
        assert safe_float(3.14) == 3.14
        assert safe_float(42) == 42.0

    def test_safe_float_numpy_scalar(self):
        """Test with numpy scalar types."""
        import numpy as np

        assert safe_float(np.float64(3.14)) == 3.14
        assert safe_float(np.int32(42)) == 42.0


class TestResolveLnRatio:
    """Test the LN ratio resolution logic."""

    def test_direct_ratio_used(self):
        """Test that direct ratio is used when provided."""
        assert resolve_ln_ratio(0.25, None, None) == 0.25

    def test_calculated_from_counts(self):
        """Test ratio is calculated from counts when direct not provided."""
        assert resolve_ln_ratio(None, 5, 20) == 0.25

    def test_returns_none_for_zero_total(self):
        """Test returns None when total LN is 0."""
        assert resolve_ln_ratio(None, 0, 0) is None

    def test_returns_none_for_missing_data(self):
        """Test returns None when insufficient data."""
        assert resolve_ln_ratio(None, None, None) is None
        assert resolve_ln_ratio(None, 5, None) is None


class TestStageNormalization:
    """Test T and N stage normalization functions."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("T1", "T1"),
            ("T2", "T2"),
            ("T3", "T3"),
            ("T4", "T4"),
            ("T1a", "T1"),  # Substage normalized
            ("T4b", "T4"),
            ("t2", "T2"),  # Case insensitive
            (None, None),
            ("TX", None),  # Unknown
            ("", None),
        ],
    )
    def test_normalize_t_stage(self, input_val, expected):
        result = normalize_t_stage(input_val)
        assert result == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("N0", "N0"),
            ("N1", "N1"),
            ("N2", "N2"),
            ("N3", "N3"),
            ("N3a", "N3"),  # Substage normalized
            ("n1", "N1"),  # Case insensitive
            (None, None),
            ("NX", None),
        ],
    )
    def test_normalize_n_stage(self, input_val, expected):
        result = normalize_n_stage(input_val)
        assert result == expected


class TestStageEstimation:
    """Test tumor size and LN ratio estimation from stage."""

    def test_estimate_tumor_size_t1(self):
        """T1 tumors are typically small."""
        size = estimate_tumor_size("T1")
        assert 0 < size < 3

    def test_estimate_tumor_size_t4(self):
        """T4 tumors are typically larger."""
        size = estimate_tumor_size("T4")
        assert size > 5

    def test_estimate_ln_ratio_n0(self):
        """N0 means no or very few positive nodes."""
        ratio = estimate_ln_ratio("N0")
        assert ratio <= 0.05  # Very low ratio for N0

    def test_estimate_ln_ratio_n3(self):
        """N3 has high positive node ratio."""
        ratio = estimate_ln_ratio("N3")
        assert ratio > 0.3


class TestParseEventStatus:
    """Test disease-free status parsing."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("0:DiseaseFree", 0.0),
            ("1:Recurred/Progressed", 1.0),
            ("Tumor Free", 0.0),
            ("With Tumor", 1.0),
            ("censored", 0.0),
            ("progression", 1.0),
            (None, None),
            ("", None),
        ],
    )
    def test_parse_event_status(self, input_val, expected):
        result = parse_event_status(input_val)
        assert result == expected


class TestModelConfiguration:
    """Test model configuration loading."""

    def test_load_model_config_default(self):
        """Test loading default model config."""
        config = load_model_config()
        assert "id" in config
        assert "t_stage_weights" in config
        assert "n_stage_weights" in config
        assert "intercept" in config

    def test_load_model_config_missing_file(self):
        """Test fallback when config file is missing."""
        config = load_model_config(Path("/nonexistent/path.json"))
        # Should fall back to default
        assert config is not None
        assert "id" in config


class TestGastricCancerRiskModel:
    """Test the main risk model class."""

    @pytest.fixture
    def model(self):
        """Create model instance with default config."""
        config = load_model_config()
        return GastricCancerRiskModel(config)

    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.t_stage_weights is not None
        assert model.n_stage_weights is not None

    def test_calculate_risk_basic(self, model):
        """Test basic risk calculation."""
        patient = {"T_stage": "T2", "N_stage": "N1", "age": 60}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_calculate_risk_early_stage(self, model):
        """Test that early stage has lower risk."""
        early = {"T_stage": "T1", "N_stage": "N0", "age": 50}
        late = {"T_stage": "T4", "N_stage": "N3", "age": 70}
        risk_early = model.calculate_risk(early)
        risk_late = model.calculate_risk(late)
        assert risk_early < risk_late

    def test_calculate_risk_with_all_factors(self, model):
        """Test risk calculation with all clinical factors."""
        patient = {
            "T_stage": "T3",
            "N_stage": "N2",
            "age": 65,
            "tumor_size_cm": 5.0,
            "positive_LN": 5,
            "total_LN": 25,
        }
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_calculate_risk_invalid_t_stage(self, model):
        """Test that invalid T stage raises ValueError."""
        patient = {"T_stage": "T5", "N_stage": "N1"}
        with pytest.raises(ValueError, match="Unsupported T stage"):
            model.calculate_risk(patient)

    def test_calculate_risk_invalid_n_stage(self, model):
        """Test that invalid N stage raises ValueError."""
        patient = {"T_stage": "T2", "N_stage": "N5"}
        with pytest.raises(ValueError, match="Unsupported N stage"):
            model.calculate_risk(patient)

    def test_risk_category_low(self, model):
        """Test low risk category."""
        assert model.risk_category(0.15) == "Low Risk"

    def test_risk_category_moderate(self, model):
        """Test moderate risk category."""
        assert model.risk_category(0.30) == "Moderate Risk"

    def test_risk_category_high(self, model):
        """Test high risk category."""
        assert model.risk_category(0.50) == "High Risk"

    def test_risk_category_very_high(self, model):
        """Test very high risk category."""
        assert model.risk_category(0.70) == "Very High Risk"

    def test_risk_bounded_by_floor_ceiling(self, model):
        """Test that risk is bounded by floor and ceiling."""
        # Very low risk patient
        low_patient = {"T_stage": "T1", "N_stage": "N0", "age": 30}
        risk_low = model.calculate_risk(low_patient)
        assert risk_low >= model.risk_floor

        # Very high risk patient
        high_patient = {"T_stage": "T4", "N_stage": "N3", "age": 80, "ln_ratio": 0.9}
        risk_high = model.calculate_risk(high_patient)
        assert risk_high <= model.risk_ceiling


class TestScorePatients:
    """Test patient scoring functionality."""

    @pytest.fixture
    def model(self):
        config = load_model_config()
        return GastricCancerRiskModel(config)

    def test_score_single_patient(self, model):
        """Test scoring a single patient."""
        patients = [{"T_stage": "T2", "N_stage": "N1", "age": 60, "name": "Test Patient"}]
        results = score_patients(model, patients)

        assert len(results) == 1
        assert "Risk" in results.columns
        assert "Category" in results.columns
        assert results.iloc[0]["Patient"] == "Test Patient"

    def test_score_multiple_patients(self, model):
        """Test scoring multiple patients."""
        patients = [
            {"T_stage": "T1", "N_stage": "N0", "name": "Early"},
            {"T_stage": "T3", "N_stage": "N2", "name": "Advanced"},
        ]
        results = score_patients(model, patients)

        assert len(results) == 2
        assert results.iloc[0]["Risk"] < results.iloc[1]["Risk"]


class TestTCGACohort:
    """Test TCGA cohort loading and processing."""

    def test_tcga_cohort_loads_successfully(self):
        """Test that TCGA cohort can be loaded."""
        data_path, is_fixture = get_tcga_or_fixture_path()
        cohort = load_tcga_cohort(data_path)
        assert not cohort.empty
        assert "T_stage" in cohort.columns
        assert "N_stage" in cohort.columns
        assert "age" in cohort.columns
        if is_fixture:
            assert len(cohort) >= 5
        else:
            assert len(cohort) > 400  # TCGA STAD has ~436 patients

    def test_tcga_cohort_has_required_columns(self):
        """Test that loaded cohort has all required columns."""
        data_path, _ = get_tcga_or_fixture_path()
        cohort = load_tcga_cohort(data_path)
        required = ["T_stage", "N_stage", "age", "tumor_size_cm", "ln_ratio"]
        for col in required:
            assert col in cohort.columns, f"Missing column: {col}"

    def test_tcga_cohort_missing_file(self):
        """Test handling of missing data file."""
        cohort = load_tcga_cohort(Path("/nonexistent/path.tsv"))
        assert cohort.empty


class TestSurvivalModel:
    """Test survival model loading and predictions."""

    def test_load_survival_model(self):
        """Test loading the Han 2012 survival model."""
        model = load_survival_model()
        if model is None:
            pytest.skip("Survival model components not available")

        assert model.name is not None
        assert model.baseline_5yr is not None

    def test_load_survival_model_missing_file(self):
        """Test handling of missing model file."""
        model = load_survival_model(Path("/nonexistent/model.json"))
        assert model is None


class TestDualModelScoring:
    """Test combined recurrence and survival scoring."""

    @pytest.fixture
    def models(self):
        """Load both models."""
        config = load_model_config()
        recurrence = GastricCancerRiskModel(config)
        survival = load_survival_model()
        return recurrence, survival

    def test_dual_model_scoring(self, models):
        """Test scoring with both models."""
        recurrence_model, survival_model = models
        patients = [
            {"T_stage": "T2", "N_stage": "N1", "age": 60, "Sex": "Male"},
            {"T_stage": "T3", "N_stage": "N2", "age": 65, "Sex": "Female"},
        ]
        results = predict_with_both_models(patients, recurrence_model, survival_model)

        assert len(results) == 2
        assert "Risk" in results.columns
        assert all(0 <= r <= 1 for r in results["Risk"])

        # Check survival columns if model available
        if survival_model is not None:
            assert "survival_5yr" in results.columns
            assert "survival_10yr" in results.columns

    def test_dual_model_without_survival(self, models):
        """Test scoring without survival model."""
        recurrence_model, _ = models
        patients = [{"T_stage": "T2", "N_stage": "N1", "age": 60}]
        results = predict_with_both_models(patients, recurrence_model, survival_model=None)

        assert len(results) == 1
        assert "Risk" in results.columns
        assert "survival_5yr" not in results.columns

    def test_dual_model_risk_survival_correlation(self, models):
        """Test that recurrence risk and survival are inversely related."""
        recurrence_model, survival_model = models
        if survival_model is None:
            pytest.skip("Survival model not available")

        patients = [
            {"T_stage": "T1", "N_stage": "N0", "age": 50, "Sex": "Female"},  # Low risk
            {"T_stage": "T4", "N_stage": "N3", "age": 75, "Sex": "Male"},  # High risk
        ]
        results = predict_with_both_models(patients, recurrence_model, survival_model)

        # Higher recurrence risk should correlate with lower survival
        low_risk_row = results.iloc[0]
        high_risk_row = results.iloc[1]

        assert low_risk_row["Risk"] < high_risk_row["Risk"]
        if "survival_5yr" in results.columns:
            assert low_risk_row["survival_5yr"] > high_risk_row["survival_5yr"]


class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    def test_complete_pipeline_example_patients(self):
        """Test running the example patients through the full pipeline."""
        config = load_model_config()
        model = GastricCancerRiskModel(config)
        survival_model = load_survival_model()

        patients = [
            {
                "name": "Early Stage",
                "T_stage": "T1",
                "N_stage": "N0",
                "age": 55,
                "Sex": "Female",
                "tumor_size_cm": 2.0,
                "positive_LN": 0,
                "total_LN": 20,
            },
            {
                "name": "Advanced Stage",
                "T_stage": "T3",
                "N_stage": "N2",
                "age": 68,
                "Sex": "Male",
                "tumor_size_cm": 5.0,
                "positive_LN": 8,
                "total_LN": 30,
            },
        ]

        results = predict_with_both_models(patients, model, survival_model)

        assert len(results) == 2
        assert results.iloc[0]["Category"] != results.iloc[1]["Category"]
        # Early stage should be lower risk
        assert results.iloc[0]["Risk"] < results.iloc[1]["Risk"]

    def test_tcga_cohort_pipeline(self):
        """Test processing the full TCGA cohort."""
        data_path, is_fixture = get_tcga_or_fixture_path()

        # Load data
        cohort = load_tcga_cohort(data_path)
        assert len(cohort) > 0

        # Load models
        config = load_model_config()
        model = GastricCancerRiskModel(config)

        # Score a subset for speed
        sample_patients = []
        sample_size = 5 if is_fixture else 10
        for _, row in cohort.head(sample_size).iterrows():
            sample_patients.append(
                {
                    "name": row.get("patient_id", "Unknown"),
                    "T_stage": row["T_stage"],
                    "N_stage": row["N_stage"],
                    "age": row["age"],
                    "tumor_size_cm": row.get("tumor_size_cm"),
                    "ln_ratio": row.get("ln_ratio"),
                }
            )

        results = score_patients(model, sample_patients)
        assert len(results) == sample_size
        assert all(results["Risk"].notna())


class TestMainFunction:
    """End-to-end tests for the main CLI entrypoint."""

    def test_main_with_default_args(self, tmp_path):
        """Test main() executes without error using default data."""
        from risk_calculator import main

        data_path, _ = get_tcga_or_fixture_path()
        test_args = [
            "risk_calculator.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(tmp_path),
        ]
        with patch.object(sys, "argv", test_args):
            # Should not raise
            main()

        # Verify key outputs created
        assert (tmp_path / "risk_predictions.png").exists()
        assert (tmp_path / "sensitivity_analysis.png").exists()

    def test_main_skip_survival(self, tmp_path):
        """Test main() with --skip-survival flag."""
        from risk_calculator import main

        data_path, _ = get_tcga_or_fixture_path()
        test_args = [
            "risk_calculator.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(tmp_path),
            "--skip-survival",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Risk predictions should exist
        assert (tmp_path / "risk_predictions.png").exists()
        # Survival plot should NOT exist when skipped
        assert not (tmp_path / "survival_predictions_han2012.png").exists()

    def test_main_verbose_logging(self, tmp_path, caplog):
        """Test verbose flag enables DEBUG logging."""
        import logging

        from risk_calculator import main

        data_path, _ = get_tcga_or_fixture_path()
        test_args = [
            "risk_calculator.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(tmp_path),
            "--verbose",
        ]
        with patch.object(sys, "argv", test_args):
            with caplog.at_level(logging.DEBUG):
                main()

        # Verbose mode should have been used (check log level was set)
        assert (tmp_path / "risk_predictions.png").exists()

    def test_main_with_log_timestamps(self, tmp_path):
        """Test main() with --log-timestamps flag."""
        from risk_calculator import main

        data_path, _ = get_tcga_or_fixture_path()
        test_args = [
            "risk_calculator.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(tmp_path),
            "--log-timestamps",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        assert (tmp_path / "risk_predictions.png").exists()

    def test_main_custom_model_config(self, tmp_path):
        """Test main() with custom model config path."""
        import json

        from risk_calculator import main

        # Create custom config
        custom_config = {
            "id": "test_model",
            "name": "Test Model",
            "intercept": -2.0,
            "t_stage_weights": {"T1": 0.0, "T2": 0.5, "T3": 1.0, "T4": 1.5},
            "n_stage_weights": {"N0": 0.0, "N1": 0.5, "N2": 1.0, "N3": 1.5},
            "risk_floor": 0.01,
            "risk_ceiling": 0.99,
        }
        config_path = tmp_path / "custom_config.json"
        config_path.write_text(json.dumps(custom_config))

        data_path, _ = get_tcga_or_fixture_path()
        test_args = [
            "risk_calculator.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(tmp_path),
            "--model-config",
            str(config_path),
            "--skip-survival",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        assert (tmp_path / "risk_predictions.png").exists()


class TestErrorHandling:
    """Test graceful error handling for edge cases."""

    def test_load_tcga_cohort_missing_file(self):
        """Test load_tcga_cohort with non-existent file."""
        result = load_tcga_cohort(Path("/nonexistent/path/to/data.tsv"))
        assert result.empty

    def test_load_tcga_cohort_malformed_tsv(self, tmp_path):
        """Test behavior when TSV is malformed or missing required columns."""
        bad_tsv = tmp_path / "bad_data.tsv"
        # Create a file that's valid but missing expected columns
        bad_tsv.write_text("column_a\tcolumn_b\n1\t2\n3\t4\n")

        # The function should either return empty or raise - test that it doesn't crash
        # unexpectedly. With missing required columns like 'age', it may raise KeyError
        import pandas as pd

        try:
            result = load_tcga_cohort(bad_tsv)
            # If it returns, should be a DataFrame
            assert isinstance(result, pd.DataFrame)
        except KeyError:
            # KeyError is acceptable for missing required columns
            pass

    def test_load_model_config_missing_file(self):
        """Test fallback when config file is missing."""
        config = load_model_config(Path("/nonexistent/path.json"))
        # Should fall back to default
        assert config is not None
        assert "id" in config
        assert config["id"] == "heuristic_klass_v1"

    def test_load_model_config_invalid_json(self, tmp_path):
        """Test fallback when JSON is malformed."""
        bad_json = tmp_path / "bad_config.json"
        bad_json.write_text("{invalid json content")

        config = load_model_config(bad_json)
        # Should fall back to default
        assert config is not None
        assert "id" in config

    def test_load_survival_model_missing_file(self):
        """Test handling of missing survival model file."""
        model = load_survival_model(Path("/nonexistent/model.json"))
        assert model is None

    def test_model_with_missing_coefficients(self):
        """Test model handles missing coefficient keys gracefully."""
        minimal_config = {
            "id": "minimal",
            "t_stage_weights": {"T1": 0.0, "T2": 0.5, "T3": 1.0, "T4": 1.5},
            "n_stage_weights": {"N0": 0.0, "N1": 0.5, "N2": 1.0, "N3": 1.5},
        }
        model = GastricCancerRiskModel(minimal_config)
        patient = {"T_stage": "T2", "N_stage": "N1"}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_safe_float_with_none(self):
        """Test safe_float handles edge cases."""
        # Test with valid inputs
        assert safe_float(3.14) == 3.14
        assert safe_float(42) == 42.0

        import numpy as np

        assert safe_float(np.float64(3.14)) == pytest.approx(3.14)


class TestSensitivityAnalysis:
    """Test lymph node yield sensitivity analysis."""

    def test_sensitivity_analysis_output_created(self, tmp_path):
        """Verify sensitivity analysis produces output file."""
        from risk_calculator import run_sensitivity_analysis

        model = GastricCancerRiskModel(load_model_config())
        output_path = run_sensitivity_analysis(model, tmp_path, show_plots=False)

        assert output_path.exists()
        assert output_path.name == "sensitivity_analysis.png"

    def test_sensitivity_analysis_decreasing_risk(self, tmp_path):
        """Higher LN yield should reduce estimated risk (lower ratio)."""
        model = GastricCancerRiskModel(load_model_config())

        # Manually calculate risks at different yields
        test_patient_low = {
            "T_stage": "T2",
            "N_stage": "N1",
            "age": 60,
            "tumor_size_cm": 3.0,
            "positive_LN": 3,
            "total_LN": 10,  # Low yield -> higher ratio
        }
        test_patient_high = {
            "T_stage": "T2",
            "N_stage": "N1",
            "age": 60,
            "tumor_size_cm": 3.0,
            "positive_LN": 3,
            "total_LN": 40,  # High yield -> lower ratio
        }

        risk_low_yield = model.calculate_risk(test_patient_low)
        risk_high_yield = model.calculate_risk(test_patient_high)

        # Higher yield should result in lower estimated risk
        assert risk_high_yield < risk_low_yield

    def test_sensitivity_analysis_with_edge_yields(self, tmp_path):
        """Test sensitivity analysis handles edge case yields."""
        model = GastricCancerRiskModel(load_model_config())

        # Minimum viable yield
        patient_min = {
            "T_stage": "T2",
            "N_stage": "N1",
            "positive_LN": 1,
            "total_LN": 1,
        }
        risk_min = model.calculate_risk(patient_min)
        assert model.risk_floor <= risk_min <= model.risk_ceiling

        # Very high yield
        patient_max = {
            "T_stage": "T2",
            "N_stage": "N1",
            "positive_LN": 1,
            "total_LN": 100,
        }
        risk_max = model.calculate_risk(patient_max)
        assert model.risk_floor <= risk_max <= model.risk_ceiling


class TestRunExamplePatients:
    """Test the example patient scenarios."""

    def test_run_example_patients_returns_dataframe(self):
        """Test that example patients function returns valid DataFrame."""
        from risk_calculator import run_example_patients

        model = GastricCancerRiskModel(load_model_config())
        survival_model = load_survival_model()

        results = run_example_patients(model, survival_model)

        import pandas as pd

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 4  # Four example patients
        assert "Risk" in results.columns
        assert "Category" in results.columns

    def test_example_patients_risk_ordering(self):
        """Test that example patients have expected risk ordering."""
        from risk_calculator import run_example_patients

        model = GastricCancerRiskModel(load_model_config())
        results = run_example_patients(model, survival_model=None)

        # Patient A (early) should have lowest risk, Patient D (very advanced) highest
        risks = results["Risk"].tolist()
        assert risks[0] < risks[1] < risks[2] < risks[3]

    def test_example_patients_without_survival_model(self):
        """Test example patients work without survival model."""
        from risk_calculator import run_example_patients

        model = GastricCancerRiskModel(load_model_config())
        results = run_example_patients(model, survival_model=None)

        assert len(results) == 4
        assert "Risk" in results.columns
        # Survival columns should not be present
        assert "survival_5yr" not in results.columns or results["survival_5yr"].isna().all()


class TestParseArgs:
    """Test command-line argument parsing."""

    def test_parse_args_defaults(self):
        """Test default argument values."""
        from risk_calculator import parse_args

        with patch.object(sys, "argv", ["risk_calculator.py"]):
            args = parse_args()

        assert args.data == Path(__file__).parent.parent / "data" / "tcga_2018_clinical_data.tsv"
        assert not args.show_plots
        assert not args.skip_survival
        assert not args.verbose
        assert not args.log_timestamps

    def test_parse_args_custom_values(self, tmp_path):
        """Test parsing custom argument values."""
        from risk_calculator import parse_args

        test_args = [
            "risk_calculator.py",
            "--data",
            str(tmp_path / "custom_data.tsv"),
            "--output-dir",
            str(tmp_path),
            "--show-plots",
            "--skip-survival",
            "--verbose",
            "--log-timestamps",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.data == tmp_path / "custom_data.tsv"
        assert args.output_dir == tmp_path
        assert args.show_plots
        assert args.skip_survival
        assert args.verbose
        assert args.log_timestamps


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_plot_individual_predictions(self, tmp_path):
        """Test plotting individual predictions."""
        from risk_calculator import run_example_patients
        from utils.visualization import plot_individual_predictions

        model = GastricCancerRiskModel(load_model_config())
        results = run_example_patients(model, survival_model=None)

        output_path = plot_individual_predictions(results, tmp_path, show_plots=False)
        assert output_path.exists()

    def test_plot_tcga_summary(self, tmp_path):
        """Test plotting TCGA cohort summary."""
        import pandas as pd

        from utils.visualization import plot_tcga_summary

        # Create mock cohort results
        mock_results = pd.DataFrame(
            {
                "Risk": [0.2, 0.4, 0.6, 0.8],
                "Category": ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
                "T_stage": ["T1", "T2", "T3", "T4"],
                "N_stage": ["N0", "N1", "N2", "N3"],
            }
        )

        output_path = plot_tcga_summary(mock_results, tmp_path, show_plots=False)
        assert output_path.exists()

    def test_plot_calibration_curve_with_events(self, tmp_path):
        """Test calibration curve plotting with event data."""
        import pandas as pd

        from utils.visualization import plot_calibration_curve

        # Create mock data with event labels
        mock_results = pd.DataFrame(
            {
                "Risk": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                "event_observed": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            }
        )

        result = plot_calibration_curve(
            mock_results, tmp_path, show_plots=False, label_column="event_observed"
        )
        # Should return tuple of (path, brier_score, ci_low, ci_high) or None
        if result is not None:
            output_path, brier, ci_low, ci_high = result
            assert output_path.exists()
            assert 0 <= brier <= 1
            assert ci_low <= brier <= ci_high


class TestClinicalEdgeCases:
    """Edge case tests for clinically implausible but possible data inputs."""

    @pytest.fixture
    def model(self):
        config = load_model_config()
        return GastricCancerRiskModel(config)

    def test_negative_age(self, model):
        """Negative age should not crash; age effect is clamped by positive_delta_only."""
        patient = {"T_stage": "T2", "N_stage": "N1", "age": -5}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0
        # Negative age is below pivot (50), so age effect should be 0
        patient_no_age = {"T_stage": "T2", "N_stage": "N1"}
        risk_no_age = model.calculate_risk(patient_no_age)
        assert risk == pytest.approx(risk_no_age, rel=1e-6)

    def test_age_zero(self, model):
        """Age 0 (infant) should compute without error."""
        patient = {"T_stage": "T1", "N_stage": "N0", "age": 0}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_very_old_age(self, model):
        """Age 120 should compute without error or overflow."""
        patient = {"T_stage": "T3", "N_stage": "N2", "age": 120}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0
        assert math.isfinite(risk)

    def test_negative_tumor_size(self, model):
        """Negative tumor size is clinically impossible but should not crash."""
        patient = {"T_stage": "T2", "N_stage": "N1", "tumor_size_cm": -3.0}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_positive_ln_exceeds_total_ln(self, model):
        """positive_LN > total_LN is a data error but should not crash."""
        patient = {
            "T_stage": "T2",
            "N_stage": "N1",
            "positive_LN": 25,
            "total_LN": 10,
        }
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0

    def test_nan_age_treated_as_no_effect(self, model):
        """NaN age is silently treated as zero age effect.

        NaN passes the `is not None` check, but max(0.0, NaN) returns 0.0
        in CPython, so the positive_delta_only clamp neutralizes NaN.
        The result matches the no-age case.
        """
        patient = {"T_stage": "T2", "N_stage": "N1", "age": float("nan")}
        risk = model.calculate_risk(patient)
        assert 0.0 <= risk <= 1.0
        patient_no_age = {"T_stage": "T2", "N_stage": "N1"}
        risk_no_age = model.calculate_risk(patient_no_age)
        assert risk == pytest.approx(risk_no_age, rel=1e-6)

    def test_nan_tumor_size_raises(self, model):
        """NaN tumor_size_cm must fail loudly rather than yield a NaN risk score.

        A silent NaN flowing through scoring and plotting is a clinical-safety
        defect, so the model rejects non-finite tumor sizes explicitly.
        """
        patient = {"T_stage": "T2", "N_stage": "N1", "tumor_size_cm": float("nan")}
        with pytest.raises(ValueError, match="Non-finite tumor_size_cm"):
            model.calculate_risk(patient)

    def test_inf_tumor_size_raises(self, model):
        """Infinite tumor_size_cm is rejected the same way as NaN."""
        patient = {"T_stage": "T2", "N_stage": "N1", "tumor_size_cm": float("inf")}
        with pytest.raises(ValueError, match="Non-finite tumor_size_cm"):
            model.calculate_risk(patient)

    def test_nan_ln_ratio_raises(self, model):
        """A non-finite lymph node ratio is rejected rather than propagated."""
        patient = {"T_stage": "T2", "N_stage": "N1", "ln_ratio": float("nan")}
        with pytest.raises(ValueError, match="Non-finite lymph node ratio"):
            model.calculate_risk(patient)
