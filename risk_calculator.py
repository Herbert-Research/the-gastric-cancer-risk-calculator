"""
Gastric Cancer Risk Calculator
Author: Maximilian Dressler
Purpose: Implement published risk stratification models for gastric cancer
Based on: Simplified models from KLASS literature
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any

BASE_DIR = Path(__file__).resolve().parent
# Ensure Matplotlib cache uses a writable path (containers may block ~/.config).
mplt_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib")))
mplt_config_dir.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.constants import (  # noqa: E402
    DEFAULT_N_STAGE,
    DEFAULT_T_STAGE,
    N_STAGE_PRIOR_LN_RATIO,
    T_STAGE_PRIOR_SIZE,
)
from utils.logging_config import get_logger, setup_logging  # noqa: E402
from utils.visualization import (  # noqa: E402
    FIG_SENSITIVITY,
    finalize_figure,
    plot_calibration_curve,
    plot_individual_predictions,
    plot_survival_predictions,
    plot_survival_vs_recurrence,
    plot_tcga_summary,
)

if TYPE_CHECKING:
    from models.cox_model import CoxModel
    from models.variable_mapper_tcga import Han2012VariableMapper
DEFAULT_DATA_PATH = BASE_DIR / "data" / "tcga_2018_clinical_data.tsv"
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_MODEL_CONFIG = BASE_DIR / "models" / "heuristic_klass.json"
DEFAULT_SURVIVAL_MODEL_CONFIG = BASE_DIR / "models" / "han2012_jco.json"

logger = get_logger()

try:
    from models.cox_model import CoxModel
    from models.variable_mapper_tcga import Han2012VariableMapper

    COX_MODEL_AVAILABLE = True
except ImportError:
    COX_MODEL_AVAILABLE = False
    logger.warning("Cox survival components unavailable; use --skip-survival to suppress.")

DEFAULT_CONFIG_PAYLOAD = {
    "id": "heuristic_klass_v1",
    "name": "KLASS-inspired heuristic (logistic form)",
    "description": (
        "Educational logistic model using TN stage, age, tumor size, and LN ratio heuristics. "
        "Calibrate or replace with institution-specific coefficients for production use."
    ),
    "citation": "Placeholder heuristics by M.H. Dressler (2025), inspired by KLASS literature summaries.",
    "intercept": -2.25,
    "risk_floor": 0.02,
    "risk_ceiling": 0.95,
    "t_stage_weights": {"T1": 0.0, "T2": 0.9, "T3": 1.6, "T4": 2.2},
    "n_stage_weights": {"N0": 0.0, "N1": 1.1, "N2": 1.9, "N3": 2.7},
    "age_weight": {"weight": 0.018, "pivot": 50},
    "tumor_size_weight": {"weight": 0.12},
    "ln_ratio_weight": 2.4,
}


def sigmoid(value: float) -> float:
    """
    Apply numerically stable logistic (sigmoid) transformation.

    Computes the logistic function σ(x) = 1/(1 + e^{-x}) using a branch
    that avoids overflow for large negative values.

    Parameters
    ----------
    value : float
        Input value (linear predictor or logit).

    Returns
    -------
    float
        Probability in range (0, 1).

    Notes
    -----
    For numerical stability:
    - When value ≥ 0: σ(x) = 1 / (1 + e^{-x})
    - When value < 0: σ(x) = e^x / (1 + e^x)

    This avoids computing e^{large_positive} which would overflow.

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> sigmoid(2.0)
    0.8807970779778823
    >>> sigmoid(-710)  # Large negative value, no overflow
    0.0
    """
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def load_model_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load model configuration from a JSON file.

    Reads a JSON configuration file containing logistic regression coefficients
    for the heuristic recurrence model. Falls back to bundled default config
    if the file is missing or malformed.

    Parameters
    ----------
    config_path : Path or None, optional
        Absolute path to the JSON configuration file. If None, uses
        ``DEFAULT_MODEL_CONFIG`` (models/heuristic_klass.json).

    Returns
    -------
    dict[str, Any]
        Configuration dictionary containing:
        - id : str - Model identifier
        - name : str - Human-readable model name
        - intercept : float - Logistic regression intercept (β₀)
        - t_stage_weights : dict[str, float] - T-stage coefficients
        - n_stage_weights : dict[str, float] - N-stage coefficients
        - age_weight : dict - Age effect configuration
        - tumor_size_weight : dict - Tumor size coefficient
        - ln_ratio_weight : float - Lymph node ratio coefficient
        - risk_floor, risk_ceiling : float - Output probability bounds

    See Also
    --------
    load_survival_model : Load Cox proportional hazards model.
    GastricCancerRiskModel : Model class that consumes this configuration.

    Examples
    --------
    >>> config = load_model_config()
    >>> config['id']
    'heuristic_klass_v1'
    >>> config = load_model_config(Path('custom_model.json'))
    """
    payload: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG_PAYLOAD)
    target_path = config_path or DEFAULT_MODEL_CONFIG

    if target_path and target_path.exists():
        try:
            with target_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Unable to parse %s: %s. Falling back to bundled config.", target_path, exc
            )
        except OSError as exc:
            logger.warning(
                "Unable to read %s: %s. Falling back to bundled config.", target_path, exc
            )

    return payload


def load_survival_model(config_path: Path | None = None) -> CoxModel | None:
    """
    Load Han 2012 Cox proportional hazards survival model.

    Initializes a CoxModel instance from a JSON configuration file containing
    the published regression coefficients from Han et al. (JCO 2012).

    Parameters
    ----------
    config_path : Path or None, optional
        Absolute path to the Cox model JSON configuration. If None, uses
        ``DEFAULT_SURVIVAL_MODEL_CONFIG`` (models/han2012_jco.json).

    Returns
    -------
    CoxModel or None
        Initialized CoxModel instance for survival prediction, or None if:
        - Cox model components are not available (import failed)
        - Configuration file does not exist
        - Configuration file is malformed

    Notes
    -----
    The Han 2012 model predicts overall survival after D2 gastrectomy using:
    - Age category (5 levels)
    - Sex (male/female)
    - Tumor location (upper/middle/lower)
    - Depth of invasion (5 levels)
    - Metastatic lymph nodes (5 categories)
    - Examined lymph nodes (continuous)

    **Important:** Baseline survival S₀(t) is estimated, not from the original
    publication. Institutional recalibration is required before clinical use.

    See Also
    --------
    load_model_config : Load heuristic recurrence model configuration.
    CoxModel : The Cox model class.
    Han2012VariableMapper : Maps patient data to Han 2012 format.

    References
    ----------
    Han DS, et al. Nomogram predicting long-term survival after D2 gastrectomy
    for gastric cancer. J Clin Oncol. 2012;30(31):3834-40.
    """
    if not COX_MODEL_AVAILABLE:
        return None

    target_path = config_path or DEFAULT_SURVIVAL_MODEL_CONFIG
    if not target_path or not target_path.exists():
        logger.warning("Survival model config not found at %s", target_path)
        return None

    try:
        with target_path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        return CoxModel(config)
    except Exception as exc:  # pragma: no cover - invalid external config path
        logger.exception("Unable to load survival model: %s", exc)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gastric cancer risk predictions and cohort visualisations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the TCGA clinical TSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated figures (default: project root).",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display matplotlib figures after saving them.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to JSON file containing logistic model coefficients (default: %(default)s).",
    )
    parser.add_argument(
        "--survival-model",
        type=Path,
        default=DEFAULT_SURVIVAL_MODEL_CONFIG,
        help="Path to Cox survival model config (Han 2012). Default: %(default)s.",
    )
    parser.add_argument(
        "--skip-survival",
        action="store_true",
        help="Skip survival predictions (recurrence only).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging output.",
    )
    parser.add_argument(
        "--log-timestamps",
        action="store_true",
        help="Include timestamps in log output (disabled by default for reproducible logs).",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float:
    """Return a Python float from numpy/pandas scalar wrappers."""

    if isinstance(value, Number):
        return float(value)  # type: ignore[arg-type]

    item = getattr(value, "item", None)
    if callable(item):
        inner = item()
        if isinstance(inner, Number):
            return float(inner)  # type: ignore[arg-type]

    return float(value)


class GastricCancerRiskModel:
    """Configurable logistic model based on TNM staging and clinical factors."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.t_stage_weights = config.get("t_stage_weights", {})
        self.n_stage_weights = config.get("n_stage_weights", {})
        self.age_weight_cfg = config.get("age_weight", {})
        self.tumor_weight_cfg = config.get("tumor_size_weight", {})
        self.ln_ratio_weight = float(config.get("ln_ratio_weight", 0.0))
        self.intercept = float(config.get("intercept", 0.0))
        self.risk_floor = float(config.get("risk_floor", 0.0))
        self.risk_ceiling = float(config.get("risk_ceiling", 1.0))

    def _validate_stage(self, stage: str | None, stage_type: str) -> str:
        if stage_type == "T":
            valid = self.t_stage_weights
        else:
            valid = self.n_stage_weights
        if stage not in valid:
            raise ValueError(f"Unsupported {stage_type} stage: {stage}")
        return stage  # type: ignore[return-value]

    def calculate_risk(self, patient_data: dict[str, Any]) -> float:
        """
        Calculate 5-year recurrence risk for an individual patient.

        Implements a logistic regression model with the functional form:

            logit(p) = β₀ + β_T·T + β_N·N + β_age·(age-50)⁺ + β_size·size + β_ratio·LN_ratio

        Parameters
        ----------
        patient_data : dict[str, Any]
            Patient clinical variables containing:

            - T_stage : str
                Tumor stage (T1, T2, T3, or T4)
            - N_stage : str
                Nodal stage (N0, N1, N2, or N3)
            - age : float, optional
                Patient age in years
            - tumor_size_cm : float, optional
                Tumor size in centimeters
            - ln_ratio : float, optional
                Lymph node ratio (positive/total), or provide positive_LN and total_LN

        Returns
        -------
        float
            Probability of 5-year recurrence, bounded by [risk_floor, risk_ceiling].

        Raises
        ------
        ValueError
            If T_stage or N_stage is not in the configured valid stages, or if
            ``tumor_size_cm`` or the resolved lymph node ratio is non-finite
            (NaN or infinite). Failing loudly prevents a NaN risk score from
            silently propagating through the scoring and plotting pipeline.

        Examples
        --------
        >>> model = GastricCancerRiskModel(config)
        >>> patient = {"T_stage": "T2", "N_stage": "N1", "age": 60}
        >>> risk = model.calculate_risk(patient)
        >>> 0.0 <= risk <= 1.0
        True

        Notes
        -----
        This is an educational model. Coefficients are pedagogical approximations
        and require institutional calibration before any clinical application.

        See Also
        --------
        CoxModel.calculate_survival : For overall survival predictions.
        """
        t_stage = self._validate_stage(patient_data.get("T_stage"), "T")
        n_stage = self._validate_stage(patient_data.get("N_stage"), "N")

        logit = self.intercept
        logit += float(self.t_stage_weights[t_stage])
        logit += float(self.n_stage_weights[n_stage])

        age = patient_data.get("age")
        if age is not None:
            logit += self._apply_age_effect(float(age))

        tumor_size = patient_data.get("tumor_size_cm")
        if tumor_size is not None:
            tumor_size = float(tumor_size)
            if not math.isfinite(tumor_size):
                raise ValueError(f"Non-finite tumor_size_cm: {tumor_size!r}")
            logit += tumor_size * float(self.tumor_weight_cfg.get("weight", 0.0))

        ln_ratio = resolve_ln_ratio(
            patient_data.get("ln_ratio"),
            patient_data.get("positive_LN"),
            patient_data.get("total_LN"),
        )
        if ln_ratio is not None:
            ln_ratio = float(ln_ratio)
            if not math.isfinite(ln_ratio):
                raise ValueError(f"Non-finite lymph node ratio: {ln_ratio!r}")
            bounded_ratio = max(0.0, min(1.0, ln_ratio))
            logit += bounded_ratio * self.ln_ratio_weight

        risk = sigmoid(logit)
        if self.risk_ceiling:
            risk = min(risk, self.risk_ceiling)
        if self.risk_floor:
            risk = max(risk, self.risk_floor)
        return float(risk)

    def _apply_age_effect(self, age: float) -> float:
        weight = float(self.age_weight_cfg.get("weight", 0.0))
        if weight == 0.0:
            return 0.0
        pivot = float(self.age_weight_cfg.get("pivot", 0.0))
        delta = age - pivot
        if self.age_weight_cfg.get("positive_delta_only", True):
            delta = max(0.0, delta)
        return delta * weight

    @staticmethod
    def risk_category(risk: float) -> str:
        """Categorize risk level."""

        if risk < 0.20:
            return "Low Risk"
        if risk < 0.40:
            return "Moderate Risk"
        if risk < 0.60:
            return "High Risk"
        return "Very High Risk"


def resolve_ln_ratio(
    ln_ratio: float | None, positive_ln: float | None, total_ln: float | None
) -> float | None:
    """Derive LN ratio from explicit counts if direct value missing."""

    if ln_ratio is not None:
        return float(ln_ratio)
    if total_ln is None or total_ln == 0:
        return None
    positive_ln = positive_ln if positive_ln is not None else 0.0
    if total_ln <= 0:
        return None
    return float(positive_ln) / float(total_ln)


def score_patients(model: GastricCancerRiskModel, patients: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a dataframe with risk predictions for each patient input."""

    rows = []
    for patient in patients:
        risk = model.calculate_risk(patient)
        # Compute resolved LN ratio for consistent reporting
        resolved_ln_ratio = resolve_ln_ratio(
            patient.get("ln_ratio"),
            patient.get("positive_LN"),
            patient.get("total_LN"),
        )
        rows.append(
            {
                "Patient": patient.get("name")
                or patient.get("patient_id")
                or patient.get("id", "Patient"),
                "Risk": risk,
                "Category": model.risk_category(risk),
                "T_stage": patient.get("T_stage"),
                "N_stage": patient.get("N_stage"),
                "age": patient.get("age"),
                "tumor_size_cm": patient.get("tumor_size_cm"),
                "ln_ratio": resolved_ln_ratio,
                "tumor_size_imputed": patient.get("tumor_size_imputed", False),
                "ln_ratio_imputed": patient.get("ln_ratio_imputed", False),
                "positive_LN": patient.get("positive_LN"),
                "total_LN": patient.get("total_LN"),
                "Sex": patient.get("Sex") or patient.get("sex"),
            }
        )

    return pd.DataFrame(rows)


def predict_with_both_models(
    patients: list[dict[str, Any]],
    recurrence_model: GastricCancerRiskModel,
    survival_model: CoxModel | None,
) -> pd.DataFrame:
    """
    Score patients with both recurrence and survival models.

    Applies the heuristic logistic recurrence model and (optionally) the
    Han 2012 Cox survival model to generate dual-endpoint predictions.

    Parameters
    ----------
    patients : list[dict[str, Any]]
        List of patient dictionaries, each containing clinical variables:
        - T_stage, N_stage : str (required)
        - age : float (optional)
        - Sex : str (optional, for survival model)
        - tumor_size_cm, ln_ratio, positive_LN, total_LN : float (optional)
        - icd_10 : str (optional, for tumor location inference)

    recurrence_model : GastricCancerRiskModel
        Initialized heuristic recurrence risk model.

    survival_model : CoxModel or None
        Initialized Han 2012 Cox model, or None to skip survival predictions.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Patient, Risk, Category : Recurrence model outputs
        - T_stage, N_stage, age, tumor_size_cm, ln_ratio : Input variables
        - tumor_size_imputed, ln_ratio_imputed : Imputation flags
        - survival_5yr, survival_10yr : Survival probabilities (if model provided)
        - survival_category, survival_summary : Prognosis labels (if model provided)
        - survival_imputations : Dict of imputation flags per patient

    Notes
    -----
    The Han 2012 survival model requires variable mapping via
    ``Han2012VariableMapper``, which handles TCGA-to-nomogram translation
    including tumor location imputation.

    See Also
    --------
    score_patients : Recurrence-only scoring.
    Han2012VariableMapper.map_patient_from_dict : Variable mapping logic.
    """
    results_df = score_patients(recurrence_model, patients)

    if survival_model and COX_MODEL_AVAILABLE:
        mapper = Han2012VariableMapper()
        survival_5yr: list[float | None] = []
        survival_10yr: list[float | None] = []
        survival_category: list[str | None] = []
        survival_desc: list[str | None] = []
        mapper_flags: list[dict[str, bool]] = []

        for patient in patients:
            try:
                mapped_fields = mapper.map_patient_from_dict(patient)
                mapper_flags.append(mapper.get_imputation_flags(patient))
                survival_probs = survival_model.predict_patient_survival(mapped_fields)
                surv5 = survival_probs.get(5)
                surv10 = survival_probs.get(10)
                survival_5yr.append(surv5)
                survival_10yr.append(surv10)
                if surv5 is not None:
                    category, description = survival_model.categorize_risk(surv5)
                    survival_category.append(category)
                    survival_desc.append(description)
                else:
                    survival_category.append(None)
                    survival_desc.append(None)
            except Exception as exc:  # pragma: no cover - patient-level logging path
                logger.exception(
                    "Survival prediction failed for patient %s: %s", patient.get("name"), exc
                )
                survival_5yr.append(None)
                survival_10yr.append(None)
                survival_category.append(None)
                survival_desc.append(None)
                mapper_flags.append(
                    {
                        "age_available": patient.get("age") is not None,
                        "sex_available": bool(patient.get("Sex") or patient.get("sex")),
                        "location_imputed": True,
                        "positive_ln_imputed": patient.get("positive_LN") is None,
                        "examined_ln_imputed": patient.get("total_LN") is None,
                    }
                )

        results_df["survival_5yr"] = survival_5yr
        results_df["survival_10yr"] = survival_10yr
        results_df["survival_category"] = survival_category
        results_df["survival_summary"] = survival_desc
        results_df["survival_imputations"] = mapper_flags

    return results_df


def print_survival_summary(results_df: pd.DataFrame) -> None:
    """Print summary statistics for Han 2012 survival predictions."""

    if "survival_5yr" not in results_df.columns:
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("⚠️  HAN 2012 SURVIVAL MODEL - CALIBRATION STATUS")
    logger.info("-" * 60)
    logger.info("IMPORTANT: These predictions use estimated baseline survival")
    logger.info("S₀(t) calibrated to match published cohort statistics (Han 2012).")
    logger.info("Individual predictions may differ from validated nomogram performance.")
    logger.info("Institutional recalibration required before any clinical use.")
    logger.info("=" * 60)

    logger.info("")
    logger.info("=" * 60)
    logger.info("HAN 2012 SURVIVAL MODEL SUMMARY")
    logger.info("-" * 60)

    surv5 = results_df["survival_5yr"].dropna()
    if not surv5.empty:
        logger.info("5-Year Survival:")
        logger.info("  Mean:   %.1f%%", surv5.mean() * 100)
        logger.info("  Median: %.1f%%", surv5.median() * 100)
        logger.info("  Range:  %.1f%% to %.1f%%", surv5.min() * 100, surv5.max() * 100)

    surv10 = (
        results_df["survival_10yr"].dropna()
        if "survival_10yr" in results_df
        else pd.Series(dtype=float)
    )
    if not surv10.empty:
        logger.info("")
        logger.info("10-Year Survival:")
        logger.info("  Mean:   %.1f%%", surv10.mean() * 100)
        logger.info("  Median: %.1f%%", surv10.median() * 100)
        logger.info("  Range:  %.1f%% to %.1f%%", surv10.min() * 100, surv10.max() * 100)

    if "survival_category" in results_df.columns:
        counts = results_df["survival_category"].value_counts()
        if not counts.empty:
            logger.info("")
            logger.info("Prognosis Categories:")
            for cat, count in counts.items():
                pct = count / len(results_df) * 100
                logger.info("  %s: %d (%.1f%%)", cat, count, pct)

    if {"Risk", "survival_5yr"}.issubset(results_df.columns):
        corr_df = results_df[["Risk", "survival_5yr"]].dropna()
        if len(corr_df) > 10:
            from utils.bootstrap import bootstrap_correlation

            corr, ci_low, ci_high = bootstrap_correlation(
                corr_df["Risk"].to_numpy(),
                corr_df["survival_5yr"].to_numpy(),
                n_bootstrap=1000,
                random_state=42,
            )
            logger.info("")
            logger.info(
                "Correlation (Recurrence Risk vs Survival): %.3f (95%% CI: %.3f–%.3f)",
                corr,
                ci_low,
                ci_high,
            )
            if corr < -0.5:
                logger.info("  ✓ Strong inverse relationship (as expected)")
            elif corr < -0.3:
                logger.info("  ⚠ Moderate inverse relationship")
            else:
                logger.info("  Note: Weak correlation – investigate cohort differences.")


def run_example_patients(
    recurrence_model: GastricCancerRiskModel, survival_model: CoxModel | None = None
) -> pd.DataFrame:
    """Reproduce the illustrative patient scenarios used in the README."""

    patients = [
        {
            "name": "Patient A - Early Stage",
            "T_stage": "T1",
            "N_stage": "N0",
            "age": 55,
            "Sex": "Female",
            "tumor_size_cm": 2.0,
            "positive_LN": 0,
            "total_LN": 20,
        },
        {
            "name": "Patient B - Moderate Stage",
            "T_stage": "T2",
            "N_stage": "N1",
            "age": 62,
            "Sex": "Male",
            "tumor_size_cm": 3.5,
            "positive_LN": 2,
            "total_LN": 25,
        },
        {
            "name": "Patient C - Advanced Stage",
            "T_stage": "T3",
            "N_stage": "N2",
            "age": 68,
            "Sex": "Female",
            "tumor_size_cm": 5.0,
            "positive_LN": 8,
            "total_LN": 30,
        },
        {
            "name": "Patient D - Very Advanced",
            "T_stage": "T4",
            "N_stage": "N3",
            "age": 70,
            "Sex": "Male",
            "tumor_size_cm": 6.5,
            "positive_LN": 15,
            "total_LN": 32,
        },
    ]

    results_df = predict_with_both_models(patients, recurrence_model, survival_model)

    for row in results_df.itertuples(index=False):
        logger.info("")
        logger.info("%s", row.Patient)
        logger.info("  Stage: %s%s", row.T_stage, row.N_stage)
        risk_pct = safe_float(row.Risk) * 100.0
        logger.info("  5-Year Recurrence Risk: %.1f%% (%s)", risk_pct, row.Category)

        if survival_model and hasattr(row, "survival_5yr") and row.survival_5yr:
            surv_pct = safe_float(row.survival_5yr) * 100.0
            category = getattr(row, "survival_category", "N/A")
            logger.info("  5-Year Survival: %.1f%% (%s)", surv_pct, category)

    return results_df


def run_sensitivity_analysis(
    model: GastricCancerRiskModel, output_dir: Path, show_plots: bool
) -> Path:
    """Illustrate how nodal yield impacts predictions."""

    logger.info("")
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS: Impact of Lymph Node Yield")
    logger.info("-" * 60)

    test_patient = {
        "T_stage": "T2",
        "N_stage": "N1",
        "age": 60,
        "tumor_size_cm": 3.0,
        "positive_LN": 3,
        "total_LN": 15,
    }

    ln_yields = range(10, 41, 5)
    risks_by_yield = []

    for ln_yield in ln_yields:
        test_patient["total_LN"] = ln_yield
        risk = model.calculate_risk(test_patient)
        risks_by_yield.append(risk)
        logger.info("LN Yield = %2d -> Risk = %.1f%%", ln_yield, risk * 100)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ln_yields, [r * 100 for r in risks_by_yield], "o-", linewidth=2, markersize=8)
    plt.xlabel("Total Lymph Nodes Retrieved", fontsize=12)
    plt.ylabel("5-Year Recurrence Risk (%)", fontsize=12)
    plt.title(
        "Impact of Lymph Node Yield on Risk Prediction\n(Patient: T2N1, 3 positive nodes)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = finalize_figure(fig, output_dir / FIG_SENSITIVITY, show_plots)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Key Insight: Higher LN yield reduces estimated risk due to")
    logger.info("lower positive/total ratio, highlighting importance of")
    logger.info("adequate D2 dissection for accurate staging.")
    return output_path


def load_tcga_cohort(data_path: Path) -> pd.DataFrame:
    """Load and harmonize the anonymized TCGA cohort."""

    if not data_path.exists():
        logger.warning("TCGA file not found at %s.", data_path)
        return pd.DataFrame()

    rename_map = {
        "Patient ID": "patient_id",
        "Diagnosis Age": "age",
        "American Joint Committee on Cancer Tumor Stage Code": "raw_t_stage",
        "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code": "raw_n_stage",
        "Neoadjuvant Therapy Type Administered Prior To Resection Text": "neoadjuvant",
        "Subtype": "molecular_subtype",
        "Disease Free Status": "disease_free_status",
        "Progression Free Status": "progression_free_status",
        "Sex": "sex",
        "ICD-10 Classification": "icd_10",
    }

    try:
        df = pd.read_csv(data_path, sep="\t")
    except Exception as exc:  # pragma: no cover - external TSV parsing failure
        logger.warning("Unable to parse TCGA cohort: %s", exc)
        return pd.DataFrame()

    cohort = df.rename(columns=rename_map)
    cohort["age"] = pd.to_numeric(cohort["age"], errors="coerce")
    cohort["T_stage"] = cohort["raw_t_stage"].apply(normalize_t_stage).fillna(DEFAULT_T_STAGE)
    cohort["N_stage"] = cohort["raw_n_stage"].apply(normalize_n_stage).fillna(DEFAULT_N_STAGE)

    tumor_size, tumor_imputed = _resolve_tumor_size_feature(cohort)
    ln_ratio, ln_imputed, positive_ln, total_ln = _resolve_ln_ratio_feature(cohort)

    cohort["tumor_size_cm"] = tumor_size
    cohort["ln_ratio"] = ln_ratio
    cohort["tumor_size_imputed"] = tumor_imputed
    cohort["ln_ratio_imputed"] = ln_imputed
    cohort["positive_LN"] = positive_ln
    cohort["total_LN"] = total_ln
    status_series = cohort.get("disease_free_status")
    if status_series is None:
        status_series = pd.Series(index=cohort.index, dtype=float)
    cohort["event_observed"] = status_series.apply(parse_event_status)
    cohort = cohort.dropna(subset=["age"])

    drop_cols = [col for col in ["raw_t_stage", "raw_n_stage"] if col in cohort.columns]
    return cohort.drop(columns=drop_cols)


def normalize_t_stage(value: Any) -> str | None:
    """Return simplified T stage (T1-T4)."""

    stage = _normalize_stage(value, "T")
    return stage if stage in T_STAGE_PRIOR_SIZE else None


def normalize_n_stage(value: Any) -> str | None:
    """Return simplified N stage (N0-N3)."""

    stage = _normalize_stage(value, "N")
    return stage if stage in N_STAGE_PRIOR_LN_RATIO else None


def _normalize_stage(value: Any, prefix: str) -> str | None:
    if pd.isna(value):
        return None
    cleaned = str(value).strip().upper()
    if not cleaned.startswith(prefix):
        return None
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    if digits:
        return f"{prefix}{digits}"
    return None


def estimate_tumor_size(t_stage: str) -> float:
    """Use a stage-informed proxy for tumor size when not provided."""

    return T_STAGE_PRIOR_SIZE.get(t_stage, T_STAGE_PRIOR_SIZE[DEFAULT_T_STAGE])


def estimate_ln_ratio(n_stage: str) -> float:
    """Approximate LN ratio using stage-informed priors."""

    return N_STAGE_PRIOR_LN_RATIO.get(n_stage, N_STAGE_PRIOR_LN_RATIO[DEFAULT_N_STAGE])


def _resolve_tumor_size_feature(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return tumor size (cm) and boolean mask of imputed records."""

    candidate_columns = [
        "tumor_size_cm",
        "Tumor Size (cm)",
        "Tumor size (cm)",
        "Tumor dimension (cm)",
        "Tumor Dimension (cm)",
        "Pathologic Tumor Largest Dimension (cm)",
    ]

    tumor_series = pd.Series(np.nan, index=df.index, dtype=float)
    for column in candidate_columns:
        if column in df.columns:
            tumor_series = pd.to_numeric(df[column], errors="coerce")
            break

    imputed_mask = tumor_series.isna()
    fallback = df.loc[imputed_mask, "T_stage"].apply(estimate_tumor_size)
    tumor_series.loc[imputed_mask] = fallback
    return tumor_series, imputed_mask


def _resolve_ln_ratio_feature(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return LN ratio, imputation mask, and any parsed positive/total LN counts."""

    positive_candidates = [
        "Number of Lymph Nodes Positive",
        "Regional Nodes Positive",
        "Positive Lymph Nodes",
        "lymph_nodes_positive",
    ]
    total_candidates = [
        "Number of Lymph Nodes Examined",
        "Regional Nodes Examined",
        "Total Lymph Nodes",
        "lymph_nodes_examined",
    ]

    positive_ln = _first_numeric_column(df, positive_candidates)
    total_ln = _first_numeric_column(df, total_candidates)

    ratio = pd.Series(np.nan, index=df.index, dtype=float)
    if positive_ln is not None and total_ln is not None:
        valid = (total_ln > 0) & positive_ln.notna()
        ratio.loc[valid] = (positive_ln[valid] / total_ln[valid]).clip(0.0, 1.0)

    imputed_mask = ratio.isna()
    ratio.loc[imputed_mask] = df.loc[imputed_mask, "N_stage"].apply(estimate_ln_ratio)
    if positive_ln is None:
        positive_ln = pd.Series(np.nan, index=df.index, dtype=float)
    if total_ln is None:
        total_ln = pd.Series(np.nan, index=df.index, dtype=float)
    return ratio, imputed_mask, positive_ln, total_ln


def _first_numeric_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return None


def parse_event_status(value: Any) -> float | None:
    """Derive binary event flag from TCGA textual status columns."""

    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("0") or "diseasefree" in text or "tumor free" in text or "censored" in text:
        return 0.0
    if text.startswith("1") or "progression" in text or "recurr" in text or "with tumor" in text:
        return 1.0
    return None


def analyze_tcga_cohort(
    model: GastricCancerRiskModel,
    data_path: Path,
    output_dir: Path,
    show_plots: bool,
    survival_model: CoxModel | None = None,
) -> list[Path]:
    """
    Score the TCGA STAD cohort and generate validation analyses.

    Loads the TCGA stomach adenocarcinoma clinical dataset, applies both
    prediction models, and generates publication-quality figures for
    calibration assessment and cohort-level validation.

    Parameters
    ----------
    model : GastricCancerRiskModel
        Initialized heuristic recurrence risk model.

    data_path : Path
        Path to the TCGA clinical data TSV file.

    output_dir : Path
        Directory for saving generated figures.

    show_plots : bool
        If True, display figures interactively after saving.

    survival_model : CoxModel or None, optional
        Han 2012 Cox survival model. If provided, generates survival
        distribution plots and cross-model correlation analysis.

    Returns
    -------
    list[Path]
        Paths to generated figure files:
        - tcga_cohort_summary.png : Stage-stratified risk heatmap
        - calibration_curve.png : Model calibration assessment
        - survival_predictions_han2012.png : Survival distributions (if model)
        - survival_vs_recurrence_comparison.png : Cross-model comparison (if model)

    Notes
    -----
    **Data Quality Warning:** TCGA clinical data requires 100% imputation for
    surgical variables (tumor size, LN counts, tumor location). Predictions
    represent stage-typical risk, not patient-specific estimates.

    **Calibration Interpretation:** The Brier score assesses recurrence
    predictions against disease-free survival outcomes. Poor calibration
    reflects endpoint mismatch, not model failure.

    See Also
    --------
    load_tcga_cohort : Data loading and harmonization.
    plot_calibration_curve : Calibration visualization.
    print_survival_summary : Survival model summary statistics.
    """
    cohort = load_tcga_cohort(data_path)
    if cohort.empty:
        logger.warning("TCGA cohort not analyzed (file missing or parsing failed).")
        return []

    sex_col = None
    for candidate in ["Sex", "sex", "Gender", "gender"]:
        if candidate in cohort.columns:
            sex_col = candidate
            break
    if sex_col:
        cohort["Sex"] = cohort[sex_col]
    else:
        cohort["Sex"] = "Male"
        logger.warning("Sex column not found; defaulting to Male for survival predictions.")

    patient_inputs = []
    for row in cohort.itertuples(index=False):
        patient_inputs.append(
            {
                "name": row.patient_id,
                "T_stage": row.T_stage,
                "N_stage": row.N_stage,
                "age": row.age,
                "Sex": getattr(row, "Sex", "Male"),
                "tumor_size_cm": row.tumor_size_cm,
                "ln_ratio": row.ln_ratio,
                "tumor_size_imputed": getattr(row, "tumor_size_imputed", False),
                "ln_ratio_imputed": getattr(row, "ln_ratio_imputed", False),
                "positive_LN": getattr(row, "positive_LN", None),
                "total_LN": getattr(row, "total_LN", None),
                "neoadjuvant": getattr(row, "neoadjuvant", None),
                "molecular_subtype": getattr(row, "molecular_subtype", None),
                "event_observed": getattr(row, "event_observed", None),
                "icd_10": getattr(row, "icd_10", None),
            }
        )

    cohort_results = predict_with_both_models(patient_inputs, model, survival_model)
    cohort_results = cohort_results.merge(
        cohort[["patient_id", "neoadjuvant", "molecular_subtype", "event_observed"]],
        how="left",
        left_on="Patient",
        right_on="patient_id",
    ).drop(columns=["patient_id"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("TCGA-2018 Clinical Cohort Integration")
    logger.info("-" * 60)
    logger.info("Patients scored: %d", len(cohort_results))
    logger.info("Median predicted risk: %.1f%%", cohort_results["Risk"].median() * 100)

    category_counts = cohort_results["Category"].value_counts().sort_index()
    for label, count in category_counts.items():
        logger.info("  %-12s: %d", label, count)

    subtype_counts = cohort_results["molecular_subtype"].fillna("Unknown").value_counts().head(3)
    logger.info("Top molecular subtypes represented:")
    for label, count in subtype_counts.items():
        logger.info("  %-12s: %d", label, count)

    tumor_imputed_pct = cohort_results["tumor_size_imputed"].mean() * 100
    ln_imputed_pct = cohort_results["ln_ratio_imputed"].mean() * 100
    logger.info("")
    logger.info("Data Quality Assessment:")
    logger.info("-" * 60)
    logger.info("  Tumor size imputed: %.1f%% (stage-informed estimates)", tumor_imputed_pct)
    logger.info("  LN ratio imputed: %.1f%% (N-stage-derived)", ln_imputed_pct)
    logger.info("  Tumor location imputed: 100.0% (epidemiological priors)")

    if tumor_imputed_pct > 90:
        logger.warning("")
        logger.warning("⚠️  CRITICAL: >90% variable imputation detected.")
        logger.warning("   Predictions represent stage-typical, not patient-specific, risk.")
        logger.warning("   Suitable for cohort-level validation only.")

    generated_paths: list[Path] = []
    summary_fig = plot_tcga_summary(cohort_results, output_dir, show_plots)
    generated_paths.append(summary_fig)

    calibration_result = plot_calibration_curve(
        cohort_results, output_dir, show_plots, label_column="event_observed"
    )
    if calibration_result:
        calibration_fig, brier, brier_ci_low, brier_ci_high = calibration_result
        generated_paths.append(calibration_fig)
        logger.info(
            "Brier score (recurrence model vs. DFS): %.3f (95%% CI: %.3f–%.3f)",
            brier,
            brier_ci_low,
            brier_ci_high,
        )
        logger.info("⚠️  Note: Poor calibration reflects outcome mismatch, not model failure.")
        logger.info("    The model predicts recurrence; TCGA provides disease-free survival.")
        logger.info("    These are related but distinct clinical endpoints.")

    if survival_model and "survival_5yr" in cohort_results.columns:
        survival_fig = plot_survival_predictions(cohort_results, output_dir, show_plots)
        if survival_fig:
            generated_paths.append(survival_fig)
        comparison_fig = plot_survival_vs_recurrence(cohort_results, output_dir, show_plots)
        if comparison_fig:
            generated_paths.append(comparison_fig)
        print_survival_summary(cohort_results)

        # Compute C-index (discrimination) for survival model against DFS proxy
        if "event_observed" in cohort_results.columns:
            from utils.bootstrap import concordance_index

            cindex_df = cohort_results[["survival_5yr", "event_observed"]].dropna()
            if len(cindex_df) > 10:
                c_idx = concordance_index(
                    cindex_df["event_observed"].to_numpy(),
                    (1 - cindex_df["survival_5yr"]).to_numpy(),
                )
                logger.info("")
                logger.info("Han 2012 C-index (vs. DFS proxy): %.3f", c_idx)
                if c_idx > 0.65:
                    logger.info("  ✓ Acceptable discrimination despite imputed variables")
                else:
                    logger.info("  ⚠ Limited discrimination (expected with 100% imputation)")

    return generated_paths


def main() -> None:
    args = parse_args()

    main_logger = setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        include_timestamp=args.log_timestamps,
    )

    # Ensure reproducible tumor location imputation across runs
    from models.variable_mapper_tcga import reset_imputation_seed

    reset_imputation_seed(42)

    main_logger.info("Gastric Cancer Risk Calculator (Dual Model)")
    main_logger.info("=" * 60)
    main_logger.info("Data path: %s", args.data)
    main_logger.info("Output directory: %s", args.output_dir)

    model_config = load_model_config(args.model_config)
    main_logger.info(
        "Recurrence model: %s – %s",
        model_config.get("id", "custom"),
        model_config.get("name", "N/A"),
    )
    recurrence_model = GastricCancerRiskModel(model_config)

    survival_model = None
    if args.skip_survival:
        main_logger.info("Survival model: Skipped (flag provided)")
    else:
        survival_model = load_survival_model(args.survival_model)
        if survival_model:
            main_logger.info("Survival model: Han 2012 D2 Gastrectomy Nomogram")
        else:
            main_logger.warning("Survival model: Not available (recurrence only)")

    example_results = run_example_patients(recurrence_model, survival_model)

    generated_files: list[Path] = []
    generated_files.append(
        plot_individual_predictions(example_results, args.output_dir, args.show_plots)
    )
    generated_files.append(
        run_sensitivity_analysis(recurrence_model, args.output_dir, args.show_plots)
    )

    tcga_figs = analyze_tcga_cohort(
        recurrence_model, args.data, args.output_dir, args.show_plots, survival_model
    )
    if tcga_figs:
        generated_files.extend(tcga_figs)

    main_logger.info("")
    main_logger.info("Generated files:")
    for path in generated_files:
        main_logger.info("  - %s", path)


if __name__ == "__main__":
    main()
