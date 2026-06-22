"""
Cox Proportional Hazards Model Support
"""

from __future__ import annotations

import math
from typing import Any


class CoxModel:
    """
    Cox proportional hazards model for survival prediction.
    Follows same pattern as GastricCancerRiskModel for consistency.
    """

    PROGNOSIS_DESCRIPTIONS = {
        "Excellent Prognosis": "≥85% 5-year survival; resembles KLASS function-preserving cases.",
        "Good Prognosis": "70–85% survival with favorable nodal profile.",
        "Moderate Prognosis": "50–70% survival; careful surveillance recommended.",
        "Poor Prognosis": "30–50% survival; consider intensified adjuvant plans.",
        "Very Poor Prognosis": "<30% survival; aligns with aggressive biology.",
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize from JSON configuration matching your existing pattern."""
        self.config = config
        self.model_id = config.get("id", "cox_model")
        self.name = config.get("name", "Cox Survival Model")
        self.variables = config.get("variables", {})
        self.timepoints = config.get("timepoints", [5, 10])

        baseline = config.get("baseline_survival", {})
        self.baseline_5yr = float(baseline.get("5_year_estimate", 0.65))
        self.baseline_10yr = float(baseline.get("10_year_estimate", 0.55))

    def calculate_linear_predictor(self, patient_data: dict[str, Any]) -> float:
        """
        Calculate Cox linear predictor from patient covariates.

        Computes the linear combination of regression coefficients and patient
        variables: LP = β₁X₁ + β₂X₂ + ... + βₖXₖ

        Parameters
        ----------
        patient_data : dict[str, Any]
            Patient variables in Han 2012 format:

            - age : str
                Age category ('< 40', '40-49', '50-59', '60-69', '>= 70')
            - sex : str
                Patient sex ('male' or 'female')
            - location : str
                Tumor location ('upper', 'middle', or 'lower')
            - depth_of_invasion : str
                Depth category ('mucosa', 'submucosa', 'proper_muscle',
                'subserosa', 'serosa', 'adjacent_organ_invasion')
            - metastatic_lymph_nodes : str
                Positive LN category ('0', '1-2', '3-6', '7-15', '>= 16')
            - examined_lymph_nodes : int
                Number of lymph nodes examined

        Returns
        -------
        float
            Linear predictor value. Higher values indicate worse prognosis.

        Notes
        -----
        The linear predictor is used to compute survival probabilities via:

            S(t|X) = S₀(t)^{exp(LP)}

        where S₀(t) is the baseline survival function.

        See Also
        --------
        calculate_survival : Converts linear predictor to survival probabilities.
        Han2012VariableMapper : Maps raw patient data to Han 2012 format.
        """
        lp = 0.0

        for var_name, var_config in self.variables.items():
            if var_name not in patient_data:
                continue

            var_type = var_config.get("type")

            if var_type == "categorical":
                patient_category = patient_data[var_name]
                categories = var_config.get("categories", {})
                if patient_category in categories:
                    lp += float(categories[patient_category])

            elif var_type == "continuous":
                patient_value = float(patient_data[var_name])
                coefficient = float(var_config.get("coefficient", 0.0))
                lp += coefficient * patient_value

        return lp

    def calculate_survival(self, patient_data: dict[str, Any]) -> dict[int, float]:
        """
        Calculate survival probabilities at configured timepoints.

        Applies the Cox proportional hazards model to compute survival
        probabilities using: S(t|X) = S₀(t)^{exp(LP)}

        Parameters
        ----------
        patient_data : dict[str, Any]
            Patient variables in Han 2012 format (see calculate_linear_predictor).

        Returns
        -------
        dict[int, float]
            Survival probabilities at configured timepoints:
            - Key: Years (typically 5 and 10)
            - Value: Probability of survival (0.0 to 1.0)

        Examples
        --------
        >>> model = CoxModel(config)
        >>> patient = {'age': '60-69', 'sex': 'male', 'location': 'lower',
        ...            'depth_of_invasion': 'subserosa', 'metastatic_lymph_nodes': '3-6',
        ...            'examined_lymph_nodes': 30}
        >>> survival = model.calculate_survival(patient)
        >>> survival
        {5: 0.72, 10: 0.61}

        Notes
        -----
        **Baseline Survival:** S₀(5yr) and S₀(10yr) are estimated values
        calibrated to reproduce published cohort statistics. The original
        Han 2012 publication does not provide explicit baseline survival values.

        **Clinical Translation:** Absolute probabilities require institutional
        recalibration before patient-facing applications.

        See Also
        --------
        calculate_linear_predictor : Computes the LP used in this calculation.
        categorize_risk : Converts survival probability to prognosis category.

        References
        ----------
        Han DS, et al. J Clin Oncol. 2012;30(31):3834-40.
        """
        lp = self.calculate_linear_predictor(patient_data)
        survival_5yr = self.baseline_5yr ** math.exp(lp)
        survival_10yr = self.baseline_10yr ** math.exp(lp)

        survival_5yr = max(0.0, min(1.0, survival_5yr))
        survival_10yr = max(0.0, min(1.0, survival_10yr))
        return {5: survival_5yr, 10: survival_10yr}

    def predict_patient_survival(self, patient_data: dict[str, Any]) -> dict[int, float]:
        """Public API to compute survival probabilities at configured time points."""
        return self.calculate_survival(patient_data)

    def categorize_risk(self, survival_5yr: float) -> tuple[str, str]:
        """Return prognosis category plus descriptive text."""
        category = self._survival_category_label(survival_5yr)
        description = self.PROGNOSIS_DESCRIPTIONS.get(
            category, "Survival probability outside configured ranges."
        )
        return category, description

    @staticmethod
    def _survival_category_label(survival_5yr: float) -> str:
        if survival_5yr >= 0.85:
            return "Excellent Prognosis"
        if survival_5yr >= 0.70:
            return "Good Prognosis"
        if survival_5yr >= 0.50:
            return "Moderate Prognosis"
        if survival_5yr >= 0.30:
            return "Poor Prognosis"
        return "Very Poor Prognosis"
