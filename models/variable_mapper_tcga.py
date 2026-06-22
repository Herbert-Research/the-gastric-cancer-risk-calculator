"""
Variable Mapping for Han 2012 Cox Model - TCGA STAD Edition
Maps TCGA STAD data to Han 2012 nomogram format
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Reproducible imputation RNG (seeded for deterministic sampling)
_RNG = np.random.default_rng(seed=42)


class Han2012VariableMapper:
    """Maps TCGA STAD clinical data to Han 2012 nomogram format."""

    # T stage to depth of invasion mapping
    T_STAGE_MAPPING = {
        "T1": "submucosa",
        "T1a": "mucosa",
        "T1b": "submucosa",
        "T2": "proper_muscle",
        "T3": "subserosa",
        "T4": "serosa",
        "T4a": "serosa",
        "T4b": "adjacent_organ_invasion",
    }

    # Estimated examined lymph nodes by N stage (based on Han 2012 cohort)
    N_STAGE_TO_EXAMINED_LN = {
        "N0": 25,
        "N1": 28,
        "N2": 32,
        "N3": 35,
    }

    @staticmethod
    def map_age_category(age: float | None) -> str:
        """Map continuous age to Han 2012 categories."""
        if pd.isna(age) or age is None:
            return "50-59"

        if age < 40:
            return "< 40"
        elif age < 50:
            return "40-49"
        elif age < 60:
            return "50-59"
        elif age < 70:
            return "60-69"
        else:
            return ">= 70"

    @staticmethod
    def map_sex(sex: str | None) -> str:
        """Map sex to Han 2012 format."""
        if pd.isna(sex) or sex is None:
            return "male"

        sex_lower = str(sex).lower().strip()
        if "female" in sex_lower or sex_lower == "f":
            return "female"
        return "male"

    @staticmethod
    def map_tumor_location(icd_code: str | None = None) -> str:
        """
        Estimate tumor location (upper/middle/lower third) from ICD-10 code.

        If ICD-10 code is available, infer from site:
        - C16.0 (cardia): upper
        - C16.1 (fundus): upper
        - C16.2 (body): middle
        - C16.3 (pyloric antrum): lower
        - C16.4 (pylorus): lower
        - C16.5 (lesser curvature): middle
        - C16.6 (greater curvature): middle
        - C16.8 (overlapping): imputed based on distribution
        - C16.9 (NOS): imputed based on distribution

        If no ICD code, use epidemiological distribution:
        Lower (60%), Middle (25%), Upper (15%).

        Note: Uses a seeded RNG for reproducible imputation.
        """
        if icd_code and not pd.isna(icd_code):
            icd_str = str(icd_code).strip().upper()
            if icd_str in ["C16.0", "C16.1"]:
                return "upper"
            elif icd_str == "C16.2":
                return "middle"
            elif icd_str in ["C16.3", "C16.4"]:
                return "lower"
            elif icd_str in ["C16.5", "C16.6"]:
                return "middle"
        # For C16.8, C16.9, or others, use distribution

        # Default to distribution-based sampling (seeded for reproducibility)
        locations = ["lower", "middle", "upper"]
        weights = [0.60, 0.25, 0.15]  # Epidemiological distribution
        return str(_RNG.choice(locations, p=weights))

    @staticmethod
    def map_depth_of_invasion(t_stage: str | None) -> str:
        """Map T stage to depth of invasion category."""
        if pd.isna(t_stage) or t_stage is None:
            return "proper_muscle"

        t_stage_clean = str(t_stage).strip()
        return Han2012VariableMapper.T_STAGE_MAPPING.get(t_stage_clean, "proper_muscle")

    @staticmethod
    def map_metastatic_lymph_nodes(n_stage: str | None, positive_ln: int | None = None) -> str:
        """
        Map to Han 2012 positive lymph node categories.

        Args:
            n_stage: N stage (N0-N3)
            positive_ln: Actual positive count if available
        """
        if positive_ln is not None and not pd.isna(positive_ln):
            n_pos = int(positive_ln)
            if n_pos == 0:
                return "0"
            elif n_pos <= 2:
                return "1-2"
            elif n_pos <= 6:
                return "3-6"
            elif n_pos <= 15:
                return "7-15"
            else:
                return ">= 16"

        # Estimate from N stage
        if pd.isna(n_stage) or n_stage is None:
            return "1-2"

        n_stage_clean = str(n_stage).strip()

        # Map N stage to category
        if "N0" in n_stage_clean:
            return "0"
        elif "N1" in n_stage_clean:
            return "1-2"
        elif "N2" in n_stage_clean:
            return "3-6"
        elif "N3" in n_stage_clean:
            return "7-15"
        else:
            return "1-2"

    @staticmethod
    def estimate_examined_lymph_nodes(
        n_stage: str | None, examined_nodes: int | None = None, positive_nodes: int | None = None
    ) -> int:
        """
        Estimate number of examined lymph nodes.

        Priority:
        1. Actual count if available
        2. Estimated from N stage (based on Han 2012 cohort mean of 32.4)
        """
        # If we have actual count, use it
        if examined_nodes is not None and not pd.isna(examined_nodes):
            return max(int(examined_nodes), 15)

        # If we have positive nodes, examined must be at least that many + buffer
        if positive_nodes is not None and not pd.isna(positive_nodes):
            return max(int(positive_nodes) + 15, 15)

        # Estimate from N stage using Han 2012 cohort statistics
        if pd.isna(n_stage) or n_stage is None:
            return 30  # Default to cohort median

        n_stage_clean = str(n_stage).strip()
        return Han2012VariableMapper.N_STAGE_TO_EXAMINED_LN.get(n_stage_clean, 30)

    @staticmethod
    def map_patient_from_dict(patient_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Map a patient dictionary to Han 2012 nomogram format.

        Transforms clinical variables from the risk calculator's input format
        to the categorical variables required by the Han 2012 Cox model.

        Parameters
        ----------
        patient_dict : dict[str, Any]
            Patient clinical data with keys:

            - age : float, optional
                Patient age in years (mapped to 5 age categories)
            - Sex or sex : str, optional
                Patient sex ('Male'/'Female', mapped to 'male'/'female')
            - T_stage : str, optional
                Tumor stage (T1-T4, mapped to depth of invasion)
            - N_stage : str, optional
                Nodal stage (N0-N3, used for LN estimation if counts missing)
            - positive_LN : int, optional
                Number of positive lymph nodes (preferred over N_stage)
            - total_LN : int, optional
                Total lymph nodes examined
            - icd_10 : str, optional
                ICD-10 code for tumor location (C16.x series)

        Returns
        -------
        dict[str, Any]
            Han 2012 model variables:

            - age : str - Age category ('< 40', '40-49', '50-59', '60-69', '>= 70')
            - sex : str - Sex ('male' or 'female')
            - location : str - Tumor location ('upper', 'middle', 'lower')
            - depth_of_invasion : str - Invasion depth category
            - metastatic_lymph_nodes : str - Positive LN category
            - examined_lymph_nodes : int - Examined LN count

        Notes
        -----
        **Imputation:** When surgical variables are missing (common in TCGA):

        - Tumor location: Sampled from epidemiological distribution
          (60% lower, 25% middle, 15% upper)
        - Positive LN count: Estimated from N-stage midpoint ranges
        - Examined LN count: Estimated from Han 2012 cohort statistics

        Use ``get_imputation_flags()`` to track which variables were imputed.

        See Also
        --------
        get_imputation_flags : Returns boolean flags for imputed variables.
        CoxModel.calculate_survival : Uses the mapped variables for prediction.

        Examples
        --------
        >>> mapper = Han2012VariableMapper()
        >>> patient = {'age': 65, 'Sex': 'Female', 'T_stage': 'T3', 'N_stage': 'N2'}
        >>> han_vars = mapper.map_patient_from_dict(patient)
        >>> han_vars['age']
        '60-69'
        >>> han_vars['depth_of_invasion']
        'subserosa'
        """
        age = patient_dict.get("age")
        sex = patient_dict.get("sex", patient_dict.get("Sex"))
        t_stage = patient_dict.get("T_stage")
        n_stage = patient_dict.get("N_stage")
        positive_ln = patient_dict.get("positive_LN")
        examined_ln = patient_dict.get("total_LN")
        icd_10 = patient_dict.get("icd_10")

        han_patient = {
            "age": Han2012VariableMapper.map_age_category(age),
            "sex": Han2012VariableMapper.map_sex(sex),
            "location": Han2012VariableMapper.map_tumor_location(icd_10),
            "depth_of_invasion": Han2012VariableMapper.map_depth_of_invasion(t_stage),
            "metastatic_lymph_nodes": Han2012VariableMapper.map_metastatic_lymph_nodes(
                n_stage, positive_ln
            ),
            "examined_lymph_nodes": Han2012VariableMapper.estimate_examined_lymph_nodes(
                n_stage, examined_ln, positive_ln
            ),
        }

        return han_patient

    @staticmethod
    def get_imputation_flags(patient_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Track which variables were imputed vs directly measured.

        Returns:
            Dictionary of boolean flags
        """
        sex = patient_dict.get("sex", patient_dict.get("Sex"))
        positive_ln = patient_dict.get("positive_LN")
        examined_ln = patient_dict.get("total_LN")

        return {
            "age_available": patient_dict.get("age") is not None,
            "sex_available": sex is not None and not pd.isna(sex),
            "location_imputed": True,  # Always imputed for TCGA data
            "positive_ln_imputed": positive_ln is None or pd.isna(positive_ln),
            "examined_ln_imputed": examined_ln is None or pd.isna(examined_ln),
        }


def reset_imputation_seed(seed: int = 42) -> None:
    """
    Reset the imputation RNG for reproducible sampling in tests and pipelines.
    """
    global _RNG
    _RNG = np.random.default_rng(seed=seed)
