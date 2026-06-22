"""
Interactive Streamlit Demo: Gastric Cancer Risk Calculator

This application provides an educational demonstration of risk stratification
models for gastric cancer. It integrates:
1. A heuristic recurrence risk model (KLASS-inspired structure)
2. The Han 2012 Cox proportional hazards survival nomogram

Author: Maximilian Dressler
License: MIT

⚠️ EDUCATIONAL USE ONLY - NOT FOR CLINICAL DECISION-MAKING ⚠️
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path for imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from risk_calculator import (  # noqa: E402
    COX_MODEL_AVAILABLE,
    GastricCancerRiskModel,
    load_model_config,
    load_survival_model,
)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Gastric Cancer Risk Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS Styling
# =============================================================================

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-banner {
        background-color: #FFF3CD;
        border: 1px solid #FFECB5;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    .warning-banner h4 {
        color: #856404;
        margin: 0 0 0.5rem 0;
    }
    .warning-banner p {
        color: #856404;
        margin: 0;
        font-size: 0.9rem;
    }
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-card h2 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }
    .risk-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .survival-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .survival-card h2 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }
    .survival-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .category-low {
        background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%);
    }
    .category-moderate {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    .category-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .info-box {
        background-color: #E8F4FD;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Header and Warning Banner
# =============================================================================

st.markdown(
    '<h1 class="main-header">🏥 Gastric Cancer Risk Calculator</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">Educational Demonstration of Clinical Risk Stratification Models</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="warning-banner">
    <h4>⚠️ EDUCATIONAL USE ONLY</h4>
    <p>
        This tool is for <strong>demonstration and educational purposes only</strong>.
        It is <strong>NOT validated for clinical decision-making</strong>.
        Model coefficients are pedagogical approximations. Always consult qualified
        healthcare professionals for medical decisions.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Load Models
# =============================================================================


@st.cache_resource
def load_models():
    """Load and cache the risk models."""
    recurrence_config = load_model_config()
    recurrence_model = GastricCancerRiskModel(recurrence_config)
    survival_model = load_survival_model() if COX_MODEL_AVAILABLE else None
    return recurrence_model, survival_model, recurrence_config


recurrence_model, survival_model, model_config = load_models()

# =============================================================================
# Sidebar: Patient Input Parameters
# =============================================================================

st.sidebar.header("📋 Patient Parameters")
st.sidebar.markdown("Enter clinical variables below:")

# TNM Staging Section
st.sidebar.subheader("TNM Staging")

t_stage = st.sidebar.selectbox(
    "T Stage (Tumor Depth)",
    options=["T1", "T2", "T3", "T4"],
    index=1,
    help="T1: Mucosa/Submucosa | T2: Muscularis propria | T3: Subserosa | T4: Serosa/Adjacent organs",
)

n_stage = st.sidebar.selectbox(
    "N Stage (Nodal Status)",
    options=["N0", "N1", "N2", "N3"],
    index=0,
    help="N0: No regional LN metastasis | N1: 1-2 nodes | N2: 3-6 nodes | N3: 7+ nodes",
)

# Clinical Variables Section
st.sidebar.subheader("Clinical Variables")

age = st.sidebar.slider(
    "Age (years)", min_value=18, max_value=95, value=60, step=1, help="Patient age at diagnosis"
)

tumor_size = st.sidebar.slider(
    "Tumor Size (cm)",
    min_value=0.5,
    max_value=20.0,
    value=4.0,
    step=0.5,
    help="Maximum tumor diameter in centimeters",
)

# Advanced Parameters (Expandable)
with st.sidebar.expander("🔬 Advanced Parameters"):
    st.markdown("##### Lymph Node Assessment")

    positive_ln = st.number_input(
        "Positive Lymph Nodes",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Number of metastatic lymph nodes",
    )

    total_ln = st.number_input(
        "Total Lymph Nodes Examined",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
        help="Total number of lymph nodes examined (minimum 15 recommended per guidelines)",
    )

    if total_ln < 15:
        st.warning("⚠️ Guideline recommends ≥15 lymph nodes for adequate staging")

    ln_ratio = positive_ln / total_ln if total_ln > 0 else 0.0
    st.metric("Lymph Node Ratio", f"{ln_ratio:.2%}")

    if COX_MODEL_AVAILABLE:
        st.markdown("##### Additional Han 2012 Variables")
        sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
        tumor_location = st.selectbox(
            "Tumor Location",
            options=["Upper", "Middle", "Lower"],
            index=2,
            help="Anatomical location in stomach",
        )
    else:
        sex = "Male"
        tumor_location = "Lower"

# =============================================================================
# Calculate Risk Scores
# =============================================================================

# Prepare patient data for recurrence model
patient_data = {
    "T_stage": t_stage,
    "N_stage": n_stage,
    "age": age,
    "tumor_size_cm": tumor_size,
    "ln_ratio": ln_ratio,
    "positive_LN": positive_ln,
    "total_LN": total_ln,
}

# Calculate recurrence risk
recurrence_risk = recurrence_model.calculate_risk(patient_data)

# Determine risk category using the model's canonical thresholds
risk_category = GastricCancerRiskModel.risk_category(recurrence_risk)

CATEGORY_CSS = {
    "Low Risk": ("category-low", "#38ef7d"),
    "Moderate Risk": ("category-moderate", "#F2994A"),
    "High Risk": ("category-high", "#eb3349"),
    "Very High Risk": ("category-high", "#7f0000"),
}
category_class, category_color = CATEGORY_CSS.get(risk_category, ("category-moderate", "#F2994A"))

# Calculate survival predictions if model available
survival_5yr = None
survival_10yr = None
if survival_model and COX_MODEL_AVAILABLE:
    from models.variable_mapper_tcga import Han2012VariableMapper

    han_patient = {
        "age": age,
        "Sex": sex,
        "T_stage": t_stage,
        "N_stage": n_stage,
        "positive_LN": positive_ln,
        "total_LN": total_ln,
    }

    try:
        mapper = Han2012VariableMapper()
        mapped_vars = mapper.map_patient_from_dict(han_patient)
        survival_probs = survival_model.calculate_survival(mapped_vars)
        survival_5yr = survival_probs.get(5)
        survival_10yr = survival_probs.get(10)
    except Exception as e:
        st.sidebar.error(f"Survival model error: {e}")

# =============================================================================
# Main Content: Results Display
# =============================================================================

# Results Section
st.header("📊 Risk Assessment Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
    <div class="risk-card {category_class}">
        <h2>{recurrence_risk:.1%}</h2>
        <p>5-Year Recurrence Risk</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="risk-card {category_class}">
        <h2>{risk_category}</h2>
        <p>Risk Category</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    if survival_5yr is not None:
        st.markdown(
            f"""
        <div class="survival-card">
            <h2>{survival_5yr:.1%}</h2>
            <p>5-Year Overall Survival</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="survival-card" style="opacity: 0.5;">
            <h2>N/A</h2>
            <p>Survival Model Unavailable</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# Detailed Breakdown
# =============================================================================

st.header("📈 Detailed Analysis")

tab1, tab2, tab3 = st.tabs(["🧮 Risk Breakdown", "📉 Sensitivity Analysis", "📚 Model Information"])

with tab1:
    st.subheader("Risk Factor Contributions")

    # Calculate individual contributions
    contributions = {
        "T Stage": float(model_config.get("t_stage_weights", {}).get(t_stage, 0)),
        "N Stage": float(model_config.get("n_stage_weights", {}).get(n_stage, 0)),
        "Age Effect": max(0, age - model_config.get("age_weight", {}).get("pivot", 50))
        * model_config.get("age_weight", {}).get("weight", 0),
        "Tumor Size": tumor_size * model_config.get("tumor_size_weight", {}).get("weight", 0),
        "LN Ratio": ln_ratio * model_config.get("ln_ratio_weight", 0),
    }

    # Create contribution dataframe
    contrib_df = pd.DataFrame(
        {
            "Factor": list(contributions.keys()),
            "Coefficient Contribution": list(contributions.values()),
        }
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [
            "#667eea" if v >= 0 else "#eb3349" for v in contrib_df["Coefficient Contribution"]
        ]
        bars = ax.barh(contrib_df["Factor"], contrib_df["Coefficient Contribution"], color=colors)
        ax.set_xlabel("Log-odds Contribution")
        ax.set_title("Risk Factor Contributions (Logit Scale)")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Interpretation Guide:**")
        st.markdown("""
        - Positive values **increase** recurrence risk
        - Negative values **decrease** recurrence risk
        - Larger absolute values = stronger effect
        """)

        st.markdown("**Current Patient Summary:**")
        st.markdown(f"- **Stage:** {t_stage}{n_stage}")
        st.markdown(f"- **Age:** {age} years")
        st.markdown(f"- **Tumor:** {tumor_size} cm")
        st.markdown(f"- **LN Ratio:** {ln_ratio:.2%}")

with tab2:
    st.subheader("Lymph Node Yield Sensitivity Analysis")
    st.markdown("""
    This analysis shows how the estimated risk changes with different numbers of
    lymph nodes examined. More thorough lymphadenectomy (higher LN yield) typically
    improves staging accuracy.
    """)

    # Sensitivity analysis
    ln_yields = list(range(5, 51, 5))
    risks_at_yields = []

    for yield_val in ln_yields:
        test_patient = patient_data.copy()
        test_patient["total_LN"] = yield_val
        test_patient["ln_ratio"] = positive_ln / yield_val if yield_val > 0 else 0
        risks_at_yields.append(recurrence_model.calculate_risk(test_patient))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        ln_yields,
        [r * 100 for r in risks_at_yields],
        "o-",
        linewidth=2,
        markersize=8,
        color="#667eea",
    )
    ax.axhline(
        y=recurrence_risk * 100,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Current: {total_ln} nodes",
    )
    ax.axvline(x=15, color="green", linestyle=":", alpha=0.7, label="Guideline minimum (15)")
    ax.fill_between(ln_yields, [r * 100 for r in risks_at_yields], alpha=0.2, color="#667eea")
    ax.set_xlabel("Total Lymph Nodes Examined", fontsize=12)
    ax.set_ylabel("Estimated 5-Year Recurrence Risk (%)", fontsize=12)
    ax.set_title("Risk Estimate vs. Lymph Node Yield", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("""
    💡 **Clinical Insight:** The observed decrease in risk with higher LN yield reflects
    the "Will Rogers phenomenon" — more thorough staging may upstage patients, improving
    prognostic accuracy. This is a staging effect, not a direct therapeutic benefit.
    """)

with tab3:
    st.subheader("Model Documentation")

    model_tab1, model_tab2 = st.tabs(["Recurrence Model", "Survival Model"])

    with model_tab1:
        st.markdown(f"""
        ### {model_config.get('name', 'Unknown Model')}

        **Model ID:** `{model_config.get('id', 'N/A')}`

        **Description:**
        {model_config.get('description', 'No description available.')}

        **Citation:**
        {model_config.get('citation', 'No citation available.')}

        ---

        #### Mathematical Formulation

        The model uses a logistic regression form:

        $$
        P(\\text{{recurrence}}) = \\sigma\\left(\\beta_0 + \\beta_T \\cdot T + \\beta_N \\cdot N + \\beta_{{age}} \\cdot (age - 50)^+ + \\beta_{{size}} \\cdot size + \\beta_{{ratio}} \\cdot LN_{{ratio}}\\right)
        $$

        Where $\\sigma(x) = \\frac{{1}}{{1 + e^{{-x}}}}$ is the logistic function.

        #### Coefficient Table
        """)

        # Coefficient table
        coef_data = []
        for stage, weight in model_config.get("t_stage_weights", {}).items():
            coef_data.append({"Variable": f"T Stage: {stage}", "Coefficient": weight})
        for stage, weight in model_config.get("n_stage_weights", {}).items():
            coef_data.append({"Variable": f"N Stage: {stage}", "Coefficient": weight})
        coef_data.append(
            {
                "Variable": "Age (per year >50)",
                "Coefficient": model_config.get("age_weight", {}).get("weight", 0),
            }
        )
        coef_data.append(
            {
                "Variable": "Tumor Size (per cm)",
                "Coefficient": model_config.get("tumor_size_weight", {}).get("weight", 0),
            }
        )
        coef_data.append(
            {"Variable": "LN Ratio", "Coefficient": model_config.get("ln_ratio_weight", 0)}
        )
        coef_data.append({"Variable": "Intercept", "Coefficient": model_config.get("intercept", 0)})

        st.dataframe(pd.DataFrame(coef_data), use_container_width=True)

    with model_tab2:
        if survival_model:
            st.markdown("""
            ### Han 2012 D2 Gastrectomy Survival Nomogram

            **Citation:**
            Han DS, et al. Nomogram predicting long-term survival after D2 gastrectomy
            for gastric cancer. *J Clin Oncol.* 2012;30(31):3834-40.

            **Model Type:** Cox Proportional Hazards

            **Outcome:** Overall Survival (5-year and 10-year)

            **Development Cohort:** 7,954 patients from Seoul National University Hospital

            **C-Index:** 0.78 (internal validation)

            ---

            #### ⚠️ Important Limitations

            1. **Baseline Survival S₀(t)** was not published in the original paper and has
               been estimated through calibration. These values require institutional validation.

            2. **Population Differences:** The model was developed in a Korean cohort and may
               not generalize directly to other populations.

            3. **Temporal Drift:** Treatment advances since 2012 may affect model accuracy.

            #### Predictor Variables
            - Age category (5 levels)
            - Sex (male/female)
            - Tumor location (upper/middle/lower)
            - Depth of invasion (6 levels, corresponds to T stage)
            - Metastatic lymph nodes (5 categories)
            - Examined lymph nodes (continuous, protective)
            """)

            if survival_10yr is not None:
                st.metric("10-Year Overall Survival", f"{survival_10yr:.1%}")
        else:
            st.warning("""
            ⚠️ **Survival Model Not Available**

            The Han 2012 Cox survival model components could not be loaded. This may be due to:
            - Missing model configuration file
            - Import errors in the Cox model module

            Only the recurrence risk model is currently active.
            """)

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📖 Documentation**
    [View on GitHub](https://github.com/Herbert-Research/the-gastric-cancer-risk-calculator)
    """)

with col2:
    st.markdown("""
    **📝 Citation**
    Dressler M.H. (2025). Gastric Cancer Risk Calculator.
    Educational demonstration.
    """)

with col3:
    st.markdown("""
    **⚖️ License**
    MIT License - Educational Use Only
    """)

st.caption("""
---
**Disclaimer:** This tool is provided for educational and research purposes only.
It is NOT intended for clinical use or to guide patient care decisions.
The authors assume no liability for any use of this tool.
Always consult qualified healthcare professionals for medical decisions.
""")
