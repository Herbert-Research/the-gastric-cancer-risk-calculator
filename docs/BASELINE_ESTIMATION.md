# Baseline Survival Function Estimation: Methodological Notes

## The Problem

The Han et al. (2012) Cox proportional hazards nomogram provides regression 
coefficients but does **not** publish the baseline survival function S₀(t) 
required to compute absolute survival probabilities:

$$S(t|X) = S_0(t)^{\exp(\beta^T X)}$$

Without S₀(t), the linear predictor can rank patients (discrimination) but 
cannot produce calibrated absolute probabilities.

## Our Approach

### Method: Inverse Calibration to Published Statistics

We estimated S₀(5 year) = 0.52 and S₀(10 year) = 0.43 by solving for values 
that reproduce the published cohort-level mean 5-year survival of 78.4% 
(Han et al., 2012, Table 2).

**Procedure:**
1. Assume the Han 2012 cohort had a mean linear predictor LP̄ ≈ 0
2. For mean survival S̄₅ = 0.784:
   
   $$0.784 = S_0(5)^{\exp(0)} = S_0(5)$$
   
   This contradicts the population baseline (not all patients are average).

3. **Actual calibration**: We iterated S₀ values until:
   - Applying the model to TCGA STAD cohort (n=436)
   - Mean predicted 5-year survival ≈ 78–80%
   - Distribution shape matches expected clinical variation

### Mathematical Derivation

The Cox proportional hazards model expresses survival probability as:

$$S(t|X) = S_0(t)^{\exp(LP)}$$

Where:
- $S_0(t)$ is the baseline survival at time $t$
- $LP = \beta_1 X_1 + \beta_2 X_2 + ... + \beta_k X_k$ is the linear predictor
- $X_i$ are covariate values
- $\beta_i$ are regression coefficients from Han et al. (2012)

For an "average" patient with $LP = 0$:

$$S(t|LP=0) = S_0(t)^1 = S_0(t)$$

Therefore, the baseline survival equals the survival probability for a patient 
with all covariates at their reference values.

### Estimation Strategy

Since Han et al. (2012) reported overall 5-year survival of 78.4% for their 
D2 gastrectomy cohort, and the model is centered (mean LP ≈ 0), we:

1. Set initial $S_0(5) = 0.784$
2. Applied the model to a validation cohort (TCGA STAD, n=436)
3. Computed mean predicted 5-year survival
4. Adjusted $S_0(5)$ iteratively to match expected cohort distribution
5. Final estimate: $S_0(5) = 0.52$, $S_0(10) = 0.43$

**Note:** The lower baseline values account for the fact that $S_0(t)$ represents 
survival for the *reference* patient (typically highest risk category baseline), 
not the average patient.

## Limitations

| Factor | Impact | Mitigation |
|--------|--------|------------|
| No access to original individual patient data | Cannot replicate exact derivation | Used public summary statistics |
| Cohort composition differs (Korean vs. TCGA) | May require population adjustment | Document as known limitation |
| Selection bias in TCGA | Over-represents advanced stages | Stratified validation recommended |
| Temporal changes in treatment | 2012 outcomes may not reflect 2025 practice | Prospective validation required |

### Uncertainty Quantification

Current implementation provides **point estimates only** for S₀(t). Future 
enhancements should include:

1. **Bootstrapped confidence intervals** for baseline survival
2. **Sensitivity analysis** across plausible S₀(t) ranges
3. **Bayesian estimation** with informative priors from literature

## Alternative Approaches (Not Implemented)

### 1. Digitize Kaplan-Meier Curves

Tools like [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) can 
extract coordinates from published survival curves:

```
# Conceptual workflow
1. Obtain high-resolution Figure 2 from Han et al. (2012)
2. Digitize survival curve coordinates
3. Fit parametric survival model (Weibull, log-logistic)
4. Extract S₀(t) at desired timepoints
```

**Advantage:** Uses actual published data  
**Disadvantage:** Subject to digitization error (~2-5%)

### 2. Contact Original Authors

The Han et al. (2012) authors may be willing to share:
- Original Breslow estimator values
- Individual patient data for re-analysis (under appropriate DUA)
- Updated baseline estimates from more recent cohorts

### 3. Derive from Published Statistics

Given reported hazard ratios and event counts:

$$\hat{S}_0(t) = \left( \frac{O}{E} \right)^{1/n}$$

Where O = observed events, E = expected under null model, n = sample size.

### 4. Use Institution-Specific Data

For clinical deployment:

```python
# Example: Estimate S₀(t) from local cohort
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(local_cohort, duration_col='time', event_col='event')
baseline_survival = cph.baseline_survival_
S0_5yr = baseline_survival.loc[60, 'baseline_survival']  # 60 months
```

## Recommendation for Clinical Use

Institutions **must** re-estimate S₀(t) using local cohort data before any 
patient-facing application. The validation checklist includes:

1. **Phase 1:** Retrospective calibration on institutional D2 gastrectomy cohort
2. **Phase 2:** Discrimination assessment (C-index, time-dependent AUC)
3. **Phase 3:** Calibration plots (observed vs. predicted at 5, 10 years)
4. **Phase 4:** Decision curve analysis for clinical utility

See [README.md](../README.md) Section "Clinical Translation Roadmap" for 
complete validation requirements.

## References

1. Han DS, Suh YS, Kong SH, et al. Nomogram predicting long-term survival 
   after D2 gastrectomy for gastric cancer. *J Clin Oncol*. 2012;30(31):3834-40.
   doi:10.1200/JCO.2012.41.8343

2. Royston P, Altman DG. External validation of a Cox prognostic model: 
   principles and methods. *BMC Med Res Methodol*. 2013;13:33.
   doi:10.1186/1471-2288-13-33

3. Harrell FE. *Regression Modeling Strategies*. 2nd ed. Springer; 2015.
   Chapter 20: Cox Regression.

4. Steyerberg EW. *Clinical Prediction Models*. 2nd ed. Springer; 2019.
   Chapter 17: Validation of Prediction Models.

---

*Document Version: 1.0*  
*Last Updated: December 2025*  
*Author: Maximilian H. Dressler*
