# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `GastricCancerRiskModel.calculate_risk` now raises `ValueError` on non-finite
  `tumor_size_cm` or lymph node ratio instead of silently emitting a NaN risk
  score (clinical-safety: fail loudly rather than propagate NaN downstream).
- Streamlit demo (`app.py`): replaced invalid `width="stretch"` argument that
  raised `TypeError` under the pinned Streamlit version; surfaced by the new
  `AppTest` smoke test.
- Corrected the CI status badge URL (pointed at an old repository name) and the
  coverage badge (now reflects measured coverage).
- Reconciled the TRIPOD "Results" row in `VALIDATION_REPORT.md`, which claimed
  no bootstrapping despite the pipeline reporting bootstrap 95% CIs.

### Added
- `streamlit` / `watchdog` pinned in `requirements.txt` and `requirements.lock`
  so the interactive demo installs from the documented requirements.
- Smoke tests for `app.py` (`tests/test_app_smoke.py`): always compiles, and
  executes end-to-end under Streamlit's `AppTest` when Streamlit is installed.
- Regression tests pinning the README example outputs
  (`tests/test_readme_examples.py`) so docs cannot silently drift from the code.
- Documented O(n²) complexity of `concordance_index`; documented the Streamlit
  demo in the README usage section.

### Changed
- CI now also triggers on development branches.

## [0.1.1] - 2025-11-29

### Fixed
- Removed unused `map_patient_to_han2012` function from `cox_model.py`
- Fixed all PEP8/ruff linting violations (whitespace, import order)
- Updated ruff configuration to use non-deprecated `[tool.ruff.lint]` section
- Completed type annotations for all public functions

### Added
- Pre-commit hooks for automated code quality enforcement
- GitHub Actions CI/CD pipeline with multi-Python-version testing
- Test coverage reporting with 70% minimum threshold
- SECURITY.md for responsible disclosure policy

### Changed
- Repository renamed to fix "calculater" typo

## [0.1.0] - 2025-11-26

### Added
- Han 2012 Cox proportional hazards survival model implementation
  - 5-year and 10-year overall survival predictions
  - Baseline survival calibrated to published cohort statistics
- KLASS-inspired heuristic recurrence model (educational demonstration)
- TCGA STAD cohort validation pipeline
  - Variable harmonization via `Han2012VariableMapper`
  - Stage-informed imputation for missing surgical variables
  - Brier score calibration analysis
- Sensitivity analysis for lymph node yield impact
- Publication-quality visualization suite
  - Calibration curves
  - Survival distribution histograms
  - Risk stratification heatmaps
- Docker containerization for reproducible execution
- Comprehensive test suite (13 tests, 100% pass rate)

### Documentation
- TRIPOD-compliant validation report
- Clinical translation roadmap (Phases 1-4)
- Pre-deployment checklist for institutional use
- Full mathematical specification of model coefficients

### Known Limitations
- Baseline survival S₀(t) estimated, not from original publication
- TCGA validation uses 100% imputed surgical variables
- Brier score reflects endpoint mismatch (recurrence vs. DFS), not model failure
