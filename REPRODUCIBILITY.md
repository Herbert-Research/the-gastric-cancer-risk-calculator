# Reproducing the Validation Results

This document provides step-by-step instructions to reproduce all results, figures, and metrics presented in this repository. Following these instructions ensures **complete scientific reproducibility** of the gastric cancer risk calculator validation study.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Integrity Verification](#data-integrity-verification)
3. [Reproduce All Figures](#reproduce-all-figures)
4. [Expected Output Checksums](#expected-output-checksums)
5. [Test Suite Verification](#test-suite-verification)
6. [Troubleshooting](#troubleshooting)

---

## Environment Setup

Choose **one** of the following methods to set up an identical computational environment:

### Option 1: pip (Recommended for Quick Start)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies with exact versions
pip install -r requirements.txt
```

### Option 2: Conda (Recommended for Data Science Workflows)

```bash
# Create environment from specification
conda env create -f environment.yml

# Activate environment
conda activate gastric-cancer-risk
```

### Option 3: Docker (Recommended for Complete Isolation)

```bash
# Build the Docker image
docker build -t gastric-risk .

# Run the container with output volume mounted
docker run -v $(pwd)/outputs:/app/outputs gastric-risk

# On Windows PowerShell:
docker run -v ${PWD}/outputs:/app/outputs gastric-risk
```

### Python Version Requirements

- **Minimum**: Python 3.9
- **Tested on**: Python 3.9, 3.10, 3.11
- **CI Matrix**: See `.github/workflows/ci.yml` for the full test matrix

---

## Data Integrity Verification

Before running any analysis, verify that the input data file has not been corrupted or modified:

### SHA256 Checksum Verification

The SHA256 checksum below is for the TCGA dataset downloaded independently from [cBioPortal](https://www.cbioportal.org/).
This file is not distributed with this repository.

```bash
# On Linux/macOS:
sha256sum data/tcga_2018_clinical_data.tsv

# On Windows PowerShell:
Get-FileHash data\tcga_2018_clinical_data.tsv -Algorithm SHA256 | Select-Object -ExpandProperty Hash
```

**Expected Hash:**
```
DBB2106F7DD4A2642109B1CECDA25D2203CBD3BB5F32DBAA548BEC1469B7127A
```

If the hash does not match, please re-download the data from the original source (see `data/README.md`).

### Data File Specifications

| Property | Value |
|----------|-------|
| File | `data/tcga_2018_clinical_data.tsv` |
| Format | Tab-separated values (TSV) |
| Encoding | UTF-8 |
| Rows | 443 (including header) |
| Columns | 60 clinical variables |
| Source | TCGA STAD 2018 |

---

## Reproduce All Figures

### Full Pipeline Execution

Run the complete analysis pipeline to generate all figures and metrics:

```bash
python risk_calculator.py \
    --data data/tcga_2018_clinical_data.tsv \
    --output-dir outputs/ \
    --verbose
```

### Expected Output Files

After successful execution, the following files should be created in the `outputs/` directory:

| File | Description | Size (approx.) |
|------|-------------|----------------|
| `risk_predictions.png` | Individual patient risk scores with staging distribution | ~150 KB |
| `tcga_cohort_summary.png` | Stage distribution heatmap and cohort demographics | ~120 KB |
| `sensitivity_analysis.png` | Lymph node yield sensitivity analysis | ~80 KB |
| `calibration_curve.png` | Predicted vs. observed calibration with 95% CI | ~90 KB |
| `survival_predictions_han2012.png` | Han 2012 survival model predictions | ~100 KB |
| `survival_vs_recurrence_comparison.png` | Model comparison scatter plot | ~85 KB |

### Individual Figure Generation

To generate specific figures independently:

```bash
# Risk predictions only (skip survival analysis)
python risk_calculator.py --data data/tcga_2018_clinical_data.tsv --output-dir outputs/ --skip-survival

# Survival analysis only
python -c "from risk_calculator import main; main()" --data data/tcga_2018_clinical_data.tsv
```

---

## Expected Output Checksums

Due to platform-specific font rendering and floating-point variations, PNG file checksums may differ slightly across systems. However, the following metrics should be **exactly reproducible**:

### Key Metrics (Exact Values)

| Metric | Expected Value | 95% CI |
|--------|----------------|--------|
| Brier Score (recurrence vs. DFS) | 0.502 | [0.461, 0.538] |
| Pearson Correlation (survival vs. recurrence) | -0.458 | [-0.512, -0.390] |
| Han 2012 C-index (vs. DFS proxy) | 0.488 | — |
| Cohort Size (after QC) | 436 patients | — |

Correlation computed using dataset hash `DBB2106F7DD4A2642109B1CECDA25D2203CBD3BB5F32DBAA548BEC1469B7127A` with seed 42.

### Random Seed Configuration

All stochastic operations use fixed seeds for reproducibility:

- **Imputation RNG**: `np.random.default_rng(seed=42)`
- **Bootstrap CI**: `random_state=42` (default)

---

## Test Suite Verification

### Run Full Test Suite

```bash
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

Note: All test dependencies (including pytest) are included in requirements.txt.

### CI Test Coverage Note

In CI environments where the full TCGA dataset is not available, integration tests
use a synthetic fixture (`tests/fixtures/dummy_data.tsv`) containing 5 representative
patient records. This allows CI to validate:

- Data ingestion and parsing logic
- Feature engineering pipeline
- Risk score computation
- Output formatting

**For full validation metrics**, run tests locally with the complete TCGA dataset.

**Expected Result:** `190 passed` (may vary slightly with test additions)

### Run with Coverage Report

```bash
pytest tests/ -v --cov --cov-report=term-missing --cov-report=html
```

**Expected Coverage:**
- Overall: ≥70%
- `risk_calculator.py`: ≥53% (see Phase 1 improvements for target of 85%)
- `models/cox_model.py`: ≥85%
- `utils/visualization.py`: ≥80%

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/test_cox_model.py tests/test_risk_model.py -v

# Integration tests only
pytest tests/test_risk_calculator_integration.py -v

# Property-based tests (requires hypothesis)
pytest tests/test_property_based.py -v
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'risk_calculator'

**Solution:** Ensure you are running from the repository root directory:
```bash
cd /path/to/the-gastric-cancer-risk-calculator
python risk_calculator.py --help
```

#### 2. Matplotlib Backend Error on Headless Systems

**Solution:** Set the backend before importing matplotlib:
```bash
export MPLBACKEND=Agg  # Linux/macOS
$env:MPLBACKEND = "Agg"  # Windows PowerShell
```

Or add to your script:
```python
import matplotlib
matplotlib.use('Agg')
```

#### 3. Font Warnings on Windows

**Expected behavior:** You may see warnings about missing fonts. These do not affect numerical results. To suppress:
```bash
$env:MPLCONFIGDIR = "$env:TEMP\matplotlib"
```

#### 4. Cox Model Not Available

**Symptom:** "COX_MODEL_AVAILABLE = False" in logs

**Solution:** This indicates the Cox model JSON file is missing or malformed. Verify:
```bash
# Check file exists
ls models/han2012_jco.json

# Validate JSON syntax
python -c "import json; json.load(open('models/han2012_jco.json'))"
```

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/Herbert-Research/the-gastric-cancer-risk-calculator/issues) page
2. Review the CI logs in `.github/workflows/ci.yml`
3. Open a new issue with:
   - Your Python version (`python --version`)
   - Your operating system
   - Complete error traceback
   - Steps to reproduce

---

## Citation

If you use this software for reproducible research, please cite:

```bibtex
@software{gastric_cancer_risk_calculator,
  author = {Dressler, Maximilian Herbert},
  title = {Gastric Cancer Risk Stratification Framework},
  version = {0.1.1},
  year = {2025},
  url = {https://github.com/Herbert-Research/the-gastric-cancer-risk-calculator}
}
```

See `CITATION.cff` for additional citation formats.
