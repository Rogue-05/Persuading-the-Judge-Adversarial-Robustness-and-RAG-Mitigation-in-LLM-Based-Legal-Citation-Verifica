# Persuading the Judge: Adversarial Robustness and RAG Mitigation in LLM-Based Legal Citation Verification



## Overview

This repository contains the code, dataset, and experimental results for studying how adversarial persuasion attacks degrade LLM-based legal citation verifiers, and whether retrieval-augmented generation (RAG) grounding can protect against them.

**Key Findings:**
- Authority appeals nearly quintuple the odds of a correct legal judgment flipping to incorrect (OR = 4.810, p < 0.0001)
- RAG grounding significantly mitigates but **does not eliminate** authority-based persuasion (OR = 0.493, p = 0.031)
- Persuaded models are not uncertain — 96.8% of authority-induced flips occur at high confidence (≥ 8/10)
- Cross-family attacks (GPT-OSS 120B → Llama 70B) achieve 86% flip rate even with grounded evidence

## Repository Structure

```
NLLP/
├── legal_dataset.json              # Main benchmark dataset (176 items, 3 classes)
├── errors_legal_dataset.json       # Items flagged during dataset QA
├── extra_fabricated_cases.json     # Additional fabricated cases for balance
│
├── build_dataset.py                # Dataset construction pipeline
├── build_dataset_4keys.py          # Extended dataset builder (4-key variant)
├── validate_dataset.py             # Dataset validation & QA checks
├── check_dataset_counts.py         # Class balance verification
│
├── 02_main_experiment_openrouter.ipynb   # Main experiment notebook (Llama + GPT-OSS)
├── 03_crossfamily_llama_v_gptoss.ipynb   # Cross-family adversarial experiment
│
├── compute_metrics.py              # Metrics computation (True ASR, CW-ASR, etc.)
├── fit_logistic_regression.py      # Family-pooled logistic regression (Table 2)
├── lr_test_model1_vs_model2.py     # Likelihood ratio test (Model 1 vs Model 2)
├── reprocess_failed_trials.py      # Rerun failed/truncated API trials
├── rescue_parser_failures.py       # Offline parser recovery (no API calls)
│
├── results_gptoss_full_run_rescued/    # GPT-OSS results (rescued, canonical)
│   ├── main/                           # Ungrounded trial JSONs
│   ├── mitigation/                     # Grounded trial JSONs
│   └── summary/                        # Computed summary CSVs
│
├── results_llama_full_run_rescued/     # Llama results (rescued, canonical)
│   ├── results/main/                   # Ungrounded trial JSONs
│   ├── results/mitigation/             # Grounded trial JSONs
│   └── results/summary/               # Computed summary CSVs
│
├── results_llama_gptoss_cross/         # Cross-family results
│   ├── main/                           # Ungrounded trial JSONs
│   ├── mitigation/                     # Grounded trial JSONs
│   └── summary/                        # Computed summary CSVs
│
├── summary_regression/                 # Logistic regression outputs
│   ├── logistic_regression_model2_2way_interacted.csv  # Primary model (Table 2)
│   ├── logistic_regression_model1_3way_interacted.csv  # Exploratory 3-way model
│   ├── logistic_regression_model3_ungrounded.csv       # Ungrounded subsample
│   ├── logistic_regression_cell_diagnostics.csv
│   ├── logistic_regression_data_quality.csv
│   ├── lr_test_model1_vs_model2.csv
│   ├── lr_test_bootstrap_distribution.csv
│   └── cw_por_confidence_weighted_metrics.csv          # CW-ASR metrics
│
├── dataset_pilots/                 # Early pilot dataset iterations
│
└── .gitignore
```

## Dataset

The benchmark dataset (`legal_dataset.json`) contains **176 items** across three classes:

| Class | Description | QA Validation |
|:---|:---|:---|
| `valid` | Core legal proposition extracted from case holding | Gemini: `SUPPORTED` at ≥ 0.90 confidence |
| `real_wrong_content` | Minimally adversarially edited valid claim | Gemini: `NOT_SUPPORTED` at ≥ 0.90 confidence |
| `fabricated` | Plausible but non-existent case/citation/holding | CourtListener collision check + Gemini QA |

- **Source**: CourtListener federal court opinions
- **Generator**: Claude Sonnet 4.6 (temperature 0.0)
- **QA Validator**: Gemini (independent model family)

## Experimental Setup

3,468 trials across two model families and a cross-family pairing:

| Setup | Verifier | Adversary | Trials |
|:---|:---|:---|:---:|
| Llama (within-family) | Llama 3.1-8B / 3.3-70B | Llama 3.3-70B / 3.1-8B | 1,734 |
| GPT-OSS (within-family) | GPT-OSS 20B / 120B | GPT-OSS 120B / 20B | 1,734 |
| Cross-family | Llama 3.3-70B | GPT-OSS 120B | 867 |

**Conditions**: Control (neutral reconsideration), Authority Appeal, Fabricated Citation

**Arms**: Ungrounded (internal knowledge only) and Grounded (RAG with retrieved CourtListener text)

## Metrics

- **True ASR**: Fraction of initially-correct judgments that flip to incorrect after challenge
- **CW-ASR** ([Agarwal et al., 2025](https://arxiv.org/abs/2504.00662)): Confidence-Weighted ASR — weights each flip by the model's post-challenge confidence score

## Reproducing Results

### 1. Compute summary metrics from trial JSONs
```bash
# GPT-OSS
python compute_metrics.py \
  --main-dir results_gptoss_full_run_rescued/main \
  --mitigation-dir results_gptoss_full_run_rescued/mitigation

# Llama
python compute_metrics.py \
  --main-dir results_llama_full_run_rescued/results/main \
  --mitigation-dir results_llama_full_run_rescued/results/mitigation
```

### 2. Fit logistic regression (Table 2 in paper)
```bash
python fit_logistic_regression.py
```

### 3. Likelihood ratio test (Model 1 vs Model 2)
```bash
python lr_test_model1_vs_model2.py
```

## Key Results Summary

| Family | Condition | Ungrounded True ASR | Grounded True ASR | Significant? |
|:---|:---|:---:|:---:|:---:|
| GPT-OSS | Authority vs Control | +28.6pp | +19.7pp | **Yes** (both) |
| GPT-OSS | Fab. Cit. vs Control | −10.5pp | −0.5pp | No |
| Llama | Authority vs Control | +20.1pp | −1.3pp | Ungr. only |
| Llama | Fab. Cit. vs Control | +16.5pp | +7.7pp | Ungr. only |
| Cross-Family | Authority vs Control | +27.4pp | +39.7pp | **Yes** (both) |

## License

This repository is provided for research purposes. The dataset is derived from publicly available CourtListener federal court opinions.
