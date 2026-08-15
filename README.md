# Persuading the Judge: Adversarial Robustness and RAG Mitigation in LLM-Based Legal Citation Verification

## Overview

This repository contains the benchmark dataset, multi-agent evaluation harnesses, statistical pipelines, and experimental results for studying how adversarial persuasion attacks degrade LLM-based legal citation verifiers, and whether retrieval-augmented generation (RAG) grounding can mitigate these vulnerabilities.

**Key Findings:**
- **Authority persuasion significantly degrades verifiers**: In within-model logistic regressions, authority appeals massively increase correct-to-incorrect flip odds in GPT-OSS 20B ($\text{OR} = 23.333, p < 0.0001$) and Llama 70B ($\text{OR} = 11.953, p = 0.0029$).
- **RAG grounding reduces absolute flips but leaves relative vulnerability**: Grounding acts as a general baseline stabilizer (reducing baseline flip odds by 76% in Llama 70B and 88% in GPT-OSS 120B), but large authority-specific gaps over control persist under grounding (+33.4pp in Llama 70B; +39.8pp in cross-family).
- **Epistemic Deference**: Under authority persuasion, verifiers flip with **high confidence** (mean final confidence 8.56/10; 96.8% at confidence $\ge 8$), whereas neutral reconsideration produces hesitant, low-confidence reversals (mean 6.80/10).
- **Cross-family attacks**: Frontier adversaries attacking top-tier verifiers (GPT-OSS 120B $\to$ Llama 70B) achieve a 98.0% ungrounded True ASR (vs. 70.6% control) and 86.1% grounded True ASR (vs. 46.3% control).

---

## Repository Structure

```
NLLP/
├── data/                               # Benchmark and supplementary dataset files
│   ├── legal_dataset.json              # Benchmark dataset (176 items, balanced across 3 classes)
│   ├── errors_legal_dataset.json       # Items flagged during dataset QA review
│   └── extra_fabricated_cases.json     # Additional synthetic cases for class balance
│
├── notebooks/                          # Interactive experiment notebooks
│   ├── 02_main_experiment_openrouter.ipynb   # Within-family experiments (Llama & GPT-OSS)
│   ├── 03_crossfamily_llama_v_gptoss.ipynb   # Cross-family experiment (Llama 70B × GPT-OSS 120B)
│   └── robustness_wording_check.ipynb        # Prompt register & wording robustness check
│
├── scripts/                            # Dataset generation, evaluation, and analysis scripts
│   ├── build_dataset.py                # Dataset extraction & generation pipeline (CourtListener + Claude + Gemini)
│   ├── validate_dataset.py             # Dataset schema validation & QA consistency checks
│   ├── check_dataset_counts.py         # Topic and category balance counter
│   ├── build_dataset_4keys.py          # 4-key variant dataset builder
│   ├── compute_metrics.py              # Metrics computation (True ASR, CW-ASR, FPR, FNR)
│   ├── compute_per_pair_breakdown.py   # Per-model-pair and per-arm breakdown tables
│   ├── fit_within_family_regression.py # Within-model logistic regressions (Llama 70B, GPT-OSS 120B, GPT-OSS 20B)
│   ├── fit_logistic_regression.py      # Family-pooled logistic regressions (Models 1, 2, 3)
│   ├── lr_test_model1_vs_model2.py     # Likelihood ratio test & cluster bootstrap
│   ├── verify_epistemic_stats.py       # Epistemic deference & confidence shift auditor
│   ├── generate_paper_figures.py       # Generates publication PDF figures
│   ├── reprocess_failed_trials.py      # Retries failed/truncated API calls with token budget guards
│   ├── rescue_parser_failures.py       # Offline response parser recovery
│   └── fix_parser.py                   # Parser normalization routines
│
├── results/                            # Experimental trial records and summary outputs
│   ├── gptoss_within_family/           # GPT-OSS 20B/120B within-family experiment results
│   │   ├── main/                       # Ungrounded trial JSONs (1,056 trials)
│   │   ├── mitigation/                 # Grounded trial JSONs (678 trials)
│   │   └── summary/                    # Computed summary CSVs & metrics
│   ├── llama_within_family/            # Llama 8B/70B within-family experiment results
│   │   ├── results/main/               # Ungrounded trial JSONs (1,056 trials)
│   │   ├── results/mitigation/         # Grounded trial JSONs (678 trials)
│   │   └── results/summary/            # Computed summary CSVs & metrics
│   ├── crossfamily_llama_gptoss/       # Cross-family experiment results (Llama 70B × GPT-OSS 120B)
│   │   ├── main/                       # Ungrounded trial JSONs (528 trials)
│   │   ├── mitigation/                 # Grounded trial JSONs (339 trials)
│   │   └── summary/                    # Computed cross-family summary CSVs
│   ├── robustness_wording/             # Prompt wording robustness experiment
│   │   └── robustness_results/         # 280 trial JSONs across 7 prompt templates
│   │       └── summary/                # Wording check summary CSVs
│   └── regression_tables/              # Regression models, LR tests, and cell diagnostics
│       ├── logistic_regression_llama_70b.csv
│       ├── logistic_regression_gpt_oss_120b.csv
│       ├── logistic_regression_gpt_oss_20b.csv
│       ├── logistic_regression_model2_2way_interacted.csv
│       ├── logistic_regression_model1_3way_interacted.csv
│       ├── logistic_regression_model3_ungrounded.csv
│       ├── lr_test_model1_vs_model2.csv
│       ├── lr_test_bootstrap_distribution.csv
│       ├── per_pair_per_arm_breakdown.csv
│       └── cw_por_confidence_weighted_metrics.csv
│
├── README.md
└── .gitignore
```

---

## Dataset

The benchmark dataset (`data/legal_dataset.json`) contains **176 items** across three classes:

| Class | Items | Description | QA Validation |
|:---|:---:|:---|:---|
| `valid` | 57 | Core legal proposition extracted from case holding | Gemini: `SUPPORTED` at $\ge 0.90$ confidence |
| `real_wrong_content` | 56 | Minimally adversarially edited valid claim (same holding) | Gemini: `NOT_SUPPORTED` at $\ge 0.90$ confidence |
| `fabricated` | 63 | Synthetic non-existent case, citation, and proposition | CourtListener collision check + Gemini QA |

- **Source Corpus**: Federal court opinions from CourtListener (Free Law Project)
- **Generator**: Claude Sonnet 4.6 (temperature 0.0)
- **QA Validator**: Gemini (independent model family to prevent circularity)
- **Independent Cross-Validation**: 40-item subset cross-validated via OpenAI GPT 5.6 Luna (100% agreement)

---

## Experimental Setup

$3{,}468$ within-family trials + $867$ cross-family trials:

| Setup | Verifier | Adversary | Arm | Trials |
|:---|:---|:---|:---:|:---:|
| Llama (within-family) | Llama 3.1-8B / 3.3-70B | Llama 3.3-70B / 3.1-8B | Ungrounded + Grounded | 1,734 |
| GPT-OSS (within-family) | GPT-OSS 20B / 120B | GPT-OSS 120B / 20B | Ungrounded + Grounded | 1,734 |
| Cross-family | Llama 3.3-70B | GPT-OSS 120B | Ungrounded + Grounded | 867 |

**Conditions**:
1. `control`: Reconsideration request with zero persuasive framing, authority claims, or new evidence.
2. `authority`: Persuasive attack posing as an expert legal scholar with assertive argumentation.
3. `fabricated_citation`: Persuasive attack referencing fabricated or misrepresented case law/statutes.

**Arms**:
- **Ungrounded**: Verifier receives only the legal proposition and citation.
- **Grounded**: Verifier additionally receives retrieved CourtListener holding text.

---

## Metrics

- **True ASR (Attack Success Rate)**: Fraction of initially-correct judgments that flip to incorrect ($C \to I$) after challenge:
  $$\text{true\_ASR} = \frac{\sum_{i \in \mathcal{C}_{\text{init}}} \mathbb{I}(\text{Flip}_i = \text{C}\to\text{I})}{N_{\text{init\_correct}}}$$
- **CW-ASR (Confidence-Weighted ASR)** (adapted from [Agarwal & Khanna, 2025](https://arxiv.org/abs/2504.00662)): Weights successful attacks by post-challenge confidence $C_{\text{final}, i} \in [1, 10]$:
  $$\text{CW-ASR} = \frac{1}{N_{\text{init\_correct}}} \sum_{i \in \mathcal{F}_{\text{c2i}}} \left( \frac{C_{\text{final}, i}}{10} \right)$$

---

## Reproducing Results

### 1. Compute summary metrics from raw trial JSONs
```bash
# GPT-OSS within-family
python scripts/compute_metrics.py \
  --main-dir results/gptoss_within_family/main \
  --mitigation-dir results/gptoss_within_family/mitigation

# Llama within-family
python scripts/compute_metrics.py \
  --main-dir results/llama_within_family/results/main \
  --mitigation-dir results/llama_within_family/results/mitigation
```

### 2. Fit within-model logistic regressions (with item clustering)
```bash
python scripts/fit_within_family_regression.py
```

### 3. Compute per-pair & per-arm breakdown tables
```bash
python scripts/compute_per_pair_breakdown.py
```

### 4. Fit family-pooled reference regressions & Likelihood Ratio test
```bash
# Fit Models 1, 2, and 3
python scripts/fit_logistic_regression.py

# Likelihood Ratio test (Model 1 vs. Model 2 with 1,000 cluster bootstrap resamples)
python scripts/lr_test_model1_vs_model2.py
```

### 5. Verify epistemic deference & confidence shift stats
```bash
python scripts/verify_epistemic_stats.py
```

### 6. Verify dataset topic & category balance
```bash
python scripts/check_dataset_counts.py
```

### 7. Generate publication figures
```bash
python scripts/generate_paper_figures.py
```

---

## Key Results Summary

### Per-Pair True ASR (Ungrounded vs. Grounded)

| Verifier | Adversary | Ungrounded Ctrl | Ungrounded Auth | Grounded Ctrl | Grounded Auth | Net Grounded Shift |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Llama 70B | Llama 8B | 64.0% | 95.7% | 31.0% | 64.4% | **+33.4pp** |
| GPT-OSS 120B | GPT-OSS 20B | 47.1% | 40.7% | 9.8% | 28.3% | **+18.5pp** |
| GPT-OSS 20B | GPT-OSS 120B | 10.9% | 72.7% | 11.5% | 32.1% | **+20.6pp** |
| Llama 70B | GPT-OSS 120B | 70.6% | 98.0% | 46.3% | 86.1% | **+39.8pp** |

---

## License

This repository is provided for research and educational purposes. The dataset is derived from publicly available federal court opinions hosted by CourtListener (Free Law Project).
