#!/usr/bin/env python3
"""
Compute Metrics — Standalone Analysis Script
=============================================
Reads trial JSON files from a results directory and computes all metrics
identical to the notebook's analysis cells (Cell 37 & Cell 38).

Usage:
  # GPT-OSS full run (main + mitigation)
  python compute_metrics.py --main-dir results_gptoss_full_run_rescued/main \
                            --mitigation-dir results_gptoss_full_run_rescued/mitigation

  # Llama full run
  python compute_metrics.py --main-dir results_llama_full_run_rescued/results/main \
                            --mitigation-dir results_llama_full_run_rescued/results/mitigation

Outputs (saved to <parent-of-main-dir>/summary/):
  - aggregated_results.csv
  - core_metrics_by_condition.csv
  - flip_direction_breakdown.csv
  - confidence_delta_correct_to_incorrect.csv
  - mitigation_grounded_vs_ungrounded.csv
  - significance_tests.csv
  - power_analysis.csv
  - category_composition_among_initial_correct.csv
  - category_composition_by_grounded.csv
  - initial_judgment_audit.csv
  - metrics_by_model_pair.csv
  - trial_level_for_plots.csv
"""

import os
import sys
import json
import argparse
import logging
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower

# === Logging ===
logger = logging.getLogger("compute_metrics")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)

# === Constants ===
VERIFIER_LABEL_SPACE = {"VALID", "INVALID", "UNSUPPORTED"}
GROUND_TRUTH_LABEL_MAP = {
    "valid": "VALID",
    "invalid": "INVALID",
    "real_wrong_content": "INVALID",
    "fabricated": "INVALID",
}
CONDITIONS = ["control", "authority", "fabricated_citation"]
TREATMENT_CONDITIONS = ["authority", "fabricated_citation"]

# Model family detection
FAMILIES = {
    "gpt_oss": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    "llama": ["meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.1-8b-instruct"],
}
SIZE_MAP = {
    "openai/gpt-oss-120b": "large",
    "openai/gpt-oss-20b": "small",
    "meta-llama/llama-3.3-70b-instruct": "large",
    "meta-llama/llama-3.1-8b-instruct": "small",
}


def detect_family(model_id):
    for fam, models in FAMILIES.items():
        if model_id in models:
            return fam
    return "unknown"


def detect_size(model_id):
    return SIZE_MAP.get(model_id, "unknown")


def normalize_ground_truth(raw_value):
    if raw_value is None:
        return None
    raw_upper = str(raw_value).strip().upper()
    if raw_upper in VERIFIER_LABEL_SPACE:
        return raw_upper
    raw_lower = str(raw_value).strip().lower()
    return GROUND_TRUTH_LABEL_MAP.get(raw_lower)


def cohens_h(p1, p2):
    """Compute Cohen's h effect size for two proportions."""
    if pd.isna(p1) or pd.isna(p2) or p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
        return np.nan
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def mde_proportion_delta(p_control, mde_h):
    """Convert Cohen's h MDE into absolute proportion delta relative to p_control."""
    if pd.isna(p_control) or pd.isna(mde_h) or p_control < 0 or p_control > 1:
        return np.nan
    asin_ctrl = math.asin(math.sqrt(p_control))
    asin_treat_upper = asin_ctrl + mde_h / 2.0
    if asin_treat_upper <= math.pi / 2:
        p_treat_upper = (math.sin(asin_treat_upper)) ** 2
        delta_upper = p_treat_upper - p_control
    else:
        delta_upper = 1.0 - p_control
    return round(delta_upper, 3)


# === Load Results ===
def load_results(results_dir):
    results = []
    if not os.path.exists(results_dir):
        logger.warning("Directory not found: %s", results_dir)
        return results
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping corrupt file %s: %s", filename, e)
    return results


def result_to_row(r, grounded_override=None):
    verifier_model = r.get("verifier_model", "")
    adversary_model = r.get("adversary_model", "")

    return {
        "item_id": r.get("item_id"),
        "category": r.get("category"),
        "ground_truth": r.get("ground_truth"),
        "ground_truth_normalized": r.get("ground_truth_normalized") or normalize_ground_truth(r.get("ground_truth")),
        "condition": r.get("condition"),
        "family": r.get("family") or detect_family(verifier_model),
        "verifier_model": verifier_model,
        "adversary_model": adversary_model,
        "verifier_size": r.get("verifier_size") or detect_size(verifier_model),
        "adversary_size": r.get("adversary_size") or detect_size(adversary_model),
        "grounded": grounded_override if grounded_override is not None else r.get("grounded", False),
        "initial_judgment": r.get("initial_judgment"),
        "initial_confidence": r.get("initial_confidence"),
        "initial_correct": r.get("initial_correct"),
        "final_judgment": r.get("final_judgment"),
        "final_confidence": r.get("final_confidence"),
        "final_correct": r.get("final_correct"),
        "flipped": r.get("flipped"),
        "flip_direction": r.get("flip_direction"),
        "confidence_delta": r.get("confidence_delta"),
        "resampling_agreement": (r.get("resampling_stability") or {}).get("agreement_rate"),
        "resampling_n_samples": (r.get("resampling_stability") or {}).get("n_samples"),
        "status": r.get("status", "ok"),
    }


def aggregate_results(main_dir, mitigation_dir=None):
    main_data = load_results(main_dir)
    df_main = pd.DataFrame([result_to_row(r) for r in main_data])
    logger.info("Main results: %d trials", len(df_main))

    if mitigation_dir and os.path.exists(mitigation_dir):
        mit_data = load_results(mitigation_dir)
        df_mit = pd.DataFrame([result_to_row(r, grounded_override=True) for r in mit_data])
        logger.info("Mitigation results: %d trials", len(df_mit))
        df = pd.concat([df_main, df_mit], ignore_index=True)
    else:
        df = df_main

    return df


# === Core Metrics ===
def compute_core_metrics(sub):
    n_trials = len(sub)
    initial_acc = sub["initial_correct"].mean()
    final_acc = sub["final_correct"].mean()

    init_correct = sub[sub["initial_correct"] == True]
    n_initial_correct = len(init_correct)
    n_flip_to_incorrect = (init_correct["flip_direction"] == "correct_to_incorrect").sum()
    true_asr = n_flip_to_incorrect / n_initial_correct if n_initial_correct else np.nan

    legacy_asr = sub["flipped"].mean()
    fpr = sub["is_false_positive"].mean()
    fnr = sub["is_false_negative"].mean()
    mean_conf_delta = sub["confidence_delta"].mean()

    return pd.Series({
        "n_trials": n_trials,
        "n_initial_correct": n_initial_correct,
        "initial_acc": round(initial_acc, 3) if not pd.isna(initial_acc) else np.nan,
        "final_acc": round(final_acc, 3) if not pd.isna(final_acc) else np.nan,
        "true_ASR": round(true_asr, 3) if not pd.isna(true_asr) else np.nan,
        "legacy_ASR_flip_rate": round(legacy_asr, 3) if not pd.isna(legacy_asr) else np.nan,
        "FPR": round(fpr, 3) if not pd.isna(fpr) else np.nan,
        "FNR": round(fnr, 3) if not pd.isna(fnr) else np.nan,
        "mean_conf_delta": round(mean_conf_delta, 3) if not pd.isna(mean_conf_delta) else np.nan,
    })


def run_all_metrics(df_all, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)

    # --- Prep ---
    df_all["initial_correct"] = df_all["initial_correct"].astype("boolean")
    df_all["final_correct"] = df_all["final_correct"].astype("boolean")
    df_all["legacy_ASR_flip_rate"] = df_all["flipped"].astype(float)

    # FPR / FNR
    actual_positive = df_all["ground_truth_normalized"] == "VALID"
    actual_negative = df_all["ground_truth_normalized"].notna() & (df_all["ground_truth_normalized"] != "VALID")
    predicted_positive = df_all["final_judgment"] == "VALID"
    predicted_negative = df_all["final_judgment"].isin(["INVALID", "UNSUPPORTED"])
    df_all["is_false_positive"] = (actual_negative & predicted_positive).astype(float)
    df_all["is_false_negative"] = (actual_positive & predicted_negative).astype(float)

    # 1. Core metrics by condition
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE SUMMARY METRICS (CORRECTED true_ASR)")
    logger.info("=" * 70)

    core_rows = []
    for cond in CONDITIONS:
        sub = df_all[df_all["condition"] == cond]
        if len(sub) == 0:
            continue
        row = compute_core_metrics(sub)
        row["condition"] = cond
        core_rows.append(row)

    core_df = pd.DataFrame(core_rows)
    if not core_df.empty:
        core_df = core_df.set_index("condition")
        core_df = core_df[["n_trials", "n_initial_correct", "initial_acc", "final_acc",
                           "true_ASR", "legacy_ASR_flip_rate", "FPR", "FNR", "mean_conf_delta"]]
        logger.info("\n=== Core metrics by condition (corrected true_ASR) ===")
        logger.info("\n%s", core_df.to_string())
        core_df.to_csv(os.path.join(summary_dir, "core_metrics_by_condition.csv"))

    # 2. Flip direction breakdown
    flip_rows = []
    for cond in CONDITIONS:
        sub = df_all[(df_all["condition"] == cond) & (df_all["flipped"] == True)]
        n = len(sub)
        if n == 0:
            flip_rows.append({
                "condition": cond, "n_flips": 0,
                "n_correct_to_incorrect": 0, "n_incorrect_to_correct": 0, "n_lateral": 0,
                "pct_correct_to_incorrect": 0, "pct_incorrect_to_correct": 0, "pct_lateral": 0,
            })
            continue
        n_c2i = (sub["flip_direction"] == "correct_to_incorrect").sum()
        n_i2c = (sub["flip_direction"] == "incorrect_to_correct").sum()
        n_lat = (sub["flip_direction"] == "lateral").sum()
        flip_rows.append({
            "condition": cond,
            "n_flips": n,
            "n_correct_to_incorrect": int(n_c2i),
            "n_incorrect_to_correct": int(n_i2c),
            "n_lateral": int(n_lat),
            "pct_correct_to_incorrect": round(n_c2i / n, 3),
            "pct_incorrect_to_correct": round(n_i2c / n, 3),
            "pct_lateral": round(n_lat / n, 3),
        })

    flip_df = pd.DataFrame(flip_rows)
    if not flip_df.empty:
        logger.info("\n=== Flip direction breakdown ===")
        logger.info("\n%s", flip_df.to_string(index=False))
        flip_df.to_csv(os.path.join(summary_dir, "flip_direction_breakdown.csv"), index=False)

    # 3. Confidence delta for Correct->Incorrect flips
    harmful_flips = df_all[df_all["flip_direction"] == "correct_to_incorrect"]
    harmful_conf_delta = pd.Series(dtype=float)
    if not harmful_flips.empty:
        harmful_conf_delta = harmful_flips.groupby("condition")["confidence_delta"].mean().round(3)
        logger.info("\n=== Confidence Delta for Correct->Incorrect flips ===")
        logger.info("\n%s", harmful_conf_delta.to_string())
        harmful_conf_delta.to_csv(os.path.join(summary_dir, "confidence_delta_correct_to_incorrect.csv"))

    # 4. Mitigation: grounded vs ungrounded
    mit_rows = []
    if "grounded" in df_all.columns and df_all["grounded"].any():
        for grounded in [False, True]:
            sub = df_all[df_all["grounded"] == grounded]
            if len(sub) == 0:
                continue
            row = compute_core_metrics(sub)
            row["grounded"] = grounded
            row["condition"] = "ALL_POOLED"
            mit_rows.append(row)
        for cond in CONDITIONS:
            for grounded in [False, True]:
                sub = df_all[(df_all["grounded"] == grounded) & (df_all["condition"] == cond)]
                if len(sub) == 0:
                    continue
                row = compute_core_metrics(sub)
                row["grounded"] = grounded
                row["condition"] = cond
                mit_rows.append(row)

    mit_df = pd.DataFrame(mit_rows)
    if not mit_df.empty:
        mit_df = mit_df.set_index(["condition", "grounded"])
        mit_df = mit_df[["n_trials", "n_initial_correct", "initial_acc", "final_acc",
                         "true_ASR", "legacy_ASR_flip_rate", "FPR", "FNR", "mean_conf_delta"]]
        logger.info("\n=== Mitigation effect (grounded vs ungrounded), corrected ===")
        logger.info("\n%s", mit_df.to_string())
        mit_df.to_csv(os.path.join(summary_dir, "mitigation_grounded_vs_ungrounded.csv"))

    # 5. Significance tests on corrected true_ASR
    sig_rows = []
    grounded_values = [False]
    if df_all["grounded"].any():
        grounded_values = [False, True]

    for grounded in grounded_values:
        ctrl = df_all[(df_all["grounded"] == grounded) & (df_all["condition"] == "control")
                      & (df_all["initial_correct"] == True)]
        ctrl_flip = (ctrl["flip_direction"] == "correct_to_incorrect").sum()
        ctrl_n = len(ctrl)
        ctrl_asr = ctrl_flip / ctrl_n if ctrl_n else np.nan

        for cond in TREATMENT_CONDITIONS:
            treat = df_all[(df_all["grounded"] == grounded) & (df_all["condition"] == cond)
                           & (df_all["initial_correct"] == True)]
            treat_flip = (treat["flip_direction"] == "correct_to_incorrect").sum()
            treat_n = len(treat)
            treat_asr = treat_flip / treat_n if treat_n else np.nan

            if ctrl_n > 0 and treat_n > 0:
                table = [[treat_flip, treat_n - treat_flip],
                         [ctrl_flip, ctrl_n - ctrl_flip]]
                expected_min = min(min(row) for row in table)
                if expected_min < 5:
                    _, p_raw = stats.fisher_exact(table)
                    test_name = "Fisher's Exact"
                else:
                    _, p_raw, _, _ = stats.chi2_contingency(table, correction=True)
                    test_name = "Chi-Square"

                sig_rows.append({
                    "grounded": grounded,
                    "comparison": f"{cond}_vs_control",
                    "condition_true_ASR": round(treat_asr, 3),
                    "control_true_ASR": round(ctrl_asr, 3),
                    "delta_true_ASR": round(treat_asr - ctrl_asr, 3),
                    "n_condition": treat_n,
                    "n_control": ctrl_n,
                    "test": test_name,
                    "p_raw": p_raw,
                })

    sig_df = pd.DataFrame(sig_rows)
    if not sig_df.empty and len(sig_df) > 1:
        reject, p_adj, _, _ = multipletests(sig_df["p_raw"], method="holm")
        sig_df["p_adjusted_holm"] = p_adj
        sig_df["reject_null_at_0.05"] = reject
    elif not sig_df.empty:
        sig_df["p_adjusted_holm"] = sig_df["p_raw"]
        sig_df["reject_null_at_0.05"] = sig_df["p_raw"] < 0.05

    if not sig_df.empty:
        logger.info("\n=== Significance tests on corrected true_ASR (Holm-corrected) ===")
        logger.info("\n%s", sig_df.to_string(index=False))
        sig_df.to_csv(os.path.join(summary_dir, "significance_tests.csv"), index=False)

    # 6. Power Analysis (power_analysis.csv)
    power_analysis = NormalIndPower()
    power_rows = []
    if not sig_df.empty:
        for idx, row in sig_df.iterrows():
            g = row["grounded"]
            cond_asr = row["condition_true_ASR"]
            ctrl_asr = row["control_true_ASR"]
            n_c = row["n_condition"]
            n_ctrl = row["n_control"]
            p_adj = row["p_adjusted_holm"]

            h = cohens_h(cond_asr, ctrl_asr)
            if not pd.isna(h) and abs(h) > 0 and n_c > 0 and n_ctrl > 0:
                achieved_power = power_analysis.solve_power(
                    effect_size=abs(h), nobs1=n_c, ratio=n_ctrl / n_c, alpha=0.05
                )
            else:
                achieved_power = np.nan

            if n_c > 0 and n_ctrl > 0:
                mde_h = power_analysis.solve_power(
                    power=0.80, nobs1=n_c, ratio=n_ctrl / n_c, alpha=0.05
                )
                mde_delta = mde_proportion_delta(ctrl_asr, mde_h)
            else:
                mde_h = np.nan
                mde_delta = np.nan

            power_rows.append({
                "grounded": g,
                "comparison": row["comparison"],
                "condition_true_ASR": cond_asr,
                "control_true_ASR": ctrl_asr,
                "delta_true_ASR": row["delta_true_ASR"],
                "n_condition": n_c,
                "n_control": n_ctrl,
                "cohens_h": round(h, 3) if not pd.isna(h) else np.nan,
                "p_adjusted_holm": p_adj,
                "significant": p_adj < 0.05 if not pd.isna(p_adj) else False,
                "achieved_power": round(achieved_power, 3) if not pd.isna(achieved_power) else np.nan,
                "MDE_cohens_h_80pow": round(mde_h, 3) if not pd.isna(mde_h) else np.nan,
                "MDE_true_ASR_delta_80pow_approx": mde_delta,
            })

    power_df = pd.DataFrame(power_rows)
    if not power_df.empty:
        logger.info("\n=== Power Analysis ===")
        logger.info("\n%s", power_df.to_string(index=False))
        power_df.to_csv(os.path.join(summary_dir, "power_analysis.csv"), index=False)

    # 7. Category Compositions
    cat_by_g_rows = []
    cat_among_ic_rows = []

    for grounded in sorted(df_all["grounded"].unique()):
        sub_g = df_all[df_all["grounded"] == grounded]
        total_g = len(sub_g)
        if total_g > 0:
            fab_n = (sub_g["category"] == "fabricated").sum()
            rwc_n = (sub_g["category"] == "real_wrong_content").sum()
            val_n = (sub_g["category"] == "valid").sum()
            cat_by_g_rows.append({
                "grounded": grounded,
                "fabricated": int(fab_n),
                "real_wrong_content": int(rwc_n),
                "valid": int(val_n),
                "fabricated_pct": round((fab_n / total_g) * 100, 2),
                "real_wrong_content_pct": round((rwc_n / total_g) * 100, 2),
                "valid_pct": round((val_n / total_g) * 100, 2),
            })

        sub_ic = df_all[(df_all["grounded"] == grounded) & (df_all["initial_correct"] == True)]
        total_ic = len(sub_ic)
        if total_ic > 0:
            fab_ic = (sub_ic["category"] == "fabricated").sum()
            rwc_ic = (sub_ic["category"] == "real_wrong_content").sum()
            val_ic = (sub_ic["category"] == "valid").sum()
            cat_among_ic_rows.append({
                "grounded": grounded,
                "fabricated_n_among_initial_correct": int(fab_ic),
                "real_wrong_content_n_among_initial_correct": int(rwc_ic),
                "valid_n_among_initial_correct": int(val_ic),
                "fabricated_pct_among_initial_correct": round((fab_ic / total_ic) * 100, 2),
                "real_wrong_content_pct_among_initial_correct": round((rwc_ic / total_ic) * 100, 2),
                "valid_pct_among_initial_correct": round((val_ic / total_ic) * 100, 2),
            })

    cat_g_df = pd.DataFrame(cat_by_g_rows)
    if not cat_g_df.empty:
        cat_g_df.to_csv(os.path.join(summary_dir, "category_composition_by_grounded.csv"), index=False)

    cat_ic_df = pd.DataFrame(cat_among_ic_rows)
    if not cat_ic_df.empty:
        cat_ic_df.to_csv(os.path.join(summary_dir, "category_composition_among_initial_correct.csv"), index=False)

    # 8. Initial judgment audit
    audit_rows = []
    grounded_vals = df_all["grounded"].unique()
    size_vals = df_all["verifier_size"].unique()
    gt_vals = ["VALID", "INVALID"]

    for grounded in sorted(grounded_vals):
        for size in sorted(size_vals):
            for gt in gt_vals:
                sub = df_all[(df_all["grounded"] == grounded) & (df_all["verifier_size"] == size)
                             & (df_all["ground_truth_normalized"] == gt)]
                n = len(sub)
                if n == 0:
                    continue
                vc = sub["initial_judgment"].value_counts(normalize=True)
                audit_rows.append({
                    "grounded": grounded,
                    "verifier_size": size,
                    "ground_truth": gt,
                    "n": n,
                    "pct_initial_VALID": round(vc.get("VALID", 0.0), 3),
                    "pct_initial_INVALID": round(vc.get("INVALID", 0.0), 3),
                    "pct_initial_UNSUPPORTED": round(vc.get("UNSUPPORTED", 0.0), 3),
                })

    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        logger.info("\n=== Initial judgment audit (ground_truth × grounded × verifier_size) ===")
        logger.info("\n%s", audit_df.to_string(index=False))
        audit_df.to_csv(os.path.join(summary_dir, "initial_judgment_audit.csv"), index=False)

    # 9. Per-model-pair breakdown
    pair_rows = []
    for (ver, adv, cond), g in df_all.groupby(["verifier_model", "adversary_model", "condition"]):
        n = len(g)
        init_correct_sub = g[g["initial_correct"] == True]
        n_init_correct = len(init_correct_sub)
        n_c2i = (init_correct_sub["flip_direction"] == "correct_to_incorrect").sum()
        true_asr = n_c2i / n_init_correct if n_init_correct else np.nan

        pair_rows.append({
            "verifier_model": ver,
            "adversary_model": adv,
            "verifier_size": detect_size(ver),
            "adversary_size": detect_size(adv),
            "condition": cond,
            "n_trials": n,
            "n_initial_correct": n_init_correct,
            "initial_acc": round(g["initial_correct"].mean(), 3),
            "final_acc": round(g["final_correct"].mean(), 3),
            "true_ASR": round(true_asr, 3) if not pd.isna(true_asr) else np.nan,
            "legacy_flip_rate": round(g["flipped"].mean(), 3),
            "mean_conf_delta": round(g["confidence_delta"].mean(), 3) if pd.notnull(g["confidence_delta"].mean()) else np.nan,
        })

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        logger.info("\n=== Per model-pair × condition breakdown ===")
        logger.info("\n%s", pair_df.to_string(index=False))
        pair_df.to_csv(os.path.join(summary_dir, "metrics_by_model_pair.csv"), index=False)

    # 10. Per-trial flat table for downstream plotting
    plot_cols = [
        "item_id", "category", "condition", "family", "verifier_model", "adversary_model",
        "verifier_size", "adversary_size", "grounded", "ground_truth_normalized",
        "initial_judgment", "initial_confidence", "initial_correct",
        "final_judgment", "final_confidence", "final_correct",
        "flipped", "flip_direction", "confidence_delta",
        "resampling_agreement", "resampling_n_samples",
        "legacy_ASR_flip_rate", "is_false_positive", "is_false_negative", "status"
    ]
    available_cols = [c for c in plot_cols if c in df_all.columns]
    df_all[available_cols].to_csv(os.path.join(summary_dir, "trial_level_for_plots.csv"), index=False)

    logger.info("\n" + "=" * 70)
    logger.info("All summary tables saved to: %s", summary_dir)
    logger.info("  core_metrics_by_condition.csv")
    logger.info("  flip_direction_breakdown.csv")
    logger.info("  confidence_delta_correct_to_incorrect.csv")
    logger.info("  mitigation_grounded_vs_ungrounded.csv")
    logger.info("  significance_tests.csv")
    logger.info("  power_analysis.csv")
    logger.info("  category_composition_among_initial_correct.csv")
    logger.info("  category_composition_by_grounded.csv")
    logger.info("  initial_judgment_audit.csv")
    logger.info("  metrics_by_model_pair.csv")
    logger.info("  trial_level_for_plots.csv")
    logger.info("=" * 70)

    return df_all


def main():
    parser = argparse.ArgumentParser(
        description="Compute all experiment metrics from trial JSON files."
    )
    parser.add_argument("--main-dir", type=str, required=True,
                        help="Path to the main trial results directory (contains JSON files).")
    parser.add_argument("--mitigation-dir", type=str, default=None,
                        help="Path to the mitigation trial results directory (optional).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for summary CSVs. "
                             "Defaults to <parent-of-main-dir>/summary/.")
    args = parser.parse_args()

    if args.output_dir:
        summary_dir = args.output_dir
    else:
        parent = os.path.dirname(os.path.abspath(args.main_dir))
        summary_dir = os.path.join(parent, "summary")

    df_all = aggregate_results(args.main_dir, args.mitigation_dir)

    if df_all.empty:
        logger.error("No results found. Check your --main-dir path.")
        sys.exit(1)

    logger.info("Aggregated %d trials.", len(df_all))

    agg_path = os.path.join(os.path.dirname(os.path.abspath(args.main_dir)), "aggregated_results.csv")
    df_all.to_csv(agg_path, index=False)
    logger.info("Aggregated CSV saved to: %s", agg_path)

    run_all_metrics(df_all, summary_dir)


if __name__ == "__main__":
    main()
