#!/usr/bin/env python3
"""
Fit Logistic Regression Models with Cluster-Robust Standard Errors
===================================================================
Fits logistic regression models on initially-correct trial records from the
rescued dataset, with rigorous input filtering and convergence diagnostics.

Models fitted:
  1. Model 1 (Appendix): Fully interacted 3-way logit (condition * grounded * family)
     — Use for deriving per-cell net ORs and for the 3-way Auth x Grounded x Llama
       interaction. Not primary because several 3-way cells are small.
  2. Model 2 (Primary for RQ3): Two-way interacted logit
     (condition*grounded + condition*family + grounded*family)
     — Use for the headline RQ3 test (Auth x Grounded interaction).
  3. Model 3 (Supplementary): Ungrounded arm logit (condition * family)
     — Use for ungrounded-only cross-family contrasts.

Methodology:
  - Standard Errors: Huber-White sandwich estimator clustered by item_id (CR1).
  - Target Variable: c2i (1 if verdict flipped from correct to incorrect, 0 otherwise).
  - Data Filtering: status == "ok", non-null initial_correct, non-null flip_direction.
  - No external statsmodels dependency (pure numpy/scipy).

Usage:
  python fit_logistic_regression.py
  python fit_logistic_regression.py --output-dir ./summary_csvs
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm


# === Helpers ===

def detect_family(model_id):
    """Detect model family from verifier model ID string."""
    if "gpt" in str(model_id).lower():
        return "gpt_oss"
    elif "llama" in str(model_id).lower():
        return "llama"
    return "unknown"


def load_results(results_dir):
    """Load all trial JSON files from a directory."""
    results = []
    if not os.path.exists(results_dir):
        print(f"[WARNING] Directory not found: {results_dir}")
        return results
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    results.append(json.load(f))
            except Exception:
                pass
    return results


def result_to_row(r, grounded_override=None):
    """
    Convert a trial JSON dict to a flat row dict.

    grounded_override: We pass True/False based on which folder (main vs mitigation)
    the file was loaded from. This is by design — folder membership is the ground-truth
    indicator of grounding status in the experimental setup. The per-trial `r.get("grounded")`
    field, if present, would be redundant; we override it to avoid any misfiled-trial risk.
    """
    verifier_model = r.get("verifier_model", "")
    return {
        "item_id": r.get("item_id"),
        "category": r.get("category"),
        "ground_truth": r.get("ground_truth"),
        "condition": r.get("condition"),
        "family": r.get("family") or detect_family(verifier_model),
        "verifier_model": verifier_model,
        "adversary_model": r.get("adversary_model", ""),
        "grounded": grounded_override if grounded_override is not None else r.get("grounded", False),
        "initial_correct": r.get("initial_correct"),
        "initial_judgment": r.get("initial_judgment"),
        "final_judgment": r.get("final_judgment"),
        "final_correct": r.get("final_correct"),
        "flipped": r.get("flipped"),
        "flip_direction": r.get("flip_direction"),
        "status": r.get("status", "ok"),
    }


# === Logistic Regression Engine ===

def fit_logit_clustered(X, y, cluster_ids, var_names, model_label=""):
    """
    Fits logistic regression via Newton-Raphson with step-halving line search
    and estimates cluster-robust standard errors (Huber-White sandwich, CR1).

    Returns:
      res_df: DataFrame with coefficients, cluster SEs, z-stats, p-values, ORs, CIs.
      beta: coefficient vector.
      V_cluster: cluster-robust variance-covariance matrix.
      converged: bool indicating convergence.
      n_iter: number of iterations used.
    """
    N, K = X.shape
    beta = np.zeros(K)
    max_iter = 200
    tol = 1e-8
    converged = False
    n_iter = 0

    def log_lik(b):
        xb = X @ b
        # Numerically stable log-likelihood
        return np.sum(y * xb - np.logaddexp(0, xb))

    for it in range(max_iter):
        n_iter = it + 1
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        grad = X.T @ (y - p)
        W = p * (1.0 - p)
        H = -X.T @ (W[:, None] * X)

        try:
            delta = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            print(f"  [WARNING] Singular Hessian at iteration {n_iter} — possible separation.")
            break

        # Step-halving line search: ensure log-likelihood improves
        step = 1.0
        ll_current = log_lik(beta)
        for _ in range(20):
            ll_new = log_lik(beta + step * delta)
            if ll_new >= ll_current - 1e-10:
                break
            step *= 0.5
        beta += step * delta

        if np.max(np.abs(step * delta)) < tol:
            converged = True
            break

    # Report convergence
    status_str = "CONVERGED" if converged else "DID NOT CONVERGE"
    print(f"  [{status_str}] {model_label}: {n_iter} iterations, max|delta|={np.max(np.abs(step * delta)):.2e}")
    if not converged:
        print(f"  [WARNING] Model did not converge in {max_iter} iterations. Coefficients may be unreliable.")

    # Final predicted probabilities
    p = 1.0 / (1.0 + np.exp(-X @ beta))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    W = p * (1.0 - p)
    Hessian = X.T @ (W[:, None] * X)

    # Check condition number for near-separation
    cond = np.linalg.cond(Hessian)
    if cond > 1e10:
        print(f"  [WARNING] Hessian condition number = {cond:.2e} — possible quasi-separation.")

    Bread = np.linalg.inv(Hessian)

    # Cluster-robust Meat matrix
    scores = (y - p)[:, None] * X
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    Meat = np.zeros((K, K))
    for g in unique_clusters:
        idx = (cluster_ids == g)
        u_g = scores[idx].sum(axis=0)
        Meat += np.outer(u_g, u_g)

    # Small-sample CR1 adjustment: [G / (G - 1)] * [(N - 1) / (N - K)]
    df_c = (G / (G - 1)) * ((N - 1) / (N - K))
    V_cluster = df_c * (Bread @ Meat @ Bread)

    se = np.sqrt(np.diag(V_cluster))
    z = beta / se
    p_values = 2 * (1 - norm.cdf(np.abs(z)))
    or_val = np.exp(beta)
    or_ci_lower = np.exp(beta - 1.96 * se)
    or_ci_upper = np.exp(beta + 1.96 * se)

    res_df = pd.DataFrame({
        'Predictor': var_names,
        'Coef': beta,
        'Cluster_SE': se,
        'z_stat': z,
        'p_value': p_values,
        'OR': or_val,
        'OR_95CI_Lower': or_ci_lower,
        'OR_95CI_Upper': or_ci_upper
    })
    return res_df, beta, V_cluster, converged, n_iter


def find_dir(rel_path):
    for prefix in ["results", ".", ".."]:
        p = os.path.join(prefix, rel_path)
        if os.path.isdir(p):
            return p
    return rel_path


def main():
    parser = argparse.ArgumentParser(
        description="Fit logistic regression models with cluster-robust SEs by item_id.")
    parser.add_argument("--gptoss-main",
                        default=find_dir("gptoss_within_family/main"))
    parser.add_argument("--gptoss-mit",
                        default=find_dir("gptoss_within_family/mitigation"))
    parser.add_argument("--llama-main",
                        default=find_dir("llama_within_family/results/main"))
    parser.add_argument("--llama-mit",
                        default=find_dir("llama_within_family/results/mitigation"))
    default_out = find_dir("regression_tables")
    if not os.path.isdir(default_out):
        default_out = "results/regression_tables"
    parser.add_argument("--output-dir", default=default_out)
    args = parser.parse_args()

    # ── Load ──
    gptoss_main = load_results(args.gptoss_main)
    gptoss_mit = load_results(args.gptoss_mit)
    llama_main = load_results(args.llama_main)
    llama_mit = load_results(args.llama_mit)

    rows = []
    for r in gptoss_main:
        rows.append(result_to_row(r, grounded_override=False))
    for r in gptoss_mit:
        rows.append(result_to_row(r, grounded_override=True))
    for r in llama_main:
        rows.append(result_to_row(r, grounded_override=False))
    for r in llama_mit:
        rows.append(result_to_row(r, grounded_override=True))

    df = pd.DataFrame(rows)
    n_total = len(df)
    print(f"[INFO] Total loaded trials: {n_total}")

    # ── Filter 1: status == "ok" ──
    n_bad_status = (df["status"] != "ok").sum()
    status_breakdown = df[df["status"] != "ok"].groupby(["family", "condition", "grounded", "status"]).size()
    if n_bad_status > 0:
        print(f"\n[FILTER] Dropping {n_bad_status} trials with status != 'ok':")
        print(status_breakdown.to_string())
    df = df[df["status"] == "ok"].copy()
    print(f"[INFO] After status filter: {len(df)} trials (dropped {n_bad_status})")

    # ── Filter 2: non-null initial_judgment and initial_correct ──
    null_init = df["initial_correct"].isna() | df["initial_judgment"].isna()
    n_null_init = null_init.sum()
    if n_null_init > 0:
        print(f"\n[FILTER] Dropping {n_null_init} trials with null initial_judgment/initial_correct:")
        print(df[null_init].groupby(["family", "condition", "grounded"]).size().to_string())
    df = df[~null_init].copy()
    print(f"[INFO] After initial-judgment filter: {len(df)} trials")

    # ── Select initially-correct trials ──
    df_init = df[df["initial_correct"] == True].copy()
    print(f"[INFO] Initially-correct trials: {len(df_init)}")

    # ── Filter 3: exclude trials with genuinely unparsed final_judgment ──
    # IMPORTANT: flip_direction is NaN whenever flipped=False (no flip happened).
    # That is NOT missing data — it's the legitimate "stayed correct" case (c2i=0).
    # Actual missing data = final_judgment is None/NaN (unparsed/truncated output).
    null_final = df_init["final_judgment"].isna()
    n_null_final = null_final.sum()
    if n_null_final > 0:
        print(f"\n[FILTER] Dropping {n_null_final} initially-correct trials with null final_judgment (unparsed):")
        breakdown = df_init[null_final].groupby(["family", "condition", "grounded"]).size()
        print(breakdown.to_string())
        print("\n  ^^^ If concentrated in fabricated_citation, Fab coefficients may be biased toward null.")
    df_init = df_init[~null_final].copy()
    print(f"[INFO] Final analysis sample (initially-correct, status ok, final_judgment parsed): N = {len(df_init)}")

    # ── Compute c2i ──
    # c2i=1 if verdict flipped from correct to incorrect; c2i=0 otherwise
    # (including trials that stayed correct, where flip_direction is NaN and flipped=False)
    df_init["c2i"] = (df_init["flip_direction"] == "correct_to_incorrect").astype(int)


    # ── Per-cell diagnostic table (for Model 1 interaction terms) ──
    print("\n" + "=" * 95)
    print("PER-CELL DIAGNOSTICS: n and n_c2i by (family, grounded, condition)")
    print("=" * 95)
    cell_diag = df_init.groupby(["family", "grounded", "condition"]).agg(
        n=("c2i", "count"),
        n_c2i=("c2i", "sum"),
        c2i_rate=("c2i", "mean")
    ).round(4)
    print(cell_diag.to_string())
    cell_diag.to_csv(os.path.join(args.output_dir, "logistic_regression_cell_diagnostics.csv"))

    # ── Check for near-empty cells ──
    min_cell = cell_diag["n"].min()
    min_c2i = cell_diag["n_c2i"].min()
    if min_cell < 20:
        print(f"\n  [WARNING] Smallest cell has only n={min_cell} — 3-way interaction estimates may be unstable.")
    if min_c2i < 5:
        print(f"  [WARNING] Smallest c2i count is {min_c2i} — risk of quasi-separation in that cell.")

    # ── Build indicator vectors ──
    is_auth = (df_init["condition"] == "authority").astype(float).values
    is_fab = (df_init["condition"] == "fabricated_citation").astype(float).values
    is_grounded = (df_init["grounded"] == True).astype(float).values
    is_llama = (df_init["family"] == "llama").astype(float).values
    cluster_ids = df_init["item_id"].values
    y = df_init["c2i"].values
    n_clusters = len(np.unique(cluster_ids))
    print(f"\n[INFO] Number of item clusters (G): {n_clusters}")

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 1: Fully interacted 3-way logit (condition * grounded * family)
    # Use for: derived per-cell net ORs, 3-way Auth x Grounded x Llama test.
    # Caveat: some 3-way cells may be small — check cell_diagnostics table.
    # ══════════════════════════════════════════════════════════════════════
    X1 = np.column_stack([
        np.ones(len(df_init)), is_auth, is_fab, is_grounded, is_llama,
        is_auth * is_grounded, is_fab * is_grounded, is_auth * is_llama, is_fab * is_llama,
        is_grounded * is_llama, is_auth * is_grounded * is_llama, is_fab * is_grounded * is_llama
    ])
    v1 = ["Intercept", "Authority", "Fab_Citation", "Grounded", "Family_Llama",
          "Auth_x_Grounded", "Fab_x_Grounded", "Auth_x_Llama", "Fab_x_Llama",
          "Grounded_x_Llama", "Auth_x_Grounded_x_Llama", "Fab_x_Grounded_x_Llama"]

    df_m1, b1, V1, conv1, nit1 = fit_logit_clustered(X1, y, cluster_ids, v1, "Model 1 (3-way)")
    print("\n" + "=" * 95)
    print("MODEL 1: Fully Interacted Logit (condition * grounded * family) | Clustered SEs by item_id")
    print("=" * 95)
    print(df_m1.round(4).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 2 (PRIMARY for RQ3): Two-way interacted logit
    # Use for: headline RQ3 test (Auth x Grounded), cross-family Fab divergence.
    # ══════════════════════════════════════════════════════════════════════
    X2 = np.column_stack([
        np.ones(len(df_init)), is_auth, is_fab, is_grounded, is_llama,
        is_auth * is_grounded, is_fab * is_grounded, is_auth * is_llama, is_fab * is_llama,
        is_grounded * is_llama
    ])
    v2 = ["Intercept", "Authority", "Fab_Citation", "Grounded", "Family_Llama",
          "Auth_x_Grounded", "Fab_x_Grounded", "Auth_x_Llama", "Fab_x_Llama", "Grounded_x_Llama"]

    df_m2, b2, V2, conv2, nit2 = fit_logit_clustered(X2, y, cluster_ids, v2, "Model 2 (2-way)")
    print("\n" + "=" * 95)
    print("MODEL 2 [PRIMARY]: Two-Way Interacted Logit | Clustered SEs by item_id")
    print("=" * 95)
    print(df_m2.round(4).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════
    # MODEL 3 (Supplementary): Ungrounded arm only
    # Use for: ungrounded-only cross-family contrasts.
    # ══════════════════════════════════════════════════════════════════════
    df_ungrounded = df_init[df_init["grounded"] == False].copy()
    is_auth_u = (df_ungrounded["condition"] == "authority").astype(float).values
    is_fab_u = (df_ungrounded["condition"] == "fabricated_citation").astype(float).values
    is_llama_u = (df_ungrounded["family"] == "llama").astype(float).values
    cluster_u = df_ungrounded["item_id"].values
    y_u = df_ungrounded["c2i"].values

    X3 = np.column_stack([
        np.ones(len(df_ungrounded)), is_auth_u, is_fab_u, is_llama_u,
        is_auth_u * is_llama_u, is_fab_u * is_llama_u
    ])
    v3 = ["Intercept", "Authority", "Fab_Citation", "Family_Llama", "Auth_x_Llama", "Fab_x_Llama"]

    df_m3, b3, V3, conv3, nit3 = fit_logit_clustered(X3, y_u, cluster_u, v3, "Model 3 (ungrounded)")
    print("\n" + "=" * 95)
    print("MODEL 3: Ungrounded Subsample Logit (condition * family) | Clustered SEs by item_id")
    print("=" * 95)
    print(df_m3.round(4).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════
    # DERIVED NET EFFECTS (from Model 1 coefficients)
    # ══════════════════════════════════════════════════════════════════════
    b_map = dict(zip(v1, b1))
    derived_rows = [
        {"Subgroup": "GPT-OSS Ungrounded Auth vs Ctrl",
         "Formula": "β_Auth",
         "Net_LogOdds": b_map["Authority"],
         "Net_OR": np.exp(b_map["Authority"])},
        {"Subgroup": "GPT-OSS Grounded Auth vs Ctrl",
         "Formula": "β_Auth + β_Auth×Gnd",
         "Net_LogOdds": b_map["Authority"] + b_map["Auth_x_Grounded"],
         "Net_OR": np.exp(b_map["Authority"] + b_map["Auth_x_Grounded"])},
        {"Subgroup": "Llama Ungrounded Auth vs Ctrl",
         "Formula": "β_Auth + β_Auth×Llama",
         "Net_LogOdds": b_map["Authority"] + b_map["Auth_x_Llama"],
         "Net_OR": np.exp(b_map["Authority"] + b_map["Auth_x_Llama"])},
        {"Subgroup": "Llama Grounded Auth vs Ctrl",
         "Formula": "β_Auth + β_Auth×Gnd + β_Auth×Llama + β_Auth×Gnd×Llama",
         "Net_LogOdds": (b_map["Authority"] + b_map["Auth_x_Grounded"]
                         + b_map["Auth_x_Llama"] + b_map["Auth_x_Grounded_x_Llama"]),
         "Net_OR": np.exp(b_map["Authority"] + b_map["Auth_x_Grounded"]
                          + b_map["Auth_x_Llama"] + b_map["Auth_x_Grounded_x_Llama"])},
        {"Subgroup": "GPT-OSS Ungrounded Fab vs Ctrl",
         "Formula": "β_Fab",
         "Net_LogOdds": b_map["Fab_Citation"],
         "Net_OR": np.exp(b_map["Fab_Citation"])},
        {"Subgroup": "Llama Ungrounded Fab vs Ctrl",
         "Formula": "β_Fab + β_Fab×Llama",
         "Net_LogOdds": b_map["Fab_Citation"] + b_map["Fab_x_Llama"],
         "Net_OR": np.exp(b_map["Fab_Citation"] + b_map["Fab_x_Llama"])},
    ]
    df_derived = pd.DataFrame(derived_rows)
    print("\n" + "=" * 95)
    print("DERIVED NET EFFECTS (from Model 1):")
    print("=" * 95)
    print(df_derived.round(4).to_string(index=False))

    # ── Data-quality summary for paper ──
    print("\n" + "=" * 95)
    print("DATA-QUALITY ACCOUNTING (for Methodology / Limitations)")
    print("=" * 95)
    print(f"  Total trial JSONs loaded:                    {n_total}")
    print(f"  Dropped (status != 'ok'):                    {n_bad_status}")
    print(f"  Dropped (null initial_judgment/correct):     {n_null_init}")
    print(f"  Initially-correct trials (pre final filter): {len(df_init) + n_null_final}")
    print(f"  Dropped (null final_judgment, unparsed):     {n_null_final}")
    print(f"  Final analysis sample N:                     {len(df_init)}")

    print(f"  Item clusters G:                             {n_clusters}")
    print(f"  Model 1 converged: {conv1} ({nit1} iters)")
    print(f"  Model 2 converged: {conv2} ({nit2} iters)")
    print(f"  Model 3 converged: {conv3} ({nit3} iters)")

    # ── Save CSVs ──
    os.makedirs(args.output_dir, exist_ok=True)
    df_m1.to_csv(os.path.join(args.output_dir, "logistic_regression_model1_3way_interacted.csv"), index=False)
    df_m2.to_csv(os.path.join(args.output_dir, "logistic_regression_model2_2way_interacted.csv"), index=False)
    df_m3.to_csv(os.path.join(args.output_dir, "logistic_regression_model3_ungrounded.csv"), index=False)
    df_derived.to_csv(os.path.join(args.output_dir, "logistic_regression_derived_net_effects.csv"), index=False)

    # Also save a data-quality summary CSV
    dq_rows = [
        {"metric": "total_trials_loaded", "value": n_total},
        {"metric": "dropped_bad_status", "value": n_bad_status},
        {"metric": "dropped_null_initial", "value": n_null_init},
        {"metric": "initially_correct_pre_final_filter", "value": len(df_init) + n_null_final},
        {"metric": "dropped_null_final_judgment", "value": n_null_final},
        {"metric": "final_analysis_N", "value": len(df_init)},
        {"metric": "item_clusters_G", "value": n_clusters},
        {"metric": "model1_converged", "value": conv1},
        {"metric": "model1_iterations", "value": nit1},
        {"metric": "model2_converged", "value": conv2},
        {"metric": "model2_iterations", "value": nit2},
        {"metric": "model3_converged", "value": conv3},
        {"metric": "model3_iterations", "value": nit3},
    ]
    pd.DataFrame(dq_rows).to_csv(os.path.join(args.output_dir, "logistic_regression_data_quality.csv"), index=False)

    print(f"\n[SUCCESS] Exported all CSVs to {args.output_dir}/:")
    print("  - logistic_regression_model1_3way_interacted.csv")
    print("  - logistic_regression_model2_2way_interacted.csv")
    print("  - logistic_regression_model3_ungrounded.csv")
    print("  - logistic_regression_derived_net_effects.csv")
    print("  - logistic_regression_cell_diagnostics.csv")
    print("  - logistic_regression_data_quality.csv")


if __name__ == "__main__":
    main()
