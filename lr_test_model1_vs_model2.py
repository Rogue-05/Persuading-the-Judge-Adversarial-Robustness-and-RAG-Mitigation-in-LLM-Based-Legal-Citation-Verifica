#!/usr/bin/env python3
"""
Likelihood-Ratio Test: Model 1 (3-way interacted) vs Model 2 (2-way interacted)
================================================================================
Tests whether the two 3-way interaction terms that Model 1 adds on top of
Model 2 (Auth_x_Grounded_x_Llama, Fab_x_Grounded_x_Llama) meaningfully improve
fit, or whether Model 2's simpler, pooled-across-family Auth_x_Grounded effect
is an adequate description of the data.

Model 2 is nested inside Model 1 (Model 2 = Model 1 with the two 3-way
coefficients constrained to exactly 0), so a standard LRT applies:

    LR statistic = 2 * (loglik_Model1 - loglik_Model2)
    LR statistic ~ chi-squared(df = 2)   under H0: Model 2 is adequate

IMPORTANT CAVEAT: with cluster-robust (sandwich) standard errors, the
ordinary chi-squared reference distribution for the LRT is not strictly
justified — clustering can inflate the effective likelihood-ratio statistic
beyond its nominal chi-squared distribution, similar to how it inflates
naive (non-robust) Wald test statistics. This script reports:
  (a) the naive/nominal chi-squared LRT p-value (informative but optimistic
      about significance if clustering matters a lot), and
  (b) a cluster bootstrap p-value (resample item_id clusters with
      replacement, refit both models B times, build an empirical null
      distribution of the LR statistic) — this is the more defensible number
      to report in the paper.

Usage:
  python lr_test_model1_vs_model2.py --output-dir .
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2


def detect_family(model_id):
    if "gpt" in str(model_id).lower():
        return "gpt_oss"
    elif "llama" in str(model_id).lower():
        return "llama"
    return "unknown"


def load_results(results_dir):
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
    verifier_model = r.get("verifier_model", "")
    return {
        "item_id": r.get("item_id"),
        "condition": r.get("condition"),
        "family": r.get("family") or detect_family(verifier_model),
        "grounded": grounded_override if grounded_override is not None else r.get("grounded", False),
        "initial_correct": r.get("initial_correct"),
        "initial_judgment": r.get("initial_judgment"),
        "final_judgment": r.get("final_judgment"),
        "flip_direction": r.get("flip_direction"),
        "status": r.get("status", "ok"),
    }


def build_dataframe(args):
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
    df = df[df["status"] == "ok"].copy()
    null_init = df["initial_correct"].isna() | df["initial_judgment"].isna()
    df = df[~null_init].copy()

    df_init = df[df["initial_correct"] == True].copy()
    null_final = df_init["final_judgment"].isna()
    df_init = df_init[~null_final].copy()

    df_init["c2i"] = (df_init["flip_direction"] == "correct_to_incorrect").astype(int)
    return df_init


def design_matrices(df_init):
    is_auth = (df_init["condition"] == "authority").astype(float).values
    is_fab = (df_init["condition"] == "fabricated_citation").astype(float).values
    is_grounded = (df_init["grounded"] == True).astype(float).values
    is_llama = (df_init["family"] == "llama").astype(float).values

    X1 = np.column_stack([
        np.ones(len(df_init)), is_auth, is_fab, is_grounded, is_llama,
        is_auth * is_grounded, is_fab * is_grounded, is_auth * is_llama, is_fab * is_llama,
        is_grounded * is_llama, is_auth * is_grounded * is_llama, is_fab * is_grounded * is_llama
    ])
    X2 = np.column_stack([
        np.ones(len(df_init)), is_auth, is_fab, is_grounded, is_llama,
        is_auth * is_grounded, is_fab * is_grounded, is_auth * is_llama, is_fab * is_llama,
        is_grounded * is_llama
    ])
    y = df_init["c2i"].values
    cluster_ids = df_init["item_id"].values
    return X1, X2, y, cluster_ids


def fit_logit_loglik(X, y, max_iter=200, tol=1e-8):
    """Plain (non-clustered) MLE fit via Newton-Raphson w/ step-halving.
    Returns beta and the maximized log-likelihood. Clustering does not
    change the point estimates or the log-likelihood (it only changes SEs),
    so this is valid for both the LRT and for producing Model 1/Model 2
    coefficients."""
    N, K = X.shape
    beta = np.zeros(K)

    def log_lik(b):
        xb = X @ b
        return np.sum(y * xb - np.logaddexp(0, xb))

    step = 1.0
    delta = np.zeros(K)
    for it in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        grad = X.T @ (y - p)
        W = p * (1.0 - p)
        H = -X.T @ (W[:, None] * X)
        try:
            delta = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break
        step = 1.0
        ll_current = log_lik(beta)
        for _ in range(20):
            ll_new = log_lik(beta + step * delta)
            if ll_new >= ll_current - 1e-10:
                break
            step *= 0.5
        beta += step * delta
        if np.max(np.abs(step * delta)) < tol:
            break

    ll_final = log_lik(beta)
    return beta, ll_final


def lr_statistic(X1, X2, y):
    _, ll1 = fit_logit_loglik(X1, y)
    _, ll2 = fit_logit_loglik(X2, y)
    lr_stat = 2.0 * (ll1 - ll2)
    return lr_stat, ll1, ll2


def cluster_bootstrap_pvalue(X1, X2, y, cluster_ids, observed_lr, n_boot=1000, seed=42):
    """
    Cluster (block) bootstrap: resample item_id clusters with replacement,
    rebuild the trial-level dataset from resampled clusters, refit both
    models, record the LR statistic. This respects the same dependence
    structure (repeated trials within item) that the cluster-robust SEs
    are designed for, giving a more defensible null distribution than
    the nominal chi-squared reference.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    # Precompute row indices per cluster for fast resampling
    cluster_to_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    boot_lrs = np.full(n_boot, np.nan)
    n_failed = 0
    for b in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=G, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
        Xb1 = X1[idx]
        Xb2 = X2[idx]
        yb = y[idx]

        # Skip degenerate resamples (all-0 or all-1 outcome -> no information)
        if yb.sum() == 0 or yb.sum() == len(yb):
            n_failed += 1
            continue
        try:
            lr_b, _, _ = lr_statistic(Xb1, Xb2, yb)
            if np.isfinite(lr_b):
                boot_lrs[b] = max(lr_b, 0.0)  # LR stat should be >=0 in nested models; clip tiny negatives from optimizer noise
        except Exception:
            n_failed += 1
            continue

    boot_lrs_valid = boot_lrs[~np.isnan(boot_lrs)]
    n_valid = len(boot_lrs_valid)
    boot_p = np.mean(boot_lrs_valid >= observed_lr) if n_valid > 0 else np.nan
    return boot_p, boot_lrs_valid, n_valid, n_failed


def main():
    parser = argparse.ArgumentParser(description="LR test: Model 1 (3-way) vs Model 2 (2-way, nested).")
    parser.add_argument("--gptoss-main", default="results_gptoss_full_run_rescued/main")
    parser.add_argument("--gptoss-mit", default="results_gptoss_full_run_rescued/mitigation")
    parser.add_argument("--llama-main", default="results_llama_full_run_rescued/results/main")
    parser.add_argument("--llama-mit", default="results_llama_full_run_rescued/results/mitigation")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df_init = build_dataframe(args)
    X1, X2, y, cluster_ids = design_matrices(df_init)
    n_clusters = len(np.unique(cluster_ids))
    print(f"[INFO] N = {len(df_init)}, G (item clusters) = {n_clusters}")

    df_extra = X1.shape[1] - X2.shape[1]  # = 2 (Auth_x_Grounded_x_Llama, Fab_x_Grounded_x_Llama)
    print(f"[INFO] Model 1 has {X1.shape[1]} params, Model 2 has {X2.shape[1]} params, df = {df_extra}")

    # ── Observed LR statistic ──
    lr_obs, ll1, ll2 = lr_statistic(X1, X2, y)
    print(f"\n[OBSERVED] loglik Model 1 = {ll1:.4f}")
    print(f"[OBSERVED] loglik Model 2 = {ll2:.4f}")
    print(f"[OBSERVED] LR statistic   = {lr_obs:.4f}")

    # ── Nominal chi-squared reference (naive; ignores clustering) ──
    p_chi2 = 1 - chi2.cdf(lr_obs, df=df_extra)
    print(f"\n[NAIVE CHI-SQ TEST] LR = {lr_obs:.4f}, df = {df_extra}, p = {p_chi2:.4f}")
    print("  (This treats each trial as an independent observation for the LRT itself.")
    print("   Given repeated trials per item, this p-value is likely anti-conservative")
    print("   -- i.e. it may overstate significance. Use the cluster bootstrap p below")
    print("   as the primary number to report.)")

    # ── Cluster bootstrap reference (respects item-level dependence) ──
    print(f"\n[CLUSTER BOOTSTRAP] Running {args.n_boot} cluster resamples (seed={args.seed})...")
    boot_p, boot_lrs, n_valid, n_failed = cluster_bootstrap_pvalue(
        X1, X2, y, cluster_ids, lr_obs, n_boot=args.n_boot, seed=args.seed
    )
    print(f"[CLUSTER BOOTSTRAP] Valid resamples: {n_valid} / {args.n_boot} (failed/degenerate: {n_failed})")
    if n_valid > 0:
        print(f"[CLUSTER BOOTSTRAP] Bootstrap LR distribution: mean={boot_lrs.mean():.3f}, "
              f"median={np.median(boot_lrs):.3f}, 95th pct={np.percentile(boot_lrs, 95):.3f}")
        print(f"[CLUSTER BOOTSTRAP] p-value (P[LR_boot >= LR_observed]) = {boot_p:.4f}")
    else:
        print("[CLUSTER BOOTSTRAP] WARNING: no valid bootstrap resamples produced.")

    # ── Verdict ──
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    alpha = 0.05
    if n_valid > 0:
        print(f"  Naive chi-sq p     = {p_chi2:.4f}  -> {'REJECT H0 (Model 1 preferred)' if p_chi2 < alpha else 'FAIL TO REJECT (Model 2 adequate)'}")
        print(f"  Cluster bootstrap p = {boot_p:.4f}  -> {'REJECT H0 (Model 1 preferred)' if boot_p < alpha else 'FAIL TO REJECT (Model 2 adequate)'}")
        print()
        if boot_p < alpha:
            print("  => The 3-way family-specific interaction terms earn their keep.")
            print("     The pooled Auth_x_Grounded effect in Model 2 is likely masking")
            print("     real heterogeneity across families. Report Model 1's family-specific")
            print("     breakdown as the primary RQ3 result, or report both with this test")
            print("     as justification for treating GPT-OSS and Llama separately.")
        else:
            print("  => Cannot reject the simpler, pooled model. Model 2's single")
            print("     Auth_x_Grounded coefficient is a statistically adequate summary;")
            print("     Model 1's apparent split (near-zero GPT-OSS term, large Llama-specific")
            print("     term) is consistent with sampling noise given current power.")
    else:
        print("  Bootstrap failed to produce valid resamples -- rely on naive chi-sq with caution,")
        print("  and flag this limitation explicitly if reporting.")

    # ── Save results ──
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "N": len(df_init),
        "G_clusters": n_clusters,
        "df": df_extra,
        "loglik_model1": ll1,
        "loglik_model2": ll2,
        "LR_statistic": lr_obs,
        "naive_chi2_p": p_chi2,
        "cluster_bootstrap_p": boot_p,
        "cluster_bootstrap_n_valid": n_valid,
        "cluster_bootstrap_n_failed": n_failed,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    pd.Series(out).to_csv(os.path.join(args.output_dir, "lr_test_model1_vs_model2.csv"))
    if n_valid > 0:
        pd.Series(boot_lrs, name="LR_bootstrap").to_csv(
            os.path.join(args.output_dir, "lr_test_bootstrap_distribution.csv"), index=False
        )
    print(f"\n[SUCCESS] Results saved to {args.output_dir}/lr_test_model1_vs_model2.csv")


if __name__ == "__main__":
    main()
