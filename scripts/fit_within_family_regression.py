#!/usr/bin/env python3
"""
Within-Family Logistic Regressions (replacing the pooled family regression)
===========================================================================
Fits separate logistic regressions for each model family:
  - GPT-OSS: c2i ~ condition * grounded  (item-clustered SEs)
  - Llama:   c2i ~ condition * grounded  (item-clustered SEs)

This eliminates the need for Family interaction terms and avoids pooling
across model families with very different baseline instability patterns.

Usage:
  python fit_within_family_regression.py
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import norm


def detect_family(model_id):
    if "gpt" in str(model_id).lower(): return "gpt_oss"
    elif "llama" in str(model_id).lower(): return "llama"
    return "unknown"


def load_results(results_dir):
    results = []
    if not os.path.exists(results_dir): return results
    for fn in sorted(os.listdir(results_dir)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(results_dir, fn)) as f:
                    results.append(json.load(f))
            except: pass
    return results


def result_to_row(r, grounded_override=None):
    vm = r.get("verifier_model", "")
    return {
        "item_id": r.get("item_id"),
        "category": r.get("category"),
        "condition": r.get("condition"),
        "family": r.get("family") or detect_family(vm),
        "verifier_model": vm,
        "adversary_model": r.get("adversary_model", ""),
        "grounded": grounded_override if grounded_override is not None else r.get("grounded", False),
        "initial_correct": r.get("initial_correct"),
        "final_judgment": r.get("final_judgment"),
        "flip_direction": r.get("flip_direction"),
        "status": r.get("status", "ok"),
    }


def fit_logit_clustered(X, y, cluster_ids, var_names, model_label=""):
    """Newton-Raphson logistic regression with cluster-robust SEs (CR1)."""
    N, K = X.shape
    beta = np.zeros(K)
    max_iter = 200; tol = 1e-8; converged = False; n_iter = 0

    def log_lik(b):
        xb = X @ b
        return np.sum(y * xb - np.logaddexp(0, xb))

    for it in range(max_iter):
        n_iter = it + 1
        p = np.clip(1.0 / (1.0 + np.exp(-X @ beta)), 1e-12, 1 - 1e-12)
        grad = X.T @ (y - p)
        W = p * (1.0 - p)
        H = -X.T @ (W[:, None] * X)
        try:
            delta = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            print(f"  [WARNING] Singular Hessian at iter {n_iter}")
            break
        step = 1.0
        ll_cur = log_lik(beta)
        for _ in range(20):
            if log_lik(beta + step * delta) >= ll_cur - 1e-10: break
            step *= 0.5
        beta += step * delta
        if np.max(np.abs(step * delta)) < tol:
            converged = True; break

    print(f"  [{'CONVERGED' if converged else 'DID NOT CONVERGE'}] {model_label}: {n_iter} iters")

    p = np.clip(1.0 / (1.0 + np.exp(-X @ beta)), 1e-12, 1 - 1e-12)
    W = p * (1.0 - p)
    Hessian = X.T @ (W[:, None] * X)
    Bread = np.linalg.inv(Hessian)

    scores = (y - p)[:, None] * X
    unique_cl = np.unique(cluster_ids)
    G = len(unique_cl)
    Meat = np.zeros((K, K))
    for g in unique_cl:
        u = scores[cluster_ids == g].sum(axis=0)
        Meat += np.outer(u, u)

    df_c = (G / (G - 1)) * ((N - 1) / (N - K))
    V = df_c * (Bread @ Meat @ Bread)

    se = np.sqrt(np.diag(V))
    z = beta / se
    pv = 2 * (1 - norm.cdf(np.abs(z)))

    return pd.DataFrame({
        'Predictor': var_names,
        'Coef': beta,
        'Cluster_SE': se,
        'z_stat': z,
        'p_value': pv,
        'OR': np.exp(beta),
        'OR_95CI_Lower': np.exp(beta - 1.96 * se),
        'OR_95CI_Upper': np.exp(beta + 1.96 * se),
    }), converged, G


def find_dir(rel_path):
    for prefix in ["results", ".", ".."]:
        p = os.path.join(prefix, rel_path)
        if os.path.isdir(p):
            return p
    return rel_path


def main():
    parser = argparse.ArgumentParser()
    default_out = find_dir("regression_tables")
    if not os.path.isdir(default_out):
        default_out = "results/regression_tables"
    parser.add_argument("--output-dir", default=default_out)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all trials
    rows = []
    for r in load_results(find_dir("gptoss_within_family/main")):
        rows.append(result_to_row(r, grounded_override=False))
    for r in load_results(find_dir("gptoss_within_family/mitigation")):
        rows.append(result_to_row(r, grounded_override=True))
    for r in load_results(find_dir("llama_within_family/results/main")):
        rows.append(result_to_row(r, grounded_override=False))
    for r in load_results(find_dir("llama_within_family/results/mitigation")):
        rows.append(result_to_row(r, grounded_override=True))

    df = pd.DataFrame(rows)
    df = df[df["status"] == "ok"].copy()
    df = df[df["initial_correct"] == True].copy()
    df = df[df["final_judgment"].notna()].copy()
    df["c2i"] = (df["flip_direction"] == "correct_to_incorrect").astype(int)

    print(f"Total analysis sample: N={len(df)}")

    verifier_configs = [
        ("GPT-OSS 20B", "openai/gpt-oss-20b", "gpt_oss_20b"),
        ("GPT-OSS 120B", "openai/gpt-oss-120b", "gpt_oss_120b"),
        ("Llama 8B", "meta-llama/llama-3.1-8b-instruct", "llama_8b"),
        ("Llama 70B", "meta-llama/llama-3.3-70b-instruct", "llama_70b")
    ]

    for model_name, model_id, model_code in verifier_configs:
        fam = df[df["verifier_model"] == model_id].copy()
        N_fam = len(fam)
        G_fam = fam["item_id"].nunique()
        print(f"\n{'='*80}")
        print(f"{model_name}: N={N_fam}, G={G_fam} item clusters")
        print(f"{'='*80}")

        # Cell diagnostics
        cell_diag = fam.groupby(["grounded", "condition"]).agg(
            n=("c2i", "count"), n_c2i=("c2i", "sum"), c2i_rate=("c2i", "mean")
        ).round(4)
        print(f"\nPer-cell diagnostics:")
        print(cell_diag.to_string())

        # Build design matrix: c2i ~ condition * grounded
        is_auth = (fam["condition"] == "authority").astype(float).values
        is_fab = (fam["condition"] == "fabricated_citation").astype(float).values
        is_gr = (fam["grounded"] == True).astype(float).values
        cluster = fam["item_id"].values
        y = fam["c2i"].values

        X = np.column_stack([
            np.ones(N_fam), is_auth, is_fab, is_gr,
            is_auth * is_gr, is_fab * is_gr
        ])
        vnames = ["Intercept", "Authority", "Fab_Citation", "Grounded",
                   "Auth_x_Grounded", "Fab_x_Grounded"]

        res_df, conv, G_out = fit_logit_clustered(X, y, cluster, vnames,
                                                   f"{model_name} (condition*grounded)")

        print(f"\n{model_name} Within-Model Regression (condition * grounded)")
        print(f"N={N_fam}, G={G_out} clusters, Converged={conv}")
        print(res_df.round(4).to_string(index=False))

        # Save
        outfile = os.path.join(args.output_dir, f"logistic_regression_{model_code}.csv")
        res_df.to_csv(outfile, index=False)
        print(f"Saved: {outfile}")

        # Save cell diagnostics
        cell_diag.to_csv(os.path.join(args.output_dir, f"cell_diagnostics_{model_code}.csv"))

    print("\n[SUCCESS] Within-family regressions complete.")


if __name__ == "__main__":
    main()
