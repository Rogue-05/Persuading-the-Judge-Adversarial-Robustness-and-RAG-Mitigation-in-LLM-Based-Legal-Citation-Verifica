#!/usr/bin/env python3
import json, os
import pandas as pd

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
        "condition": r.get("condition"),
        "family": r.get("family"),
        "verifier_model": vm,
        "adversary_model": r.get("adversary_model", ""),
        "grounded": grounded_override if grounded_override is not None else r.get("grounded", False),
        "initial_correct": r.get("initial_correct"),
        "final_judgment": r.get("final_judgment"),
        "flip_direction": r.get("flip_direction"),
        "status": r.get("status", "ok"),
        "initial_confidence": r.get("initial_confidence"),
        "final_confidence": r.get("final_confidence"),
    }

def find_dir(rel_path):
    for prefix in ["results", ".", ".."]:
        p = os.path.join(prefix, rel_path)
        if os.path.isdir(p):
            return p
    return rel_path

def main():
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
    df = df[df["status"] == "ok"]
    df = df[df["initial_correct"] == True]
    df = df[df["final_judgment"].notna()]
    df = df[df["flip_direction"] == "correct_to_incorrect"]
    
    # We are looking at ungrounded flips for the specific transcript analysis 
    # (since law_001 example says "ungrounded")
    df_ungr = df[df["grounded"] == False].copy()
    
    # Let's count for GPT-OSS 20B verifier under authority
    gpt_auth = df_ungr[(df_ungr["family"] == "gpt_oss") & (df_ungr["condition"] == "authority") & (df_ungr["verifier_model"] == "openai/gpt-oss-20b")]
    gpt_auth_inc = gpt_auth[gpt_auth["final_confidence"] > gpt_auth["initial_confidence"]]
    print(f"GPT-OSS 20B ungrounded authority flips: n={len(gpt_auth)}")
    print(f"  -> Flips with confidence INCREASE: {len(gpt_auth_inc)}/{len(gpt_auth)} ({len(gpt_auth_inc)/len(gpt_auth):.1%})")
    
    # Let's check Llama 70B verifier under authority
    llama_auth = df_ungr[(df_ungr["family"] == "llama") & (df_ungr["condition"] == "authority") & (df_ungr["verifier_model"] == "meta-llama/llama-3.3-70b-instruct")]
    llama_auth_inc = llama_auth[llama_auth["final_confidence"] > llama_auth["initial_confidence"]]
    print(f"Llama 70B ungrounded authority flips: n={len(llama_auth)}")
    print(f"  -> Flips with confidence INCREASE: {len(llama_auth_inc)}/{len(llama_auth)} ({len(llama_auth_inc)/len(llama_auth):.1%})")
    
if __name__ == "__main__":
    main()
