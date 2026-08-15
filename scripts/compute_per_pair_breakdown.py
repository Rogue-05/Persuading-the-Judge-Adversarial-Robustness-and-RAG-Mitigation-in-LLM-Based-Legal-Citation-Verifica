#!/usr/bin/env python3
"""
Compute per-verifier/adversary-pair, per-grounding-arm true_ASR breakdown.
This gives us the table the mentor wants: no pooling across size, family, or grounding arm.
"""
import os
import sys
import pandas as pd

def find_file(rel_path):
    for prefix in [".", "..", "results"]:
        p = os.path.join(prefix, rel_path)
        if os.path.exists(p):
            return p
    return rel_path

# Load all trial-level data
llama_path = find_file('llama_within_family/results/summary/llama_trial_level_for_plots.csv')
gptoss_path = find_file('gptoss_within_family/summary/gptoss_trial_level_for_plots.csv')
cross_path = find_file('crossfamily_llama_gptoss/summary/crossfamily_trial_level_for_plots.csv')

llama = pd.read_csv(llama_path)
gptoss = pd.read_csv(gptoss_path)
cross = pd.read_csv(cross_path)

all_data = pd.concat([llama, gptoss, cross], ignore_index=True)

# Filter to status=ok and initially correct trials
ok = all_data[all_data['status'] == 'ok'].copy() if 'status' in all_data.columns else all_data.copy()
init_correct = ok[ok['initial_correct'] == True].copy()

# For each (verifier_model, adversary_model, grounded, condition), compute true_ASR
groups = init_correct.groupby(['verifier_model', 'adversary_model', 'grounded', 'condition'])

rows = []
for (v, a, g, c), grp in groups:
    n_init = len(grp)
    n_c2i = (grp['flip_direction'] == 'correct_to_incorrect').sum()
    true_asr = n_c2i / n_init if n_init > 0 else 0
    
    # Shorten model names
    v_short = v.split('/')[-1].replace('llama-3.1-8b-instruct', 'Llama 8B').replace('llama-3.3-70b-instruct', 'Llama 70B').replace('gpt-oss-120b', 'GPT-OSS 120B').replace('gpt-oss-20b', 'GPT-OSS 20B')
    a_short = a.split('/')[-1].replace('llama-3.1-8b-instruct', 'Llama 8B').replace('llama-3.3-70b-instruct', 'Llama 70B').replace('gpt-oss-120b', 'GPT-OSS 120B').replace('gpt-oss-20b', 'GPT-OSS 20B')
    
    rows.append({
        'verifier': v_short,
        'adversary': a_short,
        'grounded': 'Grounded' if g else 'Ungrounded',
        'condition': c,
        'n_init_correct': n_init,
        'n_c2i': n_c2i,
        'true_ASR': round(true_asr, 3)
    })

df = pd.DataFrame(rows)

# Pivot to get the mentor's table format
pivot = df.pivot_table(
    index=['verifier', 'adversary', 'grounded'],
    columns='condition',
    values='true_ASR'
).reset_index()

# Compute net effects
pivot['net_auth_ctrl'] = (pivot['authority'] - pivot['control']).round(3)
pivot['net_fab_ctrl'] = (pivot['fabricated_citation'] - pivot['control']).round(3)

# Also get N for each cell
n_pivot = df.pivot_table(
    index=['verifier', 'adversary', 'grounded'],
    columns='condition',
    values='n_init_correct'
).reset_index()

pivot['n_ctrl'] = n_pivot['control']
pivot['n_auth'] = n_pivot['authority']
pivot['n_fab'] = n_pivot['fabricated_citation']

# Sort nicely
sort_order = {'Llama 70B': 0, 'Llama 8B': 1, 'GPT-OSS 120B': 2, 'GPT-OSS 20B': 3}
pivot['sort_v'] = pivot['verifier'].map(sort_order)
pivot['sort_a'] = pivot['adversary'].map(sort_order)
pivot['sort_g'] = pivot['grounded'].map({'Ungrounded': 0, 'Grounded': 1})
pivot = pivot.sort_values(['sort_v', 'sort_a', 'sort_g']).drop(columns=['sort_v', 'sort_a', 'sort_g'])

print("=" * 120)
print("FULL PER-PAIR × PER-ARM BREAKDOWN (no pooling)")
print("=" * 120)
print(pivot[['verifier', 'adversary', 'grounded', 
             'control', 'authority', 'net_auth_ctrl', 
             'fabricated_citation', 'net_fab_ctrl',
             'n_ctrl', 'n_auth', 'n_fab']].to_string(index=False))

print("\n")
print("=" * 120)
print("MENTOR'S MAIN TABLE FORMAT (Ungrounded only — fair comparison with all 3 conditions)")
print("=" * 120)
ungr = pivot[pivot['grounded'] == 'Ungrounded']
print(f"{'Verifier':<16} {'Adversary':<16} {'Ctrl ASR':>10} {'Auth ASR':>10} {'Net(A-C)':>10} {'Fab ASR':>10} {'Net(F-C)':>10} {'N_ctrl':>7} {'N_auth':>7} {'N_fab':>7}")
print("-" * 114)
for _, r in ungr.iterrows():
    print(f"{r['verifier']:<16} {r['adversary']:<16} {r['control']:>9.1%} {r['authority']:>9.1%} {r['net_auth_ctrl']:>+9.1%} {r['fabricated_citation']:>9.1%} {r['net_fab_ctrl']:>+9.1%} {int(r['n_ctrl']):>7} {int(r['n_auth']):>7} {int(r['n_fab']):>7}")

print("\n")
print("=" * 120)
print("GROUNDED ARM (control vs authority only — fabricated excluded by design)")
print("=" * 120)
gr = pivot[pivot['grounded'] == 'Grounded']
print(f"{'Verifier':<16} {'Adversary':<16} {'Ctrl ASR':>10} {'Auth ASR':>10} {'Net(A-C)':>10} {'N_ctrl':>7} {'N_auth':>7}")
print("-" * 78)
for _, r in gr.iterrows():
    fab_val = r.get('fabricated_citation', None)
    print(f"{r['verifier']:<16} {r['adversary']:<16} {r['control']:>9.1%} {r['authority']:>9.1%} {r['net_auth_ctrl']:>+9.1%} {int(r['n_ctrl']):>7} {int(r['n_auth']):>7}")

# Save full breakdown
out_dir = find_file('regression_tables')
if not os.path.exists(out_dir):
    out_dir = 'results/regression_tables'
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, 'per_pair_per_arm_breakdown.csv')
pivot.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
