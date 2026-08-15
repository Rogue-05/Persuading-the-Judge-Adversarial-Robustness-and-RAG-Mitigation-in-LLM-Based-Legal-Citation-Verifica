#!/usr/bin/env python3
"""
Generate publication-quality figures for the NLLP paper.
Outputs:
  - fig_confidence_profile.pdf  (True ASR vs CW-ASR grouped bar, doubt vs override)
  - fig_grounding_authority.pdf  (Authority true_ASR: ungrounded vs grounded, all 3 setups)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

COLORS = {
    'control': '#94A3B8',       # slate
    'authority': '#DC2626',     # red
    'fabricated': '#2563EB',    # blue
    'cwasr_control': '#CBD5E1',
    'cwasr_authority': '#FCA5A5',
    'cwasr_fabricated': '#93C5FD',
    'ungrounded': '#EF4444',
    'grounded': '#22C55E',
}

outdir = '/Users/rohannrahulshah/College/NLLP'

# ════════════════════════════════════════════════════════════════
# FIGURE 1: Confidence Profile — True ASR vs CW-ASR
# Shows that authority flips are confident; control flips are hesitant
# ════════════════════════════════════════════════════════════════

# Data from cw_por_confidence_weighted_metrics.csv — ungrounded cells only
# (Clearest contrast; grounded in appendix table)
families = ['GPT-OSS', 'Llama', 'Cross-Family']
conditions = ['Control', 'Authority', 'Fab. Cit.']

# true_ASR (ungrounded)
true_asr = {
    'GPT-OSS':       [0.283,  0.585,  0.204],
    'Llama':         [0.754,  0.971,  0.920],
    'Cross-Family':  [0.735,  0.980,  0.818],
}
# CW-ASR (ungrounded)
cw_asr = {
    'GPT-OSS':       [0.193,  0.501,  0.168],
    'Llama':         [0.406,  0.762,  0.749],
    'Cross-Family':  [0.461,  0.904,  0.484],
}
# Mean final confidence on flips (ungrounded)
mean_conf = {
    'GPT-OSS':       [6.80, 8.56, 8.21],
    'Llama':         [5.38, 7.85, 8.14],
    'Cross-Family':  [6.28, 9.23, 5.92],
}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)

bar_w = 0.25
x = np.arange(len(conditions))

true_colors = [COLORS['control'], COLORS['authority'], COLORS['fabricated']]
cw_colors = [COLORS['cwasr_control'], COLORS['cwasr_authority'], COLORS['cwasr_fabricated']]

for idx, (fam, ax) in enumerate(zip(families, axes)):
    true_vals = true_asr[fam]
    cw_vals = cw_asr[fam]
    conf_vals = mean_conf[fam]

    bars1 = ax.bar(x - bar_w/2, true_vals, bar_w, color=true_colors,
                   edgecolor='#1E293B', linewidth=0.6, label='True ASR' if idx == 0 else '')
    bars2 = ax.bar(x + bar_w/2, cw_vals, bar_w, color=cw_colors,
                   edgecolor='#1E293B', linewidth=0.6, label='CW-ASR' if idx == 0 else '')

    # Annotate with mean final confidence
    for j, (b, c) in enumerate(zip(bars1, conf_vals)):
        ax.annotate(f'{c:.1f}', xy=(b.get_x() + b.get_width(), b.get_height() + 0.02),
                    fontsize=6.5, ha='center', va='bottom', color='#475569', fontstyle='italic')

    ax.set_title(fam, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    if idx == 0:
        ax.set_ylabel('Attack Success Rate')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add a single legend for the whole figure
handles = [
    plt.Rectangle((0,0),1,1, facecolor=COLORS['authority'], edgecolor='#1E293B', linewidth=0.6),
    plt.Rectangle((0,0),1,1, facecolor=COLORS['cwasr_authority'], edgecolor='#1E293B', linewidth=0.6),
]
fig.legend(handles, ['True ASR', 'CW-ASR (confidence-weighted)'],
           loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

fig.text(0.5, -0.02, 'Italic values above True ASR bars = mean final confidence (1–10 scale) on flipped verdicts',
         ha='center', fontsize=7, color='#64748B', style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
path1 = f'{outdir}/fig_confidence_profile.pdf'
fig.savefig(path1, bbox_inches='tight', dpi=300)
print(f'Saved: {path1}')
plt.close(fig)


# ════════════════════════════════════════════════════════════════
# FIGURE 2: Authority true_ASR — Ungrounded vs Grounded
# Shows "mitigates but doesn't eliminate" across all 3 setups
# ════════════════════════════════════════════════════════════════

setups = ['GPT-OSS 20B\n(vs 120B)', 'Llama 70B\n(vs 8B)', 'Cross-Family\n(70B vs 120B)']

# Authority true_ASR (unpooled per-pair)
auth_ungr = [0.727, 0.957, 0.980]
auth_gr   = [0.321, 0.644, 0.861]

# Control true_ASR (for reference line)
ctrl_ungr = [0.109, 0.640, 0.706]
ctrl_gr   = [0.115, 0.310, 0.463]

fig, ax = plt.subplots(figsize=(4.5, 3.0))

x = np.arange(len(setups))
bar_w = 0.30

bars_u = ax.bar(x - bar_w/2, auth_ungr, bar_w, color=COLORS['ungrounded'],
                edgecolor='#1E293B', linewidth=0.7, label='Authority (Ungrounded)', alpha=0.85)
bars_g = ax.bar(x + bar_w/2, auth_gr, bar_w, color=COLORS['grounded'],
                edgecolor='#1E293B', linewidth=0.7, label='Authority (Grounded)', alpha=0.85)

# Control baselines as horizontal markers
for i in range(len(setups)):
    ax.plot([i - bar_w/2 - 0.05, i - bar_w/2 + bar_w + 0.05], [ctrl_ungr[i], ctrl_ungr[i]],
            color='#1E293B', linewidth=1.2, linestyle='--', alpha=0.5)
    ax.plot([i + bar_w/2 - 0.05, i + bar_w/2 + bar_w + 0.05], [ctrl_gr[i], ctrl_gr[i]],
            color='#1E293B', linewidth=1.2, linestyle='--', alpha=0.5)

# Annotate deltas
for i in range(len(setups)):
    delta_u = auth_ungr[i] - ctrl_ungr[i]
    delta_g = auth_gr[i] - ctrl_gr[i]
    ax.annotate(f'+{delta_u*100:.0f}pp', xy=(x[i] - bar_w/2, auth_ungr[i] + 0.01),
                fontsize=7, ha='center', va='bottom', fontweight='bold', color='#991B1B')
    ax.annotate(f'+{delta_g*100:.0f}pp', xy=(x[i] + bar_w/2, auth_gr[i] + 0.01),
                fontsize=7, ha='center', va='bottom', fontweight='bold', color='#166534')

ax.set_xticks(x)
ax.set_xticklabels(setups, fontsize=8)
ax.set_ylabel('True ASR (Authority Condition)')
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.lines import Line2D
handles = [
    plt.Rectangle((0,0),1,1, facecolor=COLORS['ungrounded'], edgecolor='#1E293B', alpha=0.85),
    plt.Rectangle((0,0),1,1, facecolor=COLORS['grounded'], edgecolor='#1E293B', alpha=0.85),
    Line2D([0],[0], color='#1E293B', linewidth=1.2, linestyle='--', alpha=0.5),
]
ax.legend(handles, ['Authority (Ungrounded)', 'Authority (Grounded)', 'Control baseline'],
          fontsize=7, loc='upper left', framealpha=0.8)

plt.tight_layout()
path2 = f'{outdir}/fig_grounding_authority.pdf'
fig.savefig(path2, bbox_inches='tight', dpi=300)
print(f'Saved: {path2}')
plt.close(fig)

print('Done. Both figures generated.')
