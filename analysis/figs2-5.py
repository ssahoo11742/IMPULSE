import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.makedirs("figs", exist_ok=True)

# Set clean aesthetic style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    }
)

with open("results/gate2_confounders.json") as f:
    conf = json.load(f)
with open("results/gate2_all.json") as f:
    g2 = json.load(f)
with open("results/scrambled_null.json") as f:
    scr = json.load(f)

# ---- Shared data preparation ----
names, rates, errs = [], [], []
for k, s in conf["summary"].items():
    n, p = s["n"], s["fp_rate"]
    names.append(k)
    rates.append(100 * p)  # percent
    se = 100 * np.sqrt(p * (1 - p) / n) if p > 0 else 0
    errs.append(se)

order = ["baseline", "f107_flat", "kp_quiet", "env_flat", "cd_nominal"]
display_labels_base = [
    "Baseline",
    "$F_{10.7}$ Flat",
    "$K_p$ Quiet",
    "Env. Flat",
    "$C_d$ Nominal",
]
idx = [names.index(o) for o in order]

selected_rates = [rates[i] for i in idx]
selected_errs = [errs[i] for i in idx]

# Correct scrambled FP rate from the summary
scr_rate = 100 * scr["summary"]["triggers"] / scr["config"]["total_valid_windows"]
scr_se = 100 * np.sqrt((scr_rate / 100) * (1 - scr_rate / 100) / scr["config"]["total_valid_windows"])

# ============================================================
# Fig A1: Original Confounder Ablation (NO scrambled bar)
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

bars = ax.bar(
    range(len(order)),
    selected_rates,
    yerr=selected_errs,
    capsize=5,
    color="#3470a3",
    edgecolor="#1f4260",
    linewidth=0.8,
    alpha=0.85,
    error_kw={"ecolor": "#222222", "linewidth": 1.2},
    width=0.55,
)

ax.set_yscale("log")
ax.set_ylim(1e-3, 1.0)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(display_labels_base)

ax.axhline(
    0.1,
    color="#d9381e",
    linestyle="--",
    linewidth=1.2,
    label="Nominal 0.1% (99.9th pct)",
)

ax.set_ylabel("False-Positive Rate per Window (%)")
ax.set_title(
    "Confounder Ablation (100-Object Subset, Threshold 13.258)", pad=12
)
ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
sns.despine(top=True, right=True)

fig.tight_layout()
fig.savefig("figs/fig_confounder_ablation.png", dpi=300)
fig.savefig("figs/fig_confounder_ablation.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Fig A2: Confounder Ablation + Scrambled Real-TLE (with labels on ALL bars)
# ============================================================
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)

display_labels_full = display_labels_base + ["Scrambled\nreal-TLE"]
full_rates = selected_rates + [scr_rate]
full_errs = selected_errs + [scr_se]

colors = ["#3470a3"] * 5 + ["#e66101"]
edge_colors = ["#1f4260"] * 5 + ["#a04000"]

bars = ax.bar(
    range(len(display_labels_full)),
    full_rates,
    yerr=full_errs,
    capsize=4,
    color=colors,
    edgecolor=edge_colors,
    linewidth=0.8,
    alpha=0.85,
    error_kw={"ecolor": "#222222", "linewidth": 1.1},
    width=0.62,
)

ax.set_yscale("log")
ax.set_ylim(1e-3, 50)
ax.set_xticks(range(len(display_labels_full)))
ax.set_xticklabels(display_labels_full)

ax.axhline(
    0.1,
    color="#d9381e",
    linestyle="--",
    linewidth=1.2,
    label="Nominal 0.1% (99.9th pct)",
)

# ---- Add percentage labels on ALL bars ----
# ---- Add percentage labels on ALL bars ----
for i, rate in enumerate(full_rates):
    if rate < 1:
        label = f"{rate:.2f}%"
        offset = 15
        fontsize = 8
        color_label = "#1f4260"
    else:
        label = f"{rate:.1f}%"
        offset = 8
        fontsize = 9
        color_label = "#a04000"
    
    ax.annotate(
        label,
        xy=(i, rate),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color_label,
    )
ax.set_ylabel("False-Positive Rate per Window (%)")
ax.set_title(
    "False-Positive Rate by Source\n(100-Object Ablations + Scrambled Real-TLE Null)",
    pad=12,
)
ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper left")
sns.despine(top=True, right=True)

fig.tight_layout()
fig.savefig("figs/fig_confounder_ablation_with_scrambled.png", dpi=300)
fig.savefig("figs/fig_confounder_ablation_with_scrambled.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================
# Fig B: Combined Synthetic Null + Scrambled Tail (unchanged)
# ============================================================
peaks_all = [
    p for r in g2["results"] for p in r["confounder_peaks"]["all"] if p is not None
]
peaks_scr = [p for po in scr["per_object"] for p in po["peaks"] if p is not None]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

ax.hist(
    peaks_all,
    bins=120,
    density=True,
    histtype="stepfilled",
    facecolor="#3470a3",
    edgecolor="#1f4260",
    alpha=0.4,
    linewidth=1.2,
    label=f"Combined Synthetic ($n={len(peaks_all):,}$)",
)

ax.hist(
    peaks_scr,
    bins=120,
    density=True,
    histtype="step",
    color="#e66101",
    linewidth=1.5,
    label=f"Scrambled Empirical ($n={len(peaks_scr):,}$)",
)

ax.axvline(
    13.258,
    color="#d9381e",
    linestyle="--",
    linewidth=1.2,
    label="Threshold 13.258",
)

ax.set_xlim(0, 60)
ax.set_xlabel("Window-Maximum Mahalanobis Distance")
ax.set_ylabel("Density")
ax.set_title("Null Distributions of Peak Mahalanobis Distance", pad=12)

ax.legend(
    frameon=True,
    facecolor="white",
    edgecolor="#e0e0e0",
    loc="upper right",
)

axin = ax.inset_axes([0.52, 0.22, 0.44, 0.42])

axin.hist(
    peaks_all,
    bins=np.logspace(0, 3, 100),
    density=True,
    histtype="step",
    color="#3470a3",
    linewidth=1.2,
)
axin.hist(
    peaks_scr,
    bins=np.logspace(0, 3, 100),
    density=True,
    histtype="step",
    color="#e66101",
    linewidth=1.2,
)
axin.axvline(13.258, color="#d9381e", linestyle="--", linewidth=1.0)

axin.set_xscale("log")
axin.set_yscale("log")
axin.set_xlim(1, 1000)
axin.set_ylim(1e-6, 1)
axin.set_title("Tail (Log-Log Scale)", fontsize=8.5, pad=4)
axin.tick_params(axis="both", labelsize=7.5)
axin.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

sns.despine(top=True, right=True)

fig.tight_layout()
fig.savefig("figs/fig_null_distribution.png", dpi=300)
fig.savefig("figs/fig_null_distribution.pdf", bbox_inches="tight")
plt.close(fig)