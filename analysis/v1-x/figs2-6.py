# make_26_figures.py -- figures for Section 2.6
import json, csv
import numpy as np
import matplotlib.pyplot as plt

UP = "/mnt/agents/upload"  # change to your project root

fed = json.load(open(f"results/fleet_expected_detected.json"))
po = fed["per_object"]
inc = np.array([o["inclination_deg"] for o in po])
alt = np.array([o["altitude_km"] for o in po])
ee = np.array([o["epsilon_effective"] for o in po]) * 100  # percent
od = np.array([o["ordem_detected"] for o in po])
md = np.array([o["master_detected"] for o in po])

print(f"epsilon_effective (%): min {ee.min():.3f}  med {np.median(ee):.3f}  max {ee.max():.3f}")

# Fig 1 (2.6.1): per-object effective efficiency vs inclination, colored by altitude
fig, ax = plt.subplots(figsize=(7, 4))
sc = ax.scatter(inc, ee, c=alt, s=8, cmap="viridis")
ax.set_yscale("log")
ax.set_xlabel("Inclination (deg)")
ax.set_ylabel(r"Effective efficiency $\epsilon_{\rm eff}$ (%)")
cb = fig.colorbar(sc); cb.set_label("Altitude (km)")
ax.set_title("Per-object effective efficiency, synthetic Gate 3 curve (n=3018)")
fig.tight_layout(); fig.savefig("figs/fig_selection_efficiency.pdf")

# Fig 2 (2.6.3): concentration of predicted detections (ORDEM vs MASTER)
s_ordem = np.sort(od)[::-1]
cum_ordem = np.cumsum(s_ordem) / s_ordem.sum()

s_master = np.sort(md)[::-1]
cum_master = np.cumsum(s_master) / s_master.sum()

x_pct = np.arange(1, len(po) + 1) / len(po) * 100

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(x_pct, cum_ordem * 100, lw=2, label="ORDEM", color="C0")
ax.plot(x_pct, cum_master * 100, lw=2, linestyle="--", label="MASTER", color="C1")

ax.set_xlabel("Top fraction of objects (%)")
ax.set_ylabel("Cumulative share of detected events (%)")
ax.set_title("Concentration of predicted detections")
ax.legend(frameon=True)
ax.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout(); fig.savefig("figs/fig_detected_concentration.pdf")

# Fig 3 (2.6.3): ORDEM/MASTER flux ratio vs altitude from the actual grid used
rows = list(csv.DictReader(open(f"results/flux_comparison.csv")))
fig, ax = plt.subplots(figsize=(6, 4))
for iv in sorted(set(r["inclination_deg"] for r in rows), key=float):
    sel = sorted([r for r in rows if r["inclination_deg"] == iv],
                 key=lambda r: float(r["altitude_km"]))
    ax.plot([float(r["altitude_km"]) for r in sel],
            [float(r["ratio_ordem_to_master"]) for r in sel],
            "o-", label=f"i={iv}")
ax.set_xlabel("Altitude (km)")
ax.set_ylabel("ORDEM/MASTER flux ratio (1 mm-1 cm)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig_flux_ratio.pdf")
print("done")