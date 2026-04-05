"""
Generate publication-quality figures from interactive 2D experiment runs.

Produces:
  figures/val_acc_comparison.pdf   – per-dataset val accuracy curves (CE vs CGA)
  figures/summary_bar.pdf         – final val accuracy bar chart across datasets
  figures/compute_vs_accuracy.pdf – val accuracy vs cumulative compute time
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── Academic style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load runs ─────────────────────────────────────────────────────────────────
RUN_DIR = os.path.join(os.path.dirname(__file__), "runs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

runs = {}
for fp in sorted(glob.glob(os.path.join(RUN_DIR, "run_*.json"))):
    with open(fp) as f:
        r = json.load(f)
    runs[r["run_id"]] = r

# ── Define comparison groups ──────────────────────────────────────────────────
# Each group: (title, CE_run_id, CGA_run_id(s))
COMPARISONS = [
    ("Two Moons",  1,  [2, 3]),
    ("XOR",        8,  [9]),
    ("Circles",   10,  [11]),
    ("Gaussian",  13,  [14]),
]

CE_COLOR  = "#2c3e50"
CGA_COLORS = ["#e74c3c", "#27ae60", "#8e44ad"]
CGA_STYLES = ["-", "--", "-."]


def _ann_count(run):
    """Total number of annotations in a run."""
    return sum(len(a) for a in run.get("annotations", []))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Validation accuracy curves per dataset
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharex=False, sharey=False)
axes = axes.flatten()

for ax, (title, ce_id, cga_ids) in zip(axes, COMPARISONS):
    ce = runs[ce_id]
    ep_ce = np.arange(1, len(ce["val_acc"]) + 1)
    ax.plot(ep_ce, np.array(ce["val_acc"]) * 100,
            color=CE_COLOR, lw=1.8, label="Cross-entropy (control)")

    for i, cga_id in enumerate(cga_ids):
        cga = runs[cga_id]
        ep_cga = np.arange(1, len(cga["val_acc"]) + 1)
        n_ann = _ann_count(cga)
        alpha = cga["alpha"]
        lbl = f"CGA Softplus ($\\alpha$={alpha}, {n_ann} ann.)"
        ax.plot(ep_cga, np.array(cga["val_acc"]) * 100,
                color=CGA_COLORS[i], lw=1.8, linestyle=CGA_STYLES[i],
                label=lbl)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 5))

fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIG_DIR, "val_acc_comparison.pdf"))
print(f"Saved val_acc_comparison.pdf")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Summary bar chart – final validation accuracy
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 3.0))

datasets = []
ce_accs = []
cga_accs = []
cga_labels_all = []

for title, ce_id, cga_ids in COMPARISONS:
    ce = runs[ce_id]
    best_cga = max(cga_ids, key=lambda i: runs[i]["val_acc"][-1])
    cga = runs[best_cga]
    datasets.append(title)
    ce_accs.append(ce["val_acc"][-1] * 100)
    cga_accs.append(cga["val_acc"][-1] * 100)

x = np.arange(len(datasets))
w = 0.32
bars_ce  = ax.bar(x - w/2, ce_accs,  w, color=CE_COLOR,  label="Cross-entropy", edgecolor="white", linewidth=0.5)
bars_cga = ax.bar(x + w/2, cga_accs, w, color="#e74c3c",  label="CGA Softplus",  edgecolor="white", linewidth=0.5)

# Value labels
for bar in list(bars_ce) + list(bars_cga):
    h = bar.get_height()
    ax.annotate(f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Final validation accuracy (%)")
ax.legend(loc="lower right")
ax.set_ylim(60, 105)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "summary_bar.pdf"))
print("Saved summary_bar.pdf")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Validation accuracy vs cumulative training time
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
axes = axes.flatten()

for ax, (title, ce_id, cga_ids) in zip(axes, COMPARISONS):
    ce = runs[ce_id]
    ax.plot(ce["cumulative_s"], np.array(ce["val_acc"]) * 100,
            color=CE_COLOR, lw=1.8, label="Cross-entropy")

    for i, cga_id in enumerate(cga_ids):
        cga = runs[cga_id]
        n_ann = _ann_count(cga)
        alpha = cga["alpha"]
        ax.plot(cga["cumulative_s"], np.array(cga["val_acc"]) * 100,
                color=CGA_COLORS[i], lw=1.8, linestyle=CGA_STYLES[i],
                label=f"CGA ($\\alpha$={alpha}, {n_ann} ann.)")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Cumulative time (s)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.9)

fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIG_DIR, "compute_vs_accuracy.pdf"))
print("Saved compute_vs_accuracy.pdf")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Training loss curves
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
axes = axes.flatten()

for ax, (title, ce_id, cga_ids) in zip(axes, COMPARISONS):
    ce = runs[ce_id]
    ep_ce = np.arange(1, len(ce["train_loss"]) + 1)
    ax.plot(ep_ce, ce["train_loss"],
            color=CE_COLOR, lw=1.8, label="Cross-entropy")

    for i, cga_id in enumerate(cga_ids):
        cga = runs[cga_id]
        ep_cga = np.arange(1, len(cga["train_loss"]) + 1)
        n_ann = _ann_count(cga)
        alpha = cga["alpha"]
        ax.plot(ep_cga, cga["train_loss"],
                color=CGA_COLORS[i], lw=1.8, linestyle=CGA_STYLES[i],
                label=f"CGA ($\\alpha$={alpha}, {n_ann} ann.)")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(loc="upper right", framealpha=0.9)

fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIG_DIR, "train_loss_curves.pdf"))
print("Saved train_loss_curves.pdf")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Print summary table (for LaTeX)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
header = f"{'Dataset':<12} {'Method':<35} {'n_train':>7} {'Ann.':>5} {'Epochs':>6} {'Val Acc':>8} {'Time(s)':>8} {'Steps':>6}"
print(header)
print("-" * len(header))

for title, ce_id, cga_ids in COMPARISONS:
    ce = runs[ce_id]
    print(f"{title:<12} {'Cross-entropy (control)':<35} {ce['n_train']:>7} {_ann_count(ce):>5} "
          f"{ce['epochs']:>6} {ce['val_acc'][-1]*100:>7.1f}% {ce['total_time_s']:>8.2f} {ce['total_grad_steps']:>6}")
    for cga_id in cga_ids:
        cga = runs[cga_id]
        label = f"CGA Softplus (a={cga['alpha']})"
        print(f"{'':12} {label:<35} {cga['n_train']:>7} {_ann_count(cga):>5} "
              f"{cga['epochs']:>6} {cga['val_acc'][-1]*100:>7.1f}% {cga['total_time_s']:>8.2f} {cga['total_grad_steps']:>6}")
    delta = runs[cga_ids[-1]]["val_acc"][-1] - ce["val_acc"][-1]
    print(f"{'':12} {'  --> Delta':35} {'':>7} {'':>5} {'':>6} {delta*100:>+7.1f}%")
    print()

print("Done.")
