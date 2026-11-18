"""Figures for project_progress_update.md.

Values are the results reported in the update (notebooks 11, 12, 13, 17 and 18),
written here so the figures can be regenerated without rerunning the experiments.
Usage, from the repository root:  uv run python documentacion/figures_en/make_update_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parent
CPC, RF, ACCENT, INK, MUTED, SOFT = "#2F6DB5", "#8C8C8C", "#D9822B", "#1F1F1F", "#6B6B6B", "#E8EFF8"
plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------------- 1. Highlights ----------------
tiles = [
    ("11 of 17", "sensors are enough\n(10 variables, EML)"),
    ("+0.11", "macro F1 over Random Forest\nwith 5 labels per activity"),
    ("2.2×", "fewer labels than Random Forest\nfor the same F1"),
    ("0.884 · 0.898", "final test on the held-out\nsubjects 13 · 14"),
]
fig, ax = plt.subplots(figsize=(15, 2.8))
ax.set(xlim=(0, 4), ylim=(0, 1)); ax.axis("off")
for i, (big, small) in enumerate(tiles):
    ax.add_patch(FancyBboxPatch((i + 0.04, 0.05), 0.92, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=SOFT, ec="none"))
    ax.text(i + 0.5, 0.64, big, ha="center", va="center", fontsize=24, fontweight="bold", color=CPC)
    ax.text(i + 0.5, 0.27, small, ha="center", va="center", fontsize=11, color=INK, linespacing=1.3)
save(fig, "update_highlights.png")


# ---------------- 2. Minimal sensors: body + elimination path ----------------
fig, (axb, axc) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [0.8, 1.6]})
# simple front view of a body; coordinates in arbitrary units
J = {"head": (0, 9.3), "neck": (0, 8.4), "sternum": (0, 7.4), "pelvis": (0, 5.4),
     "shL": (-1.1, 8.2), "shR": (1.1, 8.2), "elL": (-1.6, 6.6), "elR": (1.6, 6.6),
     "wrL": (-1.9, 5.1), "wrR": (1.9, 5.1), "haL": (-2.0, 4.5), "haR": (2.0, 4.5),
     "hipL": (-0.6, 5.2), "hipR": (0.6, 5.2), "knL": (-0.7, 2.9), "knR": (0.7, 2.9),
     "anL": (-0.75, 0.6), "anR": (0.75, 0.6), "ftL": (-1.05, 0.15), "ftR": (1.05, 0.15)}
bones = [("neck", "pelvis"), ("shL", "shR"), ("shL", "elL"), ("elL", "wrL"), ("wrL", "haL"),
         ("shR", "elR"), ("elR", "wrR"), ("wrR", "haR"), ("hipL", "hipR"), ("hipL", "knL"),
         ("knL", "anL"), ("anL", "ftL"), ("hipR", "knR"), ("knR", "anR"), ("anR", "ftR")]
for a, b in bones:
    axb.plot(*zip(J[a], J[b]), color="#CCCCCC", lw=7, solid_capstyle="round", zorder=1)
axb.add_patch(Circle(J["head"], 0.55, fc="#E6E6E6", ec="none", zorder=1))
kept = {"pelvis": (0, 5.6), "uaL": (-1.35, 7.4), "uaR": (1.35, 7.4), "faL": (-1.75, 5.85),
        "faR": (1.75, 5.85), "haL": J["haL"], "haR": J["haR"], "thL": (-0.65, 4.0),
        "thR": (0.65, 4.0), "llL": (-0.72, 1.75), "llR": (0.72, 1.75)}
dropped = {"head": J["head"], "sternum": J["sternum"], "shL": (-0.65, 8.3), "shR": (0.65, 8.3),
           "ftL": J["ftL"], "ftR": J["ftR"]}
for p in kept.values():
    axb.plot(*p, "o", ms=13, color=CPC, mec="white", mew=2, zorder=3)
for p in dropped.values():
    axb.plot(*p, "o", ms=13, color="white", mec=RF, mew=2.2, zorder=3)
axb.plot([], [], "o", ms=11, color=CPC, mec="white", label="kept (11)")
axb.plot([], [], "o", ms=11, color="white", mec=RF, mew=2, label="removed (6)")
axb.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
axb.set(xlim=(-3, 3), ylim=(-0.3, 10.1), aspect="equal"); axb.axis("off")
axb.set_title("EML: elbows, hands (manos), legs", fontsize=12.5)

steps = ["All\n30 var · 17 sensors", "EML\n10 var · 11 sensors", "Elbows + legs\n6 var · 9 sensors",
         "Elbows only\n2 var · 4 sensors"]
rf = [0.894, 0.892, 0.860, 0.714]
cpc = [0.878, 0.873, 0.861, 0.740]
x = np.arange(len(steps))
axc.axvspan(-0.3, 1.3, color=SOFT, lw=0, zorder=0)
axc.text(0.5, 0.925, "no significant loss", ha="center", color=CPC, fontsize=12, fontweight="bold")
axc.plot(x, rf, "-s", color=RF, lw=2.4, ms=9, mec="white", mew=1.5, label="Random Forest")
axc.plot(x, cpc, "-o", color=CPC, lw=2.6, ms=10, mec="white", mew=1.5, label="CPC-GRU")
for i in range(len(x)):
    axc.text(x[i] + 0.07, cpc[i] - 0.022 if i < 2 else cpc[i] + 0.012, f"{cpc[i]:.3f}", color=CPC, fontsize=10.5)
axc.set_xticks(x, steps, fontsize=10.5)
axc.set(ylim=(0.68, 0.95), xlim=(-0.3, 3.3), ylabel="Macro F1 (LOSO, subjects 1–12)")
axc.grid(axis="y", color="#E5E5E5"); axc.set_axisbelow(True)
axc.legend(loc="lower left", frameon=False)
axc.set_title("Backward elimination of sensor groups", fontsize=12.5)
fig.tight_layout()
save(fig, "minimal_sensors_eml.png")


# ---------------- 3. Few labels ----------------
n = [1, 2, 5, 10, 25, 50, 100]
cpc = [0.368, 0.473, 0.620, 0.707, 0.769, 0.800, 0.827]
rf = [0.323, 0.407, 0.513, 0.611, 0.722, 0.777, 0.816]
scratch10 = 0.615
sig = [True, True, True, True, False, False, False]
x = np.arange(len(n))
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.axvspan(-0.4, 3.4, color=SOFT, lw=0, zorder=0)
ax.text(1.5, 0.935, "CPC significantly better", ha="center", fontsize=12.5, color=CPC, fontweight="bold")
ax.axhline(0.892, color=RF, ls=":", lw=1.5); ax.axhline(0.873, color=CPC, ls=":", lw=1.5)
ax.text(6.35, 0.896, "RF, all labels (0.892)", ha="right", va="bottom", fontsize=10.5, color=MUTED)
ax.text(6.35, 0.866, "CPC, all labels (0.873)", ha="right", va="top", fontsize=10.5, color=CPC)
ax.plot(x, rf, "-s", color=RF, lw=2.4, ms=9, mec="white", mew=1.5, label="Random Forest")
ax.plot(x, cpc, "-o", color=CPC, lw=2.8, ms=10, mec="white", mew=1.5, label="CPC-GRU (pretrained without labels)")
ax.plot([3], [scratch10], "D", color=ACCENT, ms=9, mec="white", mew=1.5, label="Same network from scratch")
for i, s in enumerate(sig):
    if s:
        ax.text(x[i], cpc[i] + 0.02, "*", ha="center", fontsize=20, color=INK)
ax.annotate("", xy=(2, cpc[2]), xytext=(2, rf[2]), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
ax.text(2.1, (cpc[2] + rf[2]) / 2, "+0.11", fontsize=13, fontweight="bold", va="center")
ax.set_xticks(x, [str(v) for v in n])
ax.set(xlim=(-0.4, 6.4), ylim=(0.28, 0.97), xlabel="Labeled windows per activity",
       ylabel="Macro F1 on a new person")
ax.grid(axis="y", color="#E5E5E5"); ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False)
ax.text(0, -0.2, "* Holm-corrected Wilcoxon p < 0.05 over the 12 LOSO subjects", transform=ax.transAxes,
        fontsize=10, color=MUTED)
save(fig, "few_labels_curve.png")


# ---------------- 4. Window × horizon heatmap ----------------
win = ["2 s", "4 s", "6 s"]
hor = ["0.5 s", "1 s", "2 s"]
F = np.array([[0.876, 0.878, np.nan], [0.881, 0.878, 0.882], [0.885, 0.882, 0.871]])
fig, ax = plt.subplots(figsize=(5.6, 4.0))
im = ax.imshow(F, cmap="Blues", vmin=0.86, vmax=0.89)
for i in range(3):
    for j in range(3):
        if np.isnan(F[i, j]):
            ax.text(j, i, "—", ha="center", va="center", color=MUTED)
        else:
            ax.text(j, i, f"{F[i, j]:.3f}", ha="center", va="center", fontsize=12,
                    color="white" if F[i, j] > 0.879 else INK, fontweight="bold" if (i, j) == (1, 1) else None)
ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec=ACCENT, lw=2.5))
ax.set_xticks(range(3), hor); ax.set_yticks(range(3), win)
ax.set(xlabel="CPC prediction horizon", ylabel="Window length")
ax.spines[:].set_visible(False); ax.tick_params(length=0)
ax.set_title("Macro F1 (LOSO, 1 seed) · orange = current design", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "window_horizon_heatmap.png")


# ---------------- 5. Bassani et al. comparison on subject 11 ----------------
labels = ["BiLSTM (Bassani et al., 2025)", "CPC-GRU · 12 Hz · 4 s · 13 subjects",
          "CPC-GRU · 12 Hz · 4 s · 10 subjects", "CPC-GRU · 240 Hz · 1 s · 1 s horizon",
          "CPC-GRU · 240 Hz · 1 s · 0.5 s horizon"]
s11 = [0.903, 0.910, 0.908, 0.895, 0.888]
mean = [np.nan, 0.883, 0.880, 0.874, 0.854]
y = np.arange(len(labels))[::-1]
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.barh(y, s11, color=[RF] + [CPC] * 4, height=0.55)
ax.plot(mean, y, "D", color=ACCENT, ms=9, mec="white", mew=1.5, label="Mean over all LOSO subjects")
for yi, v in zip(y, s11):
    ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=11)
ax.set_yticks(y, labels)
ax.set(xlim=(0.84, 0.93), xlabel="Macro F1 on subject 11 (the article's test subject)")
ax.grid(axis="x", color="#E5E5E5"); ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), frameon=False)
save(fig, "bassani_subject11.png")


# ---------------- 6. Final model: LOSO vs. final test ----------------
cols = ["LOSO mean ± sd\n(subjects 1–12)", "Final test\nsubject 13", "Final test\nsubject 14"]
cpc_v, cpc_e = [0.873, 0.884, 0.898], [0.036, 0, 0]
rf_v, rf_e = [0.892, 0.885, 0.906], [0.029, 0, 0]
x = np.arange(3)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.errorbar(x - 0.1, cpc_v, yerr=cpc_e, fmt="o", color=CPC, ms=11, capsize=5, lw=2, label="CPC-GRU (EML)")
ax.errorbar(x + 0.1, rf_v, yerr=rf_e, fmt="s", color=RF, ms=10, capsize=5, lw=2, label="Random Forest (EML)")
for i in range(3):
    ax.text(x[i] - 0.17, cpc_v[i], f"{cpc_v[i]:.3f}", ha="right", va="center", color=CPC, fontsize=11)
    ax.text(x[i] + 0.17, rf_v[i], f"{rf_v[i]:.3f}", ha="left", va="center", color=MUTED, fontsize=11)
ax.set_xticks(x, cols)
ax.set(xlim=(-0.6, 2.6), ylim=(0.82, 0.94), ylabel="Macro F1")
ax.grid(axis="y", color="#E5E5E5"); ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, ncol=2)
save(fig, "final_model_scores.png")
