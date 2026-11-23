"""Figures for the sponsor presentation (poster palette: olive + slate/teal accents).

Run:  uv run python Presentacion/make_figures.py
Values come from Articulo/articulo.tex (final model, LOSO subjects 1-12).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

OLIVE = "#9AA542"
OLIVE_LINK = "#6B7A1F"
OLIVE_DARK = "#1B2208"
OLIVE_SOFT = "#E4E8C7"
SLATE = "#3E5C76"
MUTED = "#6B6B5E"
INK = "#1A1A10"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

# ---------------- 1. Body with kept / removed sensors ----------------
J = {"head": (0, 9.3), "neck": (0, 8.4), "sternum": (0, 7.4), "pelvis": (0, 5.4),
     "shL": (-1.1, 8.2), "shR": (1.1, 8.2), "elL": (-1.6, 6.6), "elR": (1.6, 6.6),
     "wrL": (-1.9, 5.1), "wrR": (1.9, 5.1), "haL": (-2.0, 4.5), "haR": (2.0, 4.5),
     "hipL": (-0.6, 5.2), "hipR": (0.6, 5.2), "knL": (-0.7, 2.9), "knR": (0.7, 2.9),
     "anL": (-0.75, 0.6), "anR": (0.75, 0.6), "ftL": (-1.05, 0.15), "ftR": (1.05, 0.15)}
bones = [("neck", "pelvis"), ("shL", "shR"), ("shL", "elL"), ("elL", "wrL"), ("wrL", "haL"),
         ("shR", "elR"), ("elR", "wrR"), ("wrR", "haR"), ("hipL", "hipR"), ("hipL", "knL"),
         ("knL", "anL"), ("anL", "ftL"), ("hipR", "knR"), ("knR", "anR"), ("anR", "ftR")]
kept = [(0, 5.6), (-1.35, 7.4), (1.35, 7.4), (-1.75, 5.85), (1.75, 5.85), J["haL"], J["haR"],
        (-0.65, 4.0), (0.65, 4.0), (-0.72, 1.75), (0.72, 1.75)]
dropped = [J["head"], J["sternum"], (-0.65, 8.3), (0.65, 8.3), J["ftL"], J["ftR"]]

fig, ax = plt.subplots(figsize=(3.6, 6.2))
for a, b in bones:
    ax.plot(*zip(J[a], J[b]), color="#D6D8C8", lw=9, solid_capstyle="round", zorder=1)
ax.add_patch(Circle(J["head"], 0.55, fc="#ECEDE2", ec="none", zorder=1))
for p in kept:
    ax.plot(*p, "o", ms=17, color=OLIVE_LINK, mec="white", mew=2.5, zorder=3)
for p in dropped:
    ax.plot(*p, "o", ms=17, color="white", mec=MUTED, mew=2.5, zorder=3)
ax.set(xlim=(-2.6, 2.6), ylim=(-0.3, 10.1), aspect="equal")
ax.axis("off")
fig.savefig(OUT / "body_sensors.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)

# ---------------- 2. Confusion matrix of the final model (English) ----------------
labels = ["N", "LF", "K", "PT", "LT", "W", "PF"]
cm = [[0.89, 0.03, 0.00, 0.04, 0.01, 0.00, 0.03],
      [0.02, 0.90, 0.02, 0.02, 0.04, 0.00, 0.00],
      [0.00, 0.01, 0.94, 0.05, 0.00, 0.00, 0.00],
      [0.07, 0.03, 0.05, 0.82, 0.02, 0.00, 0.00],
      [0.09, 0.04, 0.00, 0.03, 0.79, 0.05, 0.00],
      [0.00, 0.00, 0.00, 0.00, 0.02, 0.96, 0.01],
      [0.02, 0.00, 0.00, 0.00, 0.00, 0.03, 0.95]]
cmap = LinearSegmentedColormap.from_list("olive", ["#FFFFFF", OLIVE_SOFT, OLIVE, OLIVE_LINK, "#3D4710"])
fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
highlight = {(4, 0), (3, 0), (3, 2), (2, 3), (4, 5)}  # main error pairs discussed in the article
for i in range(7):
    for j in range(7):
        v = cm[i][j]
        txt = "·" if v == 0 else f"{v:.2f}"
        color = "white" if v > 0.5 else (SLATE if (i, j) in highlight else MUTED)
        weight = "bold" if (i == j or (i, j) in highlight) else "normal"
        ax.text(j, i, txt, ha="center", va="center", fontsize=11.5, color=color, weight=weight)
for (i, j) in highlight:
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=SLATE, lw=2))
ax.set_xticks(range(7), labels)
ax.set_yticks(range(7), labels)
ax.set_xlabel("Predicted activity")
ax.set_ylabel("True activity")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
fig.savefig(OUT / "confusion_final.png", dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)
print("figures written to", OUT)
