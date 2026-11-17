"""Genera las figuras del póster (fig_ventana.png y fig_pocas.png) con los colores de la plantilla.

Uso, desde la raíz del repositorio:  uv run python Poster/generar_figuras.py

fig_ventana.png lee la caché sincronizada a 12 Hz (notebooks/_cache/sync/fs12), que no se versiona:
se genera con el notebook 07 a partir de los datos de Zenodo. fig_pocas.png usa las medias del
experimento de pocas etiquetas (notebook 17) escritas abajo. Las dos figuras ya están en el repositorio;
sólo hace falta correr este script para modificarlas.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
REPO = OUT.parent
OLIVE, DARK, LINK, SOFT, INK, MUTED = "#9AA542", "#1B2208", "#6B7A1F", "#E4E8C7", "#1A1A10", "#6B6B5E"
plt.rcParams.update({"font.family": "serif", "font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})

# ---------------- 1. Ventana de 4 s y etiqueta del último instante ----------------
C = REPO / "notebooks/_cache/sync/fs12/bimanual"
cols = [c["name"] for c in json.load(open(C / "columns.json"))["mvnx"]]
z = np.load(C / "series.npz", allow_pickle=True)
tr = pd.read_csv(C / "trials.csv").iloc[0]
sl = slice(int(tr.offset), int(tr.offset + tr.n_steps))
t = z["time_s"][sl] - z["time_s"][sl][0]
y = z["label_mvnx"][sl]
codo = z["mvnx"][sl, cols.index("jointAngle__jRightElbow__c3")]
rodilla = z["mvnx"][sl, cols.index("jointAngle__jRightKnee__c3")]
LABELS = ["N", "LF", "K", "PT", "LT", "W", "PF"]

lab = np.where(y >= 0)[0]
t0 = t[lab[0]] + 1.0
t1 = t0 + 24
m = (t >= t0) & (t <= t1)
fig, ax = plt.subplots(figsize=(11, 3.6))
# franjas de actividad
seg_start = None
idx = np.where(m)[0]
for k, i in enumerate(idx):
    if seg_start is None:
        seg_start = i
    last = k == len(idx) - 1
    if last or y[idx[k + 1]] != y[i]:
        a, b = t[seg_start], t[i] + 1 / 12
        if y[i] >= 0:
            ax.add_patch(Rectangle((a, -10), b - a, 200, color=SOFT if (len(ax.patches) % 2 == 0) else "#F2F4E3", lw=0, zorder=0))
            if b - a > 0.9:
                ax.text((a + b) / 2, 158, LABELS[y[i]], ha="center", va="center", fontsize=15, color=DARK, fontweight="bold")
        seg_start = None
ax.plot(t[m], codo[m], color=LINK, lw=2.2, label="Flexión del codo")
ax.plot(t[m], rodilla[m], color=MUTED, lw=2.2, ls="--", label="Flexión de la rodilla")
# ventana de 4 s que termina en un instante de PT/LT
tend = t0 + 15.5
k_end = np.argmin(np.abs(t - tend))
tend = t[k_end]
ax.add_patch(Rectangle((tend - 4, -5), 4, 175, fill=False, ec=DARK, lw=2.5, zorder=3))
ax.annotate(f"etiqueta = {LABELS[y[k_end]]}", xy=(tend, 0), xytext=(tend + 0.6, 20), fontsize=15, color=DARK,
            arrowprops=dict(arrowstyle="->", color=DARK, lw=1.8), zorder=4)
ax.plot([tend], [0], "o", color=DARK, ms=9, zorder=4)
ax.text(tend - 2, 177, "ventana de 4 s", ha="center", fontsize=15, color=DARK)
ax.set(xlim=(t0, t1), ylim=(-8, 190), xlabel="Tiempo (s)", ylabel="Ángulo (°)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, frameon=False, fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "fig_ventana.png", dpi=250)
plt.close(fig)
print("ventana: ensayo", tr.mvnx_file, "etiqueta", LABELS[y[k_end]])

# ---------------- 2. Curva de pocas etiquetas ----------------
n = [1, 2, 5, 10, 25, 50, 100]
cpc = [0.368, 0.473, 0.620, 0.707, 0.769, 0.800, 0.827]
rf = [0.323, 0.407, 0.513, 0.611, 0.722, 0.777, 0.816]
sig = [True, True, True, True, False, False, False]
x = np.arange(len(n))
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axvspan(-0.4, 3.4, color=SOFT, lw=0, zorder=0)
ax.text(1.5, 0.93, "Ventaja significativa de CPC", ha="center", fontsize=16, color=DARK, fontweight="bold")
ax.axhline(0.873, color=LINK, ls=":", lw=1.8); ax.axhline(0.892, color=MUTED, ls=":", lw=1.8)
ax.text(6.35, 0.873 - 0.022, "CPC, todas las etiquetas (0.873)", ha="right", fontsize=13, color=LINK)
ax.text(6.35, 0.892 + 0.008, "RF, todas las etiquetas (0.892)", ha="right", fontsize=13, color=MUTED)
ax.plot(x, rf, "-s", color=MUTED, lw=2.6, ms=10, mec="white", mew=2, label="Random Forest")
ax.plot(x, cpc, "-o", color=LINK, lw=3, ms=11, mec="white", mew=2, label="CPC-GRU (preentrenado sin etiquetas)")
for i, s in enumerate(sig):
    if s:
        ax.text(x[i], cpc[i] + 0.025, "*", ha="center", fontsize=24, color=DARK)
ax.annotate("", xy=(2, cpc[2]), xytext=(2, rf[2]), arrowprops=dict(arrowstyle="<->", color=DARK, lw=2))
ax.text(2.12, (cpc[2] + rf[2]) / 2, "+0.11", fontsize=17, color=DARK, fontweight="bold", va="center")
ax.set_xticks(x, [str(v) for v in n])
ax.set(xlim=(-0.4, 6.4), ylim=(0.28, 0.97), xlabel="Ventanas etiquetadas por actividad",
       ylabel="F1 macro en una persona nueva")
ax.grid(axis="y", color="#DDDDD0", lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "fig_pocas.png", dpi=250)
plt.close(fig)
print("ok")
