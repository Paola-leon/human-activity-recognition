"""Figuras de la guía metodológica (documentacion/guia_metodologia_completa.md).

Uso, desde la raíz del repositorio:
    uv run --no-project --with numpy --with pandas --with matplotlib --with scipy \
        python documentacion/figuras/guia/generar_figuras_guia.py

Lee datos reales del proyecto:
- notebooks/_cache/sync/fs12 y fs240_articulo (caché de sincronización del notebook 07; no se versiona),
- checkpoints/nb11_variables_minimas_fs12 (F1 y predicciones por sujeto del modelo final y del Random Forest).
Las figuras de Holm y del esquema de fuga son ilustrativas y lo dicen en su título.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy.stats import wilcoxon, rankdata

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
LABELS = ["N", "LF", "K", "PT", "LT", "W", "PF"]
NOMBRES = {"N": "Postura neutral", "LF": "Levantar del piso", "K": "Mantener la carga", "PT": "Colocar en mesa",
           "LT": "Levantar de mesa", "W": "Caminar con carga", "PF": "Colocar en piso"}
# Okabe-Ito (apto para daltonismo), una por actividad
ACT_COL = dict(zip(LABELS, ["#BBBBBB", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]))
INK, MUTED, BLUE, ORANGE, GREEN, RED, GRAY = "#1A1A1A", "#6B6B6B", "#2563EB", "#D97706", "#16A34A", "#DC2626", "#9CA3AF"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "figure.dpi": 110})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("ok", name)


def bandas(ax, t, y, y0, y1, texto=True, alpha=0.55):
    """Franjas de color por actividad a lo largo del tiempo."""
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            if y[start] >= 0:
                lab = LABELS[y[start]]
                a, b = t[start], t[i - 1] + (t[1] - t[0])
                ax.add_patch(Rectangle((a, y0), b - a, y1 - y0, color=ACT_COL[lab], alpha=alpha, lw=0, zorder=0))
                if texto and b - a > 0.9:
                    ax.text((a + b) / 2, y1 - 0.08 * (y1 - y0), lab, ha="center", va="top", fontsize=10, fontweight="bold")
            start = i


# ---------------------------------------------------------------- datos reales de un ensayo
C12 = REPO / "notebooks/_cache/sync/fs12/bimanual"
cols12 = [c["name"] for c in json.load(open(C12 / "columns.json"))["mvnx"]]
z12 = np.load(C12 / "series.npz", allow_pickle=True)
tr12 = pd.read_csv(C12 / "trials.csv").iloc[0]
sl = slice(int(tr12.offset), int(tr12.offset + tr12.n_steps))
t12 = z12["time_s"][sl] - z12["time_s"][sl][0]
y12 = z12["label_mvnx"][sl].astype(int)
codo12 = z12["mvnx"][sl, cols12.index("jointAngle__jRightElbow__c3")]

# ================================================================ G01 ventanas que se traslapan
lab = np.where(y12 >= 0)[0]
t0 = t12[lab[0]] + 1.0
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [2.2, 1.3]}, sharex=True)
ax = axes[0]
m = (t12 >= t0) & (t12 <= t0 + 16)
bandas(ax, t12[m], y12[m], -10, 130)
ax.plot(t12[m], codo12[m], color=INK, lw=1.8)
ax.set(ylabel="Flexión del codo (°)", ylim=(-10, 130),
       title="Una ventana de 4 s cada 0.5 s: ventanas consecutivas comparten 3.5 s (87.5 %)")
ax2 = axes[1]
tend0 = t0 + 7.0
for k in range(6):
    te = tend0 + 0.5 * k
    ax2.add_patch(Rectangle((te - 4, 5 - k), 4, 0.7, fill=True, color=BLUE, alpha=0.25 + 0.1 * k, lw=0))
    ki = np.argmin(np.abs(t12 - te))
    ax2.plot([te], [5 - k + 0.35], "o", color=INK, ms=5)
    ax2.text(te + 0.12, 5 - k + 0.35, f"ventana {k + 1} → etiqueta {LABELS[y12[ki]]}", va="center", fontsize=9)
ax2.annotate("", xy=(tend0 - 3.5, 5.95), xytext=(tend0, 5.95), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
ax2.text(tend0 - 1.75, 6.1, "3.5 s compartidos entre la ventana 1 y la 2", ha="center", color=RED, fontsize=9)
ax2.set(ylim=(-0.6, 6.8), yticks=[], xlabel="Tiempo (s)")
ax2.spines["left"].set_visible(False)
ax2.set_title("Cada barra es una ventana; el punto es su último instante, que da la etiqueta", fontsize=10)
save(fig, "g01_ventanas_traslape.png")

# ================================================================ G02 submuestreo temporal 240 → 12 Hz
C240 = REPO / "notebooks/_cache/sync/fs240_articulo/bimanual"
cols240 = [c["name"] for c in json.load(open(C240 / "columns.json"))["mvnx"]]
z240 = np.load(C240 / "series.npz", mmap_mode="r", allow_pickle=True)
tr240 = pd.read_csv(C240 / "trials.csv").iloc[0]
sl2 = slice(int(tr240.offset), int(tr240.offset + tr240.n_steps))
t240 = np.asarray(z240["time_s"][sl2]); t240 = t240 - t240[0]
codo240 = np.asarray(z240["mvnx"][sl2, cols240.index("jointAngle__jRightElbow__c3")])
a, b = t0 + 2.0, t0 + 6.0
m240, m12 = (t240 >= a) & (t240 <= b), (t12 >= a) & (t12 <= b)
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(t240[m240], codo240[m240], color=GRAY, lw=1, label=f"Original a 240 Hz ({m240.sum()} muestras en 4 s)")
ax.plot(t12[m12], codo12[m12], "o-", color=BLUE, ms=5, lw=1.4, label=f"Remuestreada a 12 Hz ({m12.sum()} muestras en 4 s)")
ax.set(xlabel="Tiempo (s)", ylabel="Flexión del codo (°)",
       title="Submuestreo temporal (downsampling): filtro pasa-bajas de 4.8 Hz y una muestra cada 1/12 s")
ax.legend(frameon=False, loc="best")
save(fig, "g02_submuestreo_temporal.png")

# ================================================================ G03 escalado robusto (q5, q95)
vx, vy, vz = (z12["mvnx"][:, cols12.index(f"velocity__RightHand__{c}")] for c in "xyz")
rap = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
rap = rap[np.isfinite(rap)]
lo, hi = np.quantile(rap, [0.001, 0.999])
x = np.clip(rap, lo, hi)
q5, med, q95 = np.percentile(x, [5, 50, 95])
xs = (x - med) / (q95 - q5)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.hist(rap, bins=120, color=GRAY, alpha=0.8)
for v, nombre, c, yy in [(q5, "q5", ORANGE, 0.62), (med, "mediana", INK, 0.92), (q95, "q95", ORANGE, 0.92), (hi, "q99.9 (recorte)", RED, 0.92)]:
    ax.axvline(v, color=c, ls="--" if "q" in nombre else "-", lw=1.5)
    ax.text(v, ax.get_ylim()[1] * yy, f" {nombre}\n {v:.2f}", color=c, fontsize=9, va="top")
ax.set(xlabel="Rapidez de la mano derecha (m/s)", ylabel="Instantes", title="Antes: escala física, cola larga a la derecha")
ax = axes[1]
ax.hist(xs, bins=120, color=BLUE, alpha=0.8)
ax.axvline(0, color=INK, lw=1.5); ax.text(0, ax.get_ylim()[1] * 0.92, " mediana = 0", fontsize=9, va="top")
ax.set(xlabel="(x − mediana) / (q95 − q5)", title="Después: centrada en 0; el 90 % central mide ≈ 1")
fig.suptitle("Escalado robusto de una variable (ensayos bimanuales, todos los sujetos; ilustrativo)", y=1.02)
save(fig, "g03_escalado_robusto.png")

# ================================================================ G04 desbalance y estrategias
CONTEOS = pd.DataFrame(  # ventanas etiquetadas por sujeto y actividad (notebook 18, sección 2)
    [[1845, 304, 524, 531, 337, 133, 248], [1228, 222, 156, 405, 346, 241, 170], [1439, 268, 244, 508, 337, 264, 237],
     [1376, 316, 201, 501, 314, 253, 181], [1278, 384, 243, 539, 371, 298, 245], [1383, 270, 216, 475, 314, 279, 194],
     [1666, 332, 253, 566, 381, 225, 267], [1408, 345, 239, 621, 346, 316, 272], [1766, 267, 199, 458, 310, 214, 205],
     [1279, 240, 201, 586, 281, 245, 227], [2037, 391, 711, 594, 332, 180, 279], [1286, 250, 238, 493, 309, 312, 199]],
    index=range(1, 13), columns=LABELS)
tot = CONTEOS.sum()
cols = [ACT_COL[l] for l in LABELS]
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
ax = axes[0, 0]
ax.bar(LABELS, tot, color=cols, edgecolor=INK, lw=0.5)
for i, v in enumerate(tot):
    ax.text(i, v, f"{v:,}\n({v / tot.sum():.0%})", ha="center", va="bottom", fontsize=8)
ax.set(title="1. Datos originales (sujetos 1–12)", ylabel="Ventanas etiquetadas", ylim=(0, tot.max() * 1.25))
ax = axes[0, 1]
und = tot.copy(); und["N"] = tot.drop("N").max()
ax.bar(LABELS, tot, color="white", edgecolor=GRAY, ls="--")
ax.bar(LABELS, und, color=cols, edgecolor=INK, lw=0.5)
ax.text(0, tot["N"] * 0.6, f"se descartan\n{tot['N'] - und['N']:,}\nventanas N", ha="center", fontsize=9, color=RED)
ax.set(title="2. Submuestreo (undersampling) de N\n(probado en el notebook 12)", ylim=(0, tot.max() * 1.25))
ax = axes[1, 0]
ovr = pd.Series(und["N"], index=LABELS)
ax.bar(LABELS, und, color=cols, edgecolor=INK, lw=0.5, label="originales")
ax.bar(LABELS, ovr - und, bottom=und, color=cols, alpha=0.35, hatch="///", edgecolor=INK, lw=0.5, label="copias con ruido")
ax.legend(frameon=False, fontsize=9)
ax.set(title="3. Submuestreo de N + sobremuestreo (oversampling)\ncon copias ruidosas (esquema de Bassani et al.)", ylabel="Ventanas", ylim=(0, tot.max() * 1.25))
ax = axes[1, 1]
w = tot.sum() / (len(LABELS) * tot)
ax.bar(LABELS, w, color=cols, edgecolor=INK, lw=0.5)
for i, v in enumerate(w):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.axhline(1, color=MUTED, ls=":")
ax.set(title="4. Pesos por clase en la pérdida (este proyecto)\nno se quita ni se copia ninguna ventana", ylabel="Peso de cada error")
fig.tight_layout()
save(fig, "g04_desbalance_estrategias.png")

# ================================================================ G05 esquema LOSO del proyecto
fig, ax = plt.subplots(figsize=(11, 5.6))
CV = list(range(1, 13))
for r, held in enumerate(CV):
    val = CV[(r + 1) % 12]
    for s in range(1, 15):
        if s in (13, 14):
            c, txt = "#E5E7EB", ""
        elif s == held:
            c, txt = RED, "P"
        elif s == val:
            c, txt = ORANGE, "V"
        else:
            c, txt = BLUE, ""
        ax.add_patch(Rectangle((s - 0.45, -r - 0.4), 0.9, 0.8, color=c, alpha=0.85 if c != "#E5E7EB" else 1))
        if txt:
            ax.text(s, -r, txt, ha="center", va="center", color="white", fontweight="bold")
r = 13
for s in range(1, 15):
    c = RED if s in (13, 14) else (ORANGE if s == 12 else BLUE)
    ax.add_patch(Rectangle((s - 0.45, -r - 0.4), 0.9, 0.8, color=c, alpha=0.85))
    if s in (13, 14): ax.text(s, -r, "P", ha="center", va="center", color="white", fontweight="bold")
    if s == 12: ax.text(s, -r, "V", ha="center", va="center", color="white", fontweight="bold")
ax.set_yticks([-i for i in range(12)] + [-13], [f"Ronda {i + 1}" for i in range(12)] + ["Prueba final"])
ax.set_xticks(range(1, 15), [str(s) for s in range(1, 15)])
ax.set(xlabel="Sujeto", xlim=(0.4, 14.6), ylim=(-13.6, 0.6), title="LOSO del proyecto (12 rondas) y prueba final")
for c, l in [(BLUE, "Entrenamiento (10 sujetos)"), (ORANGE, "V = validación interna (decide épocas y C)"),
             (RED, "P = prueba (sólo se mide el F1)"), ("#E5E7EB", "Reservado: no se usa")]:
    ax.bar([0], [0], color=c, label=l)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
ax.spines[["left", "bottom"]].set_visible(False)
save(fig, "g05_loso_proyecto.png")

# ================================================================ G06 esquemas de Bassani
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
for s in range(1, 15):
    is_test = rng.random(40) < 0.3
    for j, tst in enumerate(is_test):
        ax.add_patch(Rectangle((j, -s), 0.9, 0.8, color=RED if tst else BLUE, alpha=0.85, lw=0))
ax.set(xlim=(0, 40), ylim=(-14.5, 0), title="Partición 70/30 (aleatoria por ventana, después de balancear)\ncada fila es un sujeto; cada cuadro, una ventana",
       xlabel="Ventanas de cada sujeto (muestra)")
ax.set_yticks([-s + 0.4 for s in range(1, 15)], [f"S{s}" for s in range(1, 15)])
ax.set_xticks([])
ax = axes[1]
orden = [2, 3, 4, 10, 9, 6, 7, 8, 5, 1, 13, 12, 14]
for i, s in enumerate(orden):
    ax.add_patch(Rectangle((i, 0), 0.9, 0.9, color=BLUE, alpha=0.85))
    ax.text(i + 0.45, 0.45, str(s), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
ax.add_patch(Rectangle((14, 0), 0.9, 0.9, color=RED, alpha=0.85))
ax.text(14.45, 0.45, "11", ha="center", va="center", color="white", fontweight="bold")
ax.text(6.5, -0.5, "13 sujetos de entrenamiento\n(en el orden de BiLSTM.m)", ha="center", va="top", fontsize=9)
ax.text(14.45, -0.5, "prueba", ha="center", va="top", fontsize=9, color=RED)
ax.set(xlim=(-0.3, 15.2), ylim=(-1.6, 1.4), title="Su \"LOSO\": una sola ronda, prueba en el sujeto 11")
ax.axis("off")
for c, l in [(BLUE, "Entrenamiento"), (RED, "Prueba")]:
    axes[0].bar([0], [0], color=c, label=l)
axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, -0.22), ncol=2, frameon=False)
save(fig, "g06_esquemas_bassani.png")

# ================================================================ G07 fuga en una partición aleatoria (ilustrativo)
fig, ax = plt.subplots(figsize=(11, 3.4))
m = (t12 >= t0) & (t12 <= t0 + 10)
bandas(ax, t12[m], y12[m], 0, 9.5, alpha=0.35)
rng = np.random.default_rng(3)
for k in range(9):
    te = t0 + 5 + 0.5 * k
    tst = rng.random() < 0.35
    ax.add_patch(Rectangle((te - 4, 0.5 + k), 4, 0.7, color=RED if tst else BLUE, alpha=0.8, lw=0))
    ax.text(te + 0.1, 0.85 + k, "prueba" if tst else "entrenamiento", fontsize=8, va="center", color=RED if tst else BLUE)
ax.set(yticks=[], xlabel="Tiempo (s)", ylim=(0, 10), xlim=(t0 + 0.5, t0 + 10.5),
       title="Ilustrativo: si las ventanas se reparten al azar, cada ventana de prueba tiene en entrenamiento\n"
             "vecinas casi idénticas (comparten 3.5 de sus 4 s). El modelo \"ya vio\" a esa persona en ese momento.")
ax.spines["left"].set_visible(False)
save(fig, "g07_fuga_particion_aleatoria.png")

# ================================================================ G08 F1 macro con datos reales (modelo final)
D = REPO / "checkpoints/nb11_variables_minimas_fs12"
ys, ps, f1_gru = [], [], {}
for s in range(1, 13):
    r = json.load(open(D / "gru_ft/EML" / f"ronda_s{s:02d}.json"))
    ys += r["y"]; ps += r["pred"]; f1_gru[s] = r["f1"]
ys, ps = np.array(ys), np.array(ps)
cm = np.zeros((7, 7), int)
for a_, b_ in zip(ys, ps):
    cm[a_, b_] += 1
prec = np.diag(cm) / cm.sum(0); rec = np.diag(cm) / cm.sum(1); f1c = 2 * prec * rec / (prec + rec)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else INK)
ax.set_xticks(range(7), LABELS); ax.set_yticks(range(7), LABELS)
ax.set(xlabel="Predicha", ylabel="Real", title="Matriz de confusión (conteos)\nmodelo final, 12 sujetos juntos")
ax = axes[1]
xx = np.arange(7)
ax.bar(xx - 0.27, prec, 0.27, color=ORANGE, label="Precisión: de lo que predijo X, ¿cuánto era X?")
ax.bar(xx, rec, 0.27, color=GREEN, label="Exhaustividad (recall): de lo que era X, ¿cuánto encontró?")
ax.bar(xx + 0.27, f1c, 0.27, color=BLUE, label="F1 = 2·P·R / (P + R)")
ax.axhline(f1c.mean(), color=BLUE, ls="--"); ax.text(-0.45, 1.03, f"F1 macro (línea azul) = {f1c.mean():.3f}", ha="left", color=BLUE)
acc = np.trace(cm) / cm.sum()
ax.axhline(acc, color=MUTED, ls=":"); ax.text(-0.45, 1.005, f"exactitud (línea gris) = {acc:.3f}", ha="left", color=MUTED)
ax.set_xticks(xx, LABELS); ax.set(ylim=(0.6, 1.05), title="Por actividad, y su promedio simple (macro)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
save(fig, "g08_f1_macro.png")

# ================================================================ G09 Wilcoxon con datos reales
f1_rf = {int(k): v for k, v in json.load(open(D / "rf/EML.json"))["f1"].items()}
S = list(range(1, 13))
g = np.array([f1_gru[s] for s in S]); rf = np.array([f1_rf[s] for s in S])
d = g - rf
rk = rankdata(np.abs(d))
p = wilcoxon(d).pvalue
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.2, 1]})
ax = axes[0]
for i, s in enumerate(S):
    ax.plot([0, 1], [rf[i], g[i]], color=GRAY, lw=1)
ax.scatter([0] * 12, rf, color=MUTED, zorder=3, label="Random Forest")
ax.scatter([1] * 12, g, color=BLUE, zorder=3, label="CPC-GRU")
ax.set_xticks([0, 1], ["Random Forest", "CPC-GRU"]); ax.set(ylabel="F1 macro del sujeto", title="1. Mismo sujeto, dos modelos (pareado)", xlim=(-0.4, 1.4))
ax = axes[1]
order = np.argsort(d)
ax.bar(range(12), d[order], color=[BLUE if v > 0 else MUTED for v in d[order]])
for j, i in enumerate(order):
    ax.text(j, d[i] + (0.002 if d[i] > 0 else -0.002), f"S{S[i]}\nrango {rk[i]:.0f}", ha="center", va="bottom" if d[i] > 0 else "top", fontsize=7)
ax.axhline(0, color=INK, lw=0.8)
ax.set(xticks=[], ylabel="CPC-GRU − Random Forest", title="2. Diferencias por sujeto y su rango por tamaño", ylim=(d.min() - 0.012, d.max() + 0.015))
ax = axes[2]
wpos, wneg = rk[d > 0].sum(), rk[d < 0].sum()
ax.bar(["W+ (gana CPC)", "W− (gana RF)"], [wpos, wneg], color=[BLUE, MUTED])
for i, v in enumerate([wpos, wneg]):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
ax.set(title=f"3. Suma de rangos → p = {p:.3f}\n(si no hubiera diferencia, W+ ≈ W− ≈ 39)", ylabel="Suma de rangos")
fig.suptitle("Prueba de Wilcoxon: modelo final (CPC-GRU) frente a Random Forest, 12 sujetos, todas las etiquetas", y=1.03)
fig.tight_layout()
save(fig, "g09_wilcoxon.png")

# ================================================================ G10 Holm (ilustrativo)
pv = np.array([0.004, 0.010, 0.020, 0.030, 0.200])
m_ = len(pv)
umb = 0.05 / (m_ - np.arange(m_))
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(np.arange(m_) - 0.18, pv, 0.36, color=BLUE, label="p de cada comparación (ordenados)")
ax.bar(np.arange(m_) + 0.18, umb, 0.36, color=ORANGE, label="umbral de Holm = 0.05 / (m − i + 1)")
ax.axhline(0.05, color=RED, ls=":", label="0.05 sin corregir")
for i in range(m_):
    ok = all(pv[:i + 1] <= umb[:i + 1])
    ax.text(i, max(pv[i], umb[i]) + 0.006, "rechaza H0" if ok else "se detiene", ha="center", fontsize=9, color=GREEN if ok else RED)
ax.set_xticks(range(m_), [f"comparación {i + 1}\n(m − i + 1 = {m_ - i})" for i in range(m_)])
ax.set(ylabel="p", title="Corrección de Holm (ejemplo ilustrativo con 5 comparaciones)\n0.020 y 0.030 son < 0.05, pero Holm se detiene en la comparación 3 (0.020 > 0.017)")
ax.legend(frameon=False, fontsize=9)
save(fig, "g10_holm.png")
# ================================================================ G11 ruido gaussiano (copias con ruido)
rng = np.random.default_rng(1)
m = (t12 >= t0 + 4.5) & (t12 < t0 + 8.5)
tt, xx0 = t12[m], codo12[m]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
ax = axes[0]
ax.plot(tt, xx0, color=INK, lw=2.2, label="ventana original")
for k, sd in enumerate([2.0, 2.0, 2.0]):
    ax.plot(tt, xx0 + rng.normal(0, sd, len(tt)), color=ORANGE, lw=1, alpha=0.75, label="copias con ruido (σ = 2°)" if k == 0 else None)
ax.set(xlabel="Tiempo (s)", ylabel="Flexión del codo (°)", title="Sobremuestreo con ruido gaussiano: la misma ventana + ruido aleatorio")
ax.legend(frameon=False)
ax = axes[1]
xs_ = np.linspace(-4, 4, 400)
ax.plot(xs_, np.exp(-xs_ ** 2 / 2) / np.sqrt(2 * np.pi), color=ORANGE, lw=2)
for k, (lo_, txt) in enumerate([(1, "68 %"), (2, "95 %")]):
    sel = np.abs(xs_) <= lo_
    ax.fill_between(xs_[sel], 0, np.exp(-xs_[sel] ** 2 / 2) / np.sqrt(2 * np.pi), color=ORANGE, alpha=0.25 - 0.1 * k)
    ax.text(0, 0.05 + 0.1 * (1 - k), f"±{lo_}σ: {txt}", ha="center", fontsize=9)
ax.set(xlabel="Ruido (en desviaciones estándar σ)", yticks=[], title="Distribución normal (gaussiana) del ruido")
ax.spines["left"].set_visible(False)
save(fig, "g11_ruido_gaussiano.png")
# ================================================================ G13 filtro pasa-bajas: una pasada frente a fase cero
from scipy.signal import butter, lfilter, filtfilt
m = (t240 >= t0 + 4.0) & (t240 <= t0 + 7.5)
tt, xx = t240[m], codo240[m]
ruido = np.random.default_rng(2).normal(0, 1.2, len(xx))          # vibración rápida añadida para ilustrar
xr = xx + ruido
b_, a_ = butter(4, 4.8, btype="low", fs=240)
una = lfilter(b_, a_, xr - xr[0]) + xr[0]
cero = filtfilt(b_, a_, xr)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
ax.plot(tt, xr, color=GRAY, lw=0.8, label="señal con vibraciones rápidas")
ax.plot(tt, cero, color=BLUE, lw=2.2, label="filtrada (fase cero)")
ax.set(xlabel="Tiempo (s)", ylabel="Flexión del codo (°)", title="1. El pasa-bajas quita las oscilaciones rápidas")
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
ax.plot(tt, xx, color=INK, lw=2.4, label="movimiento real")
ax.plot(tt, una, color=RED, lw=1.8, ls="--", label="filtro de una pasada: llega tarde")
ax.plot(tt, cero, color=BLUE, lw=1.8, label="fase cero (adelante y atrás): a tiempo")
k = np.argmax(np.gradient(xx))
ax.annotate("retraso", xy=(tt[k] + 0.06, xx[k]), xytext=(tt[k] + 0.45, xx[k] - 15), color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set(xlabel="Tiempo (s)", title="2. Una pasada desplaza la curva; la fase cero no")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Filtro pasa-bajas de 4.8 Hz (ilustrativo, con el codo real del sujeto 1 a 240 Hz)", y=1.02)
save(fig, "g13_filtro_fase_cero.png")
# ================================================================ G14 aliasing
tc = np.linspace(0, 1, 4000)
fs_ = 12
tm = np.arange(0, 1 + 1e-9, 1 / fs_)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(tc, np.sin(2 * np.pi * 10 * tc), color=GRAY, lw=1, label="vibración real: 10 Hz")
ax.plot(tm, np.sin(2 * np.pi * 10 * tm), "o", color=RED, ms=7, zorder=3, label="muestras a 12 Hz")
ax.plot(tc, np.sin(2 * np.pi * -2 * tc), color=RED, lw=2, ls="--", label="lo que parecen las muestras: 2 Hz")
ax.set(xlabel="Tiempo (s)", yticks=[], title="1. Sin filtro: una vibración de 10 Hz\nse disfraza de un movimiento de 2 Hz")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
ax = axes[1]
lento = np.sin(2 * np.pi * 0.5 * tc)
ax.plot(tc, lento + 0.4 * np.sin(2 * np.pi * 10 * tc), color=GRAY, lw=1, label="movimiento (0.5 Hz) + vibración (10 Hz)")
ax.plot(tm, np.sin(2 * np.pi * 0.5 * tm) + 0.4 * np.sin(2 * np.pi * 10 * tm), "o-", color=RED, ms=6, label="muestreo directo a 12 Hz: ondulación falsa")
ax.plot(tm, np.sin(2 * np.pi * 0.5 * tm), "o-", color=BLUE, ms=6, label="filtrar a 4.8 Hz y luego muestrear: correcto")
ax.set(xlabel="Tiempo (s)", yticks=[], title="2. Con el filtro pasa-bajas antes de bajar a 12 Hz,\nel aliasing desaparece")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
fig.suptitle("Aliasing (ilustrativo)", y=1.02)
save(fig, "g14_aliasing.png")
print("listo")
