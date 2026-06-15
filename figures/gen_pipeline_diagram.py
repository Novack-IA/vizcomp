"""
Gera diagrama das pipelines one-stage e two-stage.
Salva em figures/fig_pipeline.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("/home/ceia-nuc-1/vizcomp/figures")

C_INPUT  = "#4C72B0"
C_MODEL  = "#3A7D44"
C_NORMAL = "#78909C"
C_LESION = "#7B1A1A"
C_OUT    = "#37474F"
CARR     = "#444444"
BG       = "white"

def box(ax, cx, cy, w, h, text, fc, fs=9.5, tc="white"):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=fc, edgecolor="#aaaaaa", linewidth=1.2, zorder=3,
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold",
            zorder=4, multialignment="center", linespacing=1.35)

def arr(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=CARR,
                    lw=1.5, mutation_scale=13,
                    connectionstyle="arc3,rad=0",
                ), zorder=2)

def blabel(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=8, color="#333333", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

# ── figura ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.5))
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor(BG)

# ════════════════════════════════════════════════════════════════════════
# ONE-STAGE
# ════════════════════════════════════════════════════════════════════════
ax1.text(0.5, 0.96, "One-Stage", ha="center", va="top",
         fontsize=13, fontweight="bold", color="#1a1a1a")

#  Input
box(ax1, 0.5, 0.83, 0.42, 0.09, "Imagem Intraoral", C_INPUT, 10)
arr(ax1, 0.5, 0.785, 0.5, 0.695)

# Model
box(ax1, 0.5, 0.65, 0.46, 0.09, "YOLO-cls  (4 classes)", C_MODEL, 10)

# 4 outputs — x coords chosen so all boxes stay in [0.03, 0.97]
# w=0.20 → half=0.10 | centers: 0.14, 0.38, 0.63, 0.87
OXS  = [0.14, 0.38, 0.63, 0.87]
OY   = 0.375
OW   = 0.20
OH   = 0.10
OLBL = ["Normal", "Variação\ndo Normal", "OPMD", "Câncer\nOral"]
OCOL = [C_NORMAL, C_OUT, C_OUT, C_LESION]

# hub connector: vertical stem + horizontal bar + individual drops
HUB_Y = 0.54   # y of horizontal bus
ax1.plot([0.5, 0.5], [0.605, HUB_Y], color=CARR, lw=1.5, zorder=2)
ax1.plot([OXS[0], OXS[-1]], [HUB_Y, HUB_Y], color=CARR, lw=1.5, zorder=2)
for cx in OXS:
    ax1.plot([cx, cx], [HUB_Y, OY + OH/2 + 0.012], color=CARR, lw=1.5, zorder=2)
    arr(ax1, cx, OY + OH/2 + 0.015, cx, OY + OH/2)

for cx, lbl, col in zip(OXS, OLBL, OCOL):
    box(ax1, cx, OY, OW, OH, lbl, col, 8.5)

ax1.text(0.5, 0.09, "Predição final em uma única etapa",
         ha="center", fontsize=8.5, color="#777777", style="italic")

# ════════════════════════════════════════════════════════════════════════
# TWO-STAGE
# ════════════════════════════════════════════════════════════════════════
ax2.text(0.5, 0.96, "Two-Stage", ha="center", va="top",
         fontsize=13, fontweight="bold", color="#1a1a1a")

# Input
box(ax2, 0.5, 0.83, 0.42, 0.09, "Imagem Intraoral", C_INPUT, 10)
arr(ax2, 0.5, 0.785, 0.5, 0.695)

# Stage 1
box(ax2, 0.5, 0.65, 0.56, 0.09,
    "Estágio 1 — YOLO-cls\n(Normal  vs  Anormal)", C_MODEL, 9.5)

# Branch hub from Stage1
S1B   = 0.605   # stage1 bottom
HUB2  = 0.565   # hub y
NX    = 0.20    # Normal cx
S2X   = 0.68    # Stage2 cx

ax2.plot([0.5, 0.5],  [S1B,   HUB2], color=CARR, lw=1.5, zorder=2)
ax2.plot([NX,  S2X],  [HUB2,  HUB2], color=CARR, lw=1.5, zorder=2)

# Left drop → Normal
ax2.plot([NX, NX], [HUB2, 0.505], color=CARR, lw=1.5, zorder=2)
arr(ax2, NX, 0.507, NX, 0.500)
blabel(ax2, 0.30, HUB2 + 0.028, "Normal")

# Right drop → Stage 2
ax2.plot([S2X, S2X], [HUB2, 0.505], color=CARR, lw=1.5, zorder=2)
arr(ax2, S2X, 0.507, S2X, 0.500)
blabel(ax2, 0.58, HUB2 + 0.028, "Anormal")

# Normal box (encerra)
box(ax2, NX, 0.455, 0.30, 0.09, "Normal\n(encerra)", C_NORMAL, 9.5)

# Stage 2 box
box(ax2, S2X, 0.455, 0.44, 0.09,
    "Estágio 2 — YOLO-cls\n(3 classes)", C_MODEL, 9.5)

# 3 outputs from Stage2
# w=0.20 | centers centered on S2X=0.68: 0.68-0.24=0.44, 0.68, 0.68+0.24=0.92
# rightmost right edge: 0.92+0.10=1.02 → shift left: 0.42, 0.66, 0.90 → 0.90+0.10=1.00 ✓
O2XS  = [0.42, 0.66, 0.90]
O2Y   = 0.23
O2W   = 0.21
O2H   = 0.095
O2LBL = ["Variação\ndo Normal", "OPMD", "Câncer\nOral"]

HUB3  = 0.36   # hub y for stage2 outputs
S2B   = 0.410  # stage2 bottom

ax2.plot([S2X, S2X],    [S2B,  HUB3], color=CARR, lw=1.5, zorder=2)
ax2.plot([O2XS[0], O2XS[-1]], [HUB3, HUB3], color=CARR, lw=1.5, zorder=2)
for cx in O2XS:
    ax2.plot([cx, cx], [HUB3, O2Y + O2H/2 + 0.012], color=CARR, lw=1.5, zorder=2)
    arr(ax2, cx, O2Y + O2H/2 + 0.015, cx, O2Y + O2H/2)

for cx, lbl in zip(O2XS, O2LBL):
    box(ax2, cx, O2Y, O2W, O2H, lbl, C_LESION, 8.5)

ax2.text(0.5, 0.07,
         "★  Estágio 1 otimizado para alta sensibilidade\n(minimizar falsos negativos)",
         ha="center", fontsize=8, color="#B71C1C", style="italic")

# ── salvar ──────────────────────────────────────────────────────────────
fig.tight_layout(pad=2.0)
path = OUT / "fig_pipeline.pdf"
fig.savefig(path, bbox_inches="tight", facecolor=BG)
print(f"Salvo: {path}")
plt.close(fig)
