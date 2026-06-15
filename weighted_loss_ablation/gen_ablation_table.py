"""
Gera tabela de ablação: YOLO26m com vs sem weighted loss.
Salva em figures/fig_ablation.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("/home/ceia-nuc-1/vizcomp/figures")

# ── dados ────────────────────────────────────────────────────────────────
#  (label, binary_w, binary_nw, onestage_w, onestage_nw)
ROWS = [
    # grupo, métrica, n_label, bin_w, bin_nw, os_w, os_nw
    # n_label: "bin_n | os_n" ou valor único se igual
    ("Geral",           "Acurácia (%)",       "494",    93.32, 92.51, 90.69, 89.88),
    ("Média\nMacro",    "Precisão (%)",       "—",      87.31, 84.25, 77.13, 75.01),
    ("Média\nMacro",    "Sensibilidade (%)",  "—",      81.79, 81.98, 69.57, 56.86),
    ("Média\nMacro",    "F1-Score (%)",       "—",      84.24, 83.06, 71.46, 63.13),
    ("Média\nMacro",    "AUC",                "—",       0.87,  0.89,  0.93,  0.91),
    ("Normal",          "Sensibilidade (%)",  "429",    97.44, 96.27, 96.74, 97.67),
    ("Anormal /\nOPMD", "Sensibilidade (%)",  "65 | 25",66.15, 67.69, 76.00, 52.00),
    ("Câncer\nOral",    "Sensibilidade (%)",  "— | 4",    "—",   "—", 75.00, 50.00),
    ("Variação\nNormal","Sensibilidade (%)",  "— | 36",   "—",   "—", 30.56, 27.78),
]

GREEN = "#2E7D32"
RED   = "#B71C1C"
GRAY  = "#455A64"
HEAD  = "#37474F"
ALT   = "#F5F5F5"

def fmt(v):
    if v == "—": return "—"
    return f"{v:.2f}" if isinstance(v, float) and v < 2 else f"{v:.2f}"

def delta_str(w, nw):
    if w == "—": return "—"
    d = w - nw
    s = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
    return s

def delta_color(w, nw):
    if w == "—": return GRAY
    return GREEN if (w - nw) >= 0 else RED

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")
fig.patch.set_facecolor("white")

# ── cabeçalho ─────────────────────────────────────────────────────────
#  0:grupo  1:métrica  2:n(val)  3:bw  4:bnw  5:bΔ  6:osw  7:osnw  8:osΔ
col_x = [0.00, 0.13, 0.24, 0.34, 0.44, 0.54, 0.64, 0.75, 0.85, 0.95, 1.05]

# linha de título de grupo (binária / one-stage)
Y_TOP = 0.97
ax.text((col_x[3]+col_x[5])/2 + 0.05, Y_TOP,
        "Binária (T₁)", ha="center", va="top",
        fontsize=11, fontweight="bold", color=HEAD)
ax.text((col_x[6]+col_x[8])/2 + 0.05, Y_TOP,
        "Multiclasse One-Stage (T₂)", ha="center", va="top",
        fontsize=11, fontweight="bold", color=HEAD)

# separadores verticais
SEP_Y0, SEP_Y1 = 0.00, 0.93
for xsep in [col_x[2], col_x[3], col_x[6], col_x[9]]:
    ax.plot([xsep, xsep], [SEP_Y0, SEP_Y1], color="#BDBDBD", lw=0.8, zorder=1)

# linha horizontal sob título
ax.plot([col_x[3], col_x[9]], [0.91, 0.91], color="#BDBDBD", lw=0.8)

# cabeçalhos das colunas
Y_HEAD = 0.87
col_labels = [
    (col_x[2], "n (val)"),
    (col_x[3], "Com\npeso"), (col_x[4], "Sem\npeso"), (col_x[5], "Δ"),
    (col_x[6], "Com\npeso"), (col_x[7], "Sem\npeso"), (col_x[8], "Δ"),
]
for i, (cx, lbl) in enumerate(col_labels):
    weight = "bold" if lbl == "Δ" else "normal"
    color  = GRAY if lbl == "n (val)" else HEAD
    ax.text(cx + 0.05, Y_HEAD, lbl, ha="center", va="top",
            fontsize=9, color=color, fontweight=weight, multialignment="center")

ax.plot([0, 1.05], [0.82, 0.82], color="#888888", lw=1.2)

# ── linhas de dados ────────────────────────────────────────────────────
ROW_H   = 0.082
Y_START = 0.80
prev_grupo = None

for row_i, (grupo, metrica, n_lbl, bw, bnw, osw, osnw) in enumerate(ROWS):
    y = Y_START - row_i * ROW_H

    # fundo alternado
    if row_i % 2 == 1:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y - ROW_H + 0.01), 1.05, ROW_H - 0.005,
            boxstyle="square,pad=0", fc=ALT, ec="none", zorder=0))

    # grupo (só quando muda)
    if grupo != prev_grupo:
        ax.text(col_x[0], y - ROW_H/2 + 0.01, grupo,
                ha="left", va="center", fontsize=8.5,
                color=HEAD, fontweight="bold", multialignment="center")
        prev_grupo = grupo

    # métrica
    ax.text(col_x[1], y - ROW_H/2 + 0.01, metrica,
            ha="left", va="center", fontsize=8.5, color="#333333")

    # n (val)
    ax.text(col_x[2]+0.05, y - ROW_H/2 + 0.01, n_lbl,
            ha="center", va="center", fontsize=8, color=GRAY)

    # binária
    ax.text(col_x[3]+0.05, y - ROW_H/2 + 0.01, fmt(bw),
            ha="center", va="center", fontsize=8.5,
            fontweight="bold" if bw != "—" and bw > bnw else "normal",
            color="#1a1a1a")
    ax.text(col_x[4]+0.05, y - ROW_H/2 + 0.01, fmt(bnw),
            ha="center", va="center", fontsize=8.5,
            fontweight="bold" if bnw != "—" and bnw > bw else "normal",
            color="#1a1a1a")
    dc = delta_color(bw, bnw)
    ax.text(col_x[5]+0.05, y - ROW_H/2 + 0.01, delta_str(bw, bnw),
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=dc)

    # one-stage
    ax.text(col_x[6]+0.05, y - ROW_H/2 + 0.01, fmt(osw),
            ha="center", va="center", fontsize=8.5,
            fontweight="bold" if osw != "—" and osw > osnw else "normal",
            color="#1a1a1a")
    ax.text(col_x[7]+0.05, y - ROW_H/2 + 0.01, fmt(osnw),
            ha="center", va="center", fontsize=8.5,
            fontweight="bold" if osnw != "—" and osnw > osw else "normal",
            color="#1a1a1a")
    dc2 = delta_color(osw, osnw)
    ax.text(col_x[8]+0.05, y - ROW_H/2 + 0.01, delta_str(osw, osnw),
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=dc2)

# linha final
bot = Y_START - len(ROWS) * ROW_H + 0.01
ax.plot([0, 1.05], [bot, bot], color="#888888", lw=1.0)

# nota de rodapé
ax.text(0, bot - 0.04,
        "n (val): instâncias no conjunto de teste por classe.  "
        "Anormal/OPMD: n binário | n multiclasse.  "
        "Δ = Com peso − Sem peso.  Verde: melhorou.  Vermelho: piorou.",
        ha="left", va="top", fontsize=7.5, color=GRAY, style="italic")

ax.set_xlim(0, 1.05)
ax.set_ylim(bot - 0.10, 1.02)

fig.tight_layout(pad=0.5)
path = OUT / "fig_ablation.pdf"
fig.savefig(path, bbox_inches="tight", facecolor="white")
print(f"Salvo: {path}")
plt.close(fig)
