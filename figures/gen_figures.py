"""
Gera as figuras para o artigo:
  Fig 1 — Comparação de sensibilidade macro (nossos modelos vs baseline)
  Fig 2 — Matriz de confusão do YOLO26m na tarefa multiclasse (4 classes)
  Fig 3 — Matriz de confusão do YOLO26m na tarefa binária
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationValidator

# ------------------------------------------------------------------ #
DATA_ROOT  = Path("/home/ceia-nuc-1/vizcomp/data")
RUNS_ROOT  = Path("/home/ceia-nuc-1/vizcomp/runs")
OUT        = Path("/home/ceia-nuc-1/vizcomp/figures")
IMGSZ      = 320
VAL_BATCH  = 16
# ------------------------------------------------------------------ #

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


# ───────────────────────────────────────────────────────────────────
# Validator com captura de probabilidades
# ───────────────────────────────────────────────────────────────────
class _ProbaValidator(ClassificationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self._probs   = []
        self._targets = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        self._probs.append(preds.detach().cpu().float().numpy())
        self._targets.append(batch["cls"].cpu().numpy())

    def arrays(self):
        return np.concatenate(self._targets), np.concatenate(self._probs)


def run_val(weights_path, data_dir):
    captured = []

    class _Cap(_ProbaValidator):
        def finalize_metrics(self, *a, **kw):
            super().finalize_metrics(*a, **kw)
            captured.append(self)

    model = YOLO(str(weights_path))
    model.val(
        validator=_Cap,
        data=str(data_dir),
        split="val",
        imgsz=IMGSZ,
        batch=VAL_BATCH,
        device=0,
        workers=0,
        verbose=False,
        plots=False,
        exist_ok=True,
    )
    v = captured[0]
    class_names = [v.names[i] for i in sorted(v.names.keys())]
    y_true, y_prob = v.arrays()
    y_pred = y_prob.argmax(axis=1)
    return y_true.astype(int), y_pred.astype(int), class_names


# ───────────────────────────────────────────────────────────────────
# Fig 1 — Barras: sensibilidade macro (nossos vs baseline)
# ───────────────────────────────────────────────────────────────────
def fig_sensitivity_bars():
    models   = ["YOLOv8m", "YOLOv11m", "YOLO26m"]
    our_bin  = [79.14,  79.67,  81.79]   # macro sensitivity binária
    our_mc   = [57.06,  52.94,  69.57]   # macro sensitivity multiclasse
    base_bin = 79.94   # melhor baseline binário (ResNet18, macro sensitivity)
    base_mc  = 61.57   # melhor baseline multiclasse (macro sensitivity)

    x = np.arange(len(models))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, vals, base, title, ylabel in [
        (axes[0], our_bin, base_bin,
         "(a) Detecção Binária (T₁)", "Sensibilidade Macro (%)"),
        (axes[1], our_mc,  base_mc,
         "(b) Classificação Multiclasse (T₂)", ""),
    ]:
        bars = ax.bar(x, vals, width=w*2, color=["#4C72B0","#55A868","#C44E52"],
                      zorder=3, edgecolor="white", linewidth=0.5)
        ax.axhline(base, color="black", linestyle="--", linewidth=1.4,
                   label=f"Melhor baseline ({base:.2f}%)")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(title, pad=8)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.set_ylim(40, 95)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout()
    path = OUT / "fig_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Salvo: {path}")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
# Fig 2 e 3 — Matrizes de confusão
# ───────────────────────────────────────────────────────────────────
def fig_confusion(y_true, y_pred, class_names, title, filename):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(4 + 0.6*n, 3.5 + 0.5*n))

    # Anotação: "N\n(XX%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(n):
        for j in range(n):
            annot[i, j] = f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)"

    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"shrink": 0.7, "format": mticker.PercentFormatter(xmax=1, decimals=0)},
    )
    ax.set_xlabel("Predito", labelpad=8)
    ax.set_ylabel("Real", labelpad=8)
    ax.set_title(title, pad=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"Salvo: {path}")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Fig 1: barras de sensibilidade ===")
    fig_sensitivity_bars()

    print("\n=== Fig 2: confusão YOLO26m — multiclasse ===")
    y_true, y_pred, names = run_val(
        RUNS_ROOT / "yolo26m_one_stage" / "weights" / "best.pt",
        DATA_ROOT / "one_stage",
    )
    # nomes mais curtos para o plot
    short = {"normal": "Normal", "variation": "Variação",
             "opmd": "OPMD", "oral_cancer": "Câncer Oral"}
    names_short = [short.get(n, n) for n in names]
    fig_confusion(y_true, y_pred, names_short,
                  "YOLO26m-cls — Classificação Multiclasse",
                  "fig_confusion_multiclass.pdf")

    print("\n=== Fig 3: confusão YOLO26m — binária ===")
    y_true_b, y_pred_b, names_b = run_val(
        RUNS_ROOT / "yolo26m_binary" / "weights" / "best.pt",
        DATA_ROOT / "binary",
    )
    short_b = {"normal": "Normal", "abnormal": "Anormal"}
    names_b_short = [short_b.get(n, n) for n in names_b]
    fig_confusion(y_true_b, y_pred_b, names_b_short,
                  "YOLO26m-cls — Detecção Binária",
                  "fig_confusion_binary.pdf")

    print("\nFiguras geradas em:", OUT)
