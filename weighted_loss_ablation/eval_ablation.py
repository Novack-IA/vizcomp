"""
Compara YOLO26m com e sem weighted loss nas tarefas binary e one_stage.
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.classify.val import ClassificationValidator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

DATA_ROOT = Path("/home/ceia-nuc-1/vizcomp/data")
RUNS_ROOT = Path("/home/ceia-nuc-1/vizcomp/runs")

MODELS = {
    "binary": {
        "weighted":  RUNS_ROOT / "yolo26m_binary"          / "weights/best.pt",
        "noweight":  RUNS_ROOT / "yolo26m_binary_noweight"  / "weights/best.pt",
    },
    "one_stage": {
        "weighted":  RUNS_ROOT / "yolo26m_one_stage"         / "weights/best.pt",
        "noweight":  RUNS_ROOT / "yolo26m_one_stage_noweight" / "weights/best.pt",
    },
}


class _ProbaValidator(ClassificationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self._probs, self._targets = [], []

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
    model.val(validator=_Cap, data=str(data_dir), split="val",
              imgsz=320, batch=16, device=0, workers=0,
              verbose=False, plots=False, exist_ok=True)

    v = captured[0]
    names = [v.names[i] for i in sorted(v.names.keys())]
    y_true, y_prob = v.arrays()
    return y_true.astype(int), y_prob, names


def metrics(y_true, y_prob, names):
    n  = len(names)
    yp = np.argmax(y_prob, axis=1)
    acc      = accuracy_score(y_true, yp) * 100
    sens_mac = recall_score(y_true, yp, average="macro",  zero_division=0) * 100
    f1_mac   = f1_score    (y_true, yp, average="macro",  zero_division=0) * 100
    prec_mac = precision_score(y_true, yp, average="macro", zero_division=0) * 100
    auc = (roc_auc_score(y_true, y_prob[:, 1])
           if n == 2
           else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    sens_cls = recall_score(y_true, yp, average=None, zero_division=0) * 100
    return dict(acc=acc, prec_mac=prec_mac, sens_mac=sens_mac,
                f1_mac=f1_mac, auc=auc, sens_cls=sens_cls, names=names)


def print_comparison(task, w, nw):
    print(f"\n{'='*58}")
    print(f"  {task.upper()}")
    print(f"{'='*58}")
    print(f"{'Métrica':<28} {'Weighted':>10} {'No-Weight':>10} {'Δ':>8}")
    print(f"{'-'*58}")

    rows = [
        ("Acurácia (%)",        w["acc"],      nw["acc"]),
        ("Precisão macro (%)",  w["prec_mac"], nw["prec_mac"]),
        ("Sensibilidade macro (%)", w["sens_mac"], nw["sens_mac"]),
        ("F1-macro (%)",        w["f1_mac"],   nw["f1_mac"]),
        ("AUC",                 w["auc"],      nw["auc"]),
    ]
    for label, vw, vnw in rows:
        delta = vw - vnw
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<26} {vw:>10.2f} {vnw:>10.2f} {sign}{delta:>7.2f}")

    print(f"{'-'*58}")
    print("  Sensibilidade por classe:")
    for i, name in enumerate(w["names"]):
        vw  = w["sens_cls"][i]
        vnw = nw["sens_cls"][i]
        delta = vw - vnw
        sign  = "+" if delta >= 0 else ""
        print(f"    {name:<24} {vw:>10.2f} {vnw:>10.2f} {sign}{delta:>7.2f}")


for task, paths in MODELS.items():
    print(f"\nCarregando {task}...")
    yt_w, yp_w, names = run_val(paths["weighted"],  DATA_ROOT / task)
    yt_n, yp_n, _     = run_val(paths["noweight"],  DATA_ROOT / task)
    m_w  = metrics(yt_w, yp_w, names)
    m_nw = metrics(yt_n, yp_n, names)
    print_comparison(task, m_w, m_nw)

print()
