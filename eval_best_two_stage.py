"""
Avalia o pipeline two-stage com a melhor combinação:
  Stage 1 (binário):    YOLO26m
  Stage 2 (3 classes):  YOLOv8m
Avaliação end-to-end no val set do one_stage (4 classes).
"""

import numpy as np
from pathlib import Path
import torch
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

DATA_ROOT = Path("/home/ceia-nuc-1/vizcomp/data")
RUNS_ROOT = Path("/home/ceia-nuc-1/vizcomp/runs")

LABEL4    = ["normal", "opmd", "oral_cancer", "variation"]  # ordem alfabética
LABEL3    = ["opmd", "oral_cancer", "variation"]
IMGSZ     = 320

# ── transform idêntico ao treino ─────────────────────────────────────────
try:
    from ultralytics.data.augment import classify_transforms
    _transform = classify_transforms(IMGSZ)
except Exception:
    import torchvision.transforms as T
    def _cc(img):
        w, h = img.size; m = min(w, h)
        return img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2)).resize((IMGSZ, IMGSZ), Image.BICUBIC)
    _transform = T.Compose([T.Lambda(_cc), T.ToTensor(),
                             T.Normalize([.485,.456,.406],[.229,.224,.225])])

def infer(torch_model, device, path):
    img = Image.open(str(path)).convert("RGB")
    t   = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = torch_model(t)
    if isinstance(out, (list, tuple)): out = out[0]
    return torch.softmax(out, -1)[0].cpu().numpy()

# ── carregar modelos ──────────────────────────────────────────────────────
print("Carregando Stage 1: YOLO26m binary...")
m1 = YOLO(str(RUNS_ROOT / "yolo26m_binary"  / "weights/best.pt"))
print("Carregando Stage 2: YOLOv8m stage2...")
m2 = YOLO(str(RUNS_ROOT / "yolov8m_stage2"  / "weights/best.pt"))

dev   = torch.device("cuda:0")
t1    = m1.model.eval().to(dev)
t2    = m2.model.eval().to(dev)
n1    = m1.names   # {0: 'abnormal', 1: 'normal'} ou similar
n2    = m2.names   # {0: 'opmd', 1: 'oral_cancer', 2: 'variation'} ordem alfabética

# mapa: índice stage2 → índice global (LABEL4)
l4idx = {c: i for i, c in enumerate(LABEL4)}
s2map = {j: l4idx[n2[j]] for j in n2}

# ── inferência e2e ────────────────────────────────────────────────────────
val_dir = DATA_ROOT / "one_stage" / "val"
y_true, y_prob = [], []
normal_idx = l4idx["normal"]

routed_to_s2 = 0
for cls_dir in sorted(d for d in val_dir.iterdir() if d.is_dir()):
    true_idx = l4idx[cls_dir.name]
    for img_path in sorted(cls_dir.iterdir()):
        p1 = infer(t1, dev, img_path)
        lbl1 = n1[int(np.argmax(p1))]

        prob4 = np.zeros(4)
        if lbl1 == "normal":
            prob4[normal_idx] = 1.0
        else:
            routed_to_s2 += 1
            p2 = infer(t2, dev, img_path)
            for j, gidx in s2map.items():
                prob4[gidx] = p2[j]

        y_true.append(true_idx)
        y_prob.append(prob4)

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = np.argmax(y_prob, axis=1)

print(f"\nImagens roteadas ao Stage 2: {routed_to_s2}/{len(y_true)}")

# ── métricas ──────────────────────────────────────────────────────────────
def spec_per_class(yt, yp, n):
    out = []
    for c in range(n):
        tn = np.sum((yp != c) & (yt != c))
        fp = np.sum((yp == c) & (yt != c))
        out.append(tn / (tn + fp) * 100 if (tn+fp) > 0 else 0.0)
    return out

n = len(LABEL4)
acc      = accuracy_score(y_true, y_pred) * 100
sens_mac = recall_score(y_true, y_pred, average="macro",  zero_division=0) * 100
prec_mac = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
f1_mac   = f1_score(y_true, y_pred,   average="macro",  zero_division=0) * 100
spec_mac = np.mean(spec_per_class(y_true, y_pred, n))
auc_mac  = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

sens_cls = recall_score(y_true, y_pred, average=None, zero_division=0) * 100
prec_cls = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
f1_cls   = f1_score(y_true, y_pred,   average=None, zero_division=0) * 100
spec_cls = spec_per_class(y_true, y_pred, n)

print(f"\n{'='*52}")
print(f"  Two-Stage (YOLO26m binário + YOLOv8m stage2)")
print(f"{'='*52}")
print(f"  Acurácia           {acc:.2f}%")
print(f"  Precisão macro     {prec_mac:.2f}%")
print(f"  Sensibilidade mac  {sens_mac:.2f}%")
print(f"  Especificidade mac {spec_mac:.2f}%")
print(f"  F1-macro           {f1_mac:.2f}%")
print(f"  AUC macro          {auc_mac:.3f}")
print(f"\n  Por classe:")
for i, cls in enumerate(LABEL4):
    print(f"    {cls:15}  sens={sens_cls[i]:.2f}%  prec={prec_cls[i]:.2f}%  "
          f"spec={spec_cls[i]:.2f}%  f1={f1_cls[i]:.2f}%")

# matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print(f"\n  Matriz de confusão (linhas=real, colunas=predito):")
print(f"  {'':15} " + "  ".join(f"{c:10}" for c in LABEL4))
for i, cls in enumerate(LABEL4):
    print(f"  {cls:15} " + "  ".join(f"{cm[i,j]:10}" for j in range(n)))
