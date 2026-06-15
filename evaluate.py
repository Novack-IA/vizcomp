"""
Avalia modelos treinados e reporta todas as métricas do paper:
  - Acurácia
  - Precisão, Sensibilidade, Especificidade, F1 (macro + micro + por classe)
  - AUC-ROC (macro + por classe)

Usa ClassificationValidator do Ultralytics para preprocessing idêntico ao treino.
ProbaValidator captura as probabilidades por amostra (necessário pro AUC).

Uso:
    python evaluate.py                              # avalia todos
    python evaluate.py --model yolov8m --task binary
    python evaluate.py --two-stage --model yolov8m
"""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from ultralytics import YOLO
from ultralytics.models.yolo.classify.val import ClassificationValidator

DATA_ROOT   = Path("/home/ceia-nuc-1/vizcomp/data")
RUNS_ROOT   = Path("/home/ceia-nuc-1/vizcomp/runs")
RESULTS_DIR = Path("/home/ceia-nuc-1/vizcomp/results")

MODELS = ["yolov8m", "yolov11m", "yolo26m"]
TASKS  = ["one_stage", "binary", "stage2"]

LABEL4 = ["normal", "variation", "opmd", "oral_cancer"]
LABEL3 = ["variation", "opmd", "oral_cancer"]

IMGSZ     = 320
VAL_BATCH = 16   # sem gradients, seguro com batches maiores


# ── Validator com captura de probabilidades ────────────────────────────────────

class _ProbaValidator(ClassificationValidator):
    """
    Subclasse de ClassificationValidator que armazena probabilidades completas
    por amostra em self._probs / self._targets (necessário para AUC e métricas
    por classe além das que o Ultralytics reporta nativamente).
    """

    def init_metrics(self, model):
        super().init_metrics(model)
        self._probs: list[np.ndarray]   = []
        self._targets: list[np.ndarray] = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        # preds: (N, C) tensor de probabilidades após softmax
        self._probs.append(preds.detach().cpu().float().numpy())
        self._targets.append(batch["cls"].cpu().numpy())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self._targets), np.concatenate(self._probs)


def _run_val(
    model: YOLO,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Executa model.val() com _ProbaValidator e retorna (y_true, y_prob, class_names).
    Usa o mesmo preprocessing que o treino (CenterCrop Ultralytics + ImageNet norm).
    """
    captured: list[_ProbaValidator] = []

    class _CapturingValidator(_ProbaValidator):
        def finalize_metrics(self, *args, **kwargs):
            super().finalize_metrics(*args, **kwargs)
            captured.append(self)

    model.val(
        validator=_CapturingValidator,
        data=str(data_dir),
        split="val",
        imgsz=IMGSZ,
        batch=VAL_BATCH,
        device=0,
        workers=0,   # evita ConnectionResetError no thread pin_memory ao fechar subprocess
        verbose=False,
        plots=False,
        exist_ok=True,
    )

    v = captured[0]
    class_names = [v.names[i] for i in sorted(v.names.keys())]
    y_true, y_prob = v.arrays()
    return y_true, y_prob, class_names


# ── Transforms para inferência imagem-a-imagem (two-stage) ────────────────────

def _get_classify_transform(imgsz: int):
    """
    Retorna o transform de validação do Ultralytics:
    CenterCrop (crop quadrado central → resize) + ToTensor + ImageNet norm.
    Compatível com classify_transforms() da ultralytics.data.augment.
    """
    try:
        from ultralytics.data.augment import classify_transforms
        return classify_transforms(imgsz)
    except ImportError:
        pass
    try:
        from ultralytics.data.utils import classify_transforms
        return classify_transforms(imgsz)
    except ImportError:
        pass

    # fallback manual — replica exatamente o CenterCrop do Ultralytics:
    # crop quadrado central (min_dim × min_dim) → resize BICUBIC → ToTensor → Normalize
    import torchvision.transforms as T

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def _center_crop_resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        m = min(h, w)
        top, left = (h - m) // 2, (w - m) // 2
        return img.crop((left, top, left + m, top + m)).resize((imgsz, imgsz), Image.BICUBIC)

    return T.Compose([
        T.Lambda(_center_crop_resize),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


_TRANSFORM = _get_classify_transform(IMGSZ)


def _infer_image(torch_model, device, img_path: Path) -> np.ndarray:
    """Retorna vetor de probabilidades (C,) para uma imagem."""
    img = Image.open(str(img_path)).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = torch_model(tensor)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.softmax(out, dim=-1)[0].cpu().numpy()


# ── Métricas ───────────────────────────────────────────────────────────────────

def _specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> list[float]:
    specs = []
    for c in range(n):
        tn = np.sum((y_pred != c) & (y_true != c))
        fp = np.sum((y_pred == c) & (y_true != c))
        specs.append(tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0)
    return specs


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    n   = len(class_names)
    yp  = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, yp) * 100

    prec_mac  = precision_score(y_true, yp, average="macro",  zero_division=0) * 100
    sens_mac  = recall_score   (y_true, yp, average="macro",  zero_division=0) * 100
    f1_mac    = f1_score       (y_true, yp, average="macro",  zero_division=0) * 100
    spec_mac  = np.mean(_specificity_per_class(y_true, yp, n))

    prec_mic  = precision_score(y_true, yp, average="micro",  zero_division=0) * 100
    sens_mic  = recall_score   (y_true, yp, average="micro",  zero_division=0) * 100
    f1_mic    = f1_score       (y_true, yp, average="micro",  zero_division=0) * 100
    spec_mic  = acc  # micro specificity == accuracy para multiclasse

    try:
        auc_mac = (
            roc_auc_score(y_true, y_prob[:, 1])
            if n == 2
            else roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except Exception:
        auc_mac = float("nan")

    prec_cls = precision_score(y_true, yp, average=None, zero_division=0) * 100
    sens_cls = recall_score   (y_true, yp, average=None, zero_division=0) * 100
    f1_cls   = f1_score       (y_true, yp, average=None, zero_division=0) * 100
    spec_cls = _specificity_per_class(y_true, yp, n)

    try:
        auc_cls = roc_auc_score(
            y_true,
            y_prob if n > 2 else y_prob[:, 1],
            multi_class="ovr" if n > 2 else "raise",
            average=None,
        )
    except Exception:
        auc_cls = [float("nan")] * n

    rows: list[dict] = []
    rows.append({"group": "overall", "metric": "accuracy",    "value": round(acc,      2)})
    rows.append({"group": "macro",   "metric": "precision",   "value": round(prec_mac, 2)})
    rows.append({"group": "macro",   "metric": "sensitivity", "value": round(sens_mac, 2)})
    rows.append({"group": "macro",   "metric": "specificity", "value": round(spec_mac, 2)})
    rows.append({"group": "macro",   "metric": "f1_score",    "value": round(f1_mac,   2)})
    rows.append({"group": "macro",   "metric": "auc",         "value": round(auc_mac,  2)})
    rows.append({"group": "micro",   "metric": "precision",   "value": round(prec_mic, 2)})
    rows.append({"group": "micro",   "metric": "sensitivity", "value": round(sens_mic, 2)})
    rows.append({"group": "micro",   "metric": "specificity", "value": round(spec_mic, 2)})
    rows.append({"group": "micro",   "metric": "f1_score",    "value": round(f1_mic,   2)})

    for i, cls in enumerate(class_names):
        g = f"class_{cls}"
        v_auc = float(auc_cls[i]) if not isinstance(auc_cls, float) else auc_cls
        rows.append({"group": g, "metric": "precision",   "value": round(float(prec_cls[i]), 2)})
        rows.append({"group": g, "metric": "sensitivity", "value": round(float(sens_cls[i]), 2)})
        rows.append({"group": g, "metric": "specificity", "value": round(float(spec_cls[i]), 2)})
        rows.append({"group": g, "metric": "f1_score",    "value": round(float(f1_cls[i]),   2)})
        rows.append({"group": g, "metric": "auc",         "value": round(v_auc,              2)})

    return {"rows": rows}


# ── Avaliação single-task ──────────────────────────────────────────────────────

def evaluate_single(
    model_name: str,
    task: str,
    output_json: Path | None = None,
) -> pd.DataFrame:
    print(f"\nAvaliando {model_name} / {task} ...")
    model = load_model(model_name, task)
    y_true, y_prob, class_names = _run_val(model, DATA_ROOT / task)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    metrics = compute_metrics(y_true, y_prob, class_names)
    df = pd.DataFrame(metrics["rows"])
    df.insert(0, "model", model_name)
    df.insert(1, "task",  task)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(df.to_dict("records"), indent=2))
        print(f"  Salvo em {output_json}")

    return df


# ── Avaliação two-stage E2E ────────────────────────────────────────────────────

def evaluate_two_stage_e2e(
    model_name: str,
    output_json: Path | None = None,
) -> pd.DataFrame:
    """
    Pipeline two-stage end-to-end.
    Stage 1 (binary) → se abnormal, Stage 2 (3 classes) → label final.
    Inferência imagem-a-imagem com os transforms corretos do Ultralytics.
    """
    print(f"\nAvaliando two-stage E2E: {model_name} ...")
    model_bin    = load_model(model_name, "binary")
    model_stage2 = load_model(model_name, "stage2")

    torch_bin    = model_bin.model.eval()
    torch_stage2 = model_stage2.model.eval()
    device = next(torch_bin.parameters()).device

    bin_names = model_bin.names
    s2_names  = model_stage2.names
    label2idx = {c: i for i, c in enumerate(LABEL4)}
    s2_to_global = {n: label2idx[n] for n in LABEL3}

    val_dir = DATA_ROOT / "one_stage" / "val"
    y_true_all, y_prob_all = [], []

    for cls_dir in sorted(d for d in val_dir.iterdir() if d.is_dir()):
        true_idx = label2idx[cls_dir.name]
        for img_path in sorted(cls_dir.iterdir()):
            bin_probs = _infer_image(torch_bin, device, img_path)
            bin_label = bin_names[int(np.argmax(bin_probs))]

            prob4 = np.zeros(4)
            if bin_label == "normal":
                prob4[label2idx["normal"]] = 1.0
            else:
                s2_probs = _infer_image(torch_stage2, device, img_path)
                for j, name in s2_names.items():
                    prob4[s2_to_global[name]] = s2_probs[j]

            y_true_all.append(true_idx)
            y_prob_all.append(prob4)

    del model_bin, model_stage2
    gc.collect()
    torch.cuda.empty_cache()

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)
    metrics = compute_metrics(y_true, y_prob, LABEL4)

    df = pd.DataFrame(metrics["rows"])
    df.insert(0, "model", model_name)
    df.insert(1, "task",  "two_stage_e2e")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(df.to_dict("records"), indent=2))
        print(f"  Salvo em {output_json}")

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(model_name: str, task: str) -> YOLO:
    run_dir = RUNS_ROOT / f"{model_name}_{task}"
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        candidates = list(run_dir.rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError(f"Não encontrei best.pt em {run_dir}")
        best = candidates[0]
    return YOLO(best)


def _run_subprocess(extra_args: list[str], output_json: Path) -> bool:
    cmd = [sys.executable, __file__] + extra_args + ["--output", str(output_json)]
    print(f"\n{'='*60}")
    print(f"Subprocesso: {' '.join(cmd[2:])}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(f"  ERRO (código {ret.returncode}) — pulando")
        return False
    return True


def _combine_jsons(json_dir: Path) -> pd.DataFrame:
    dfs = []
    for f in sorted(json_dir.glob("*.json")):
        dfs.append(pd.DataFrame(json.loads(f.read_text())))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     choices=MODELS,                    default=None)
    parser.add_argument("--task",      choices=TASKS + ["two_stage_e2e"], default=None)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--output",    type=Path,                         default=None,
                        help="Salva resultado como JSON (subprocesso interno)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_dir = RESULTS_DIR / "runs"
    json_dir.mkdir(parents=True, exist_ok=True)

    # Modo subprocesso: roda UMA avaliação e sai
    if args.output:
        if args.two_stage:
            evaluate_two_stage_e2e(args.model, output_json=args.output)
        else:
            evaluate_single(args.model, args.task, output_json=args.output)
        return

    # Modo direto: model + task especificados (sem subprocess)
    if args.model and (args.task or args.two_stage):
        if args.two_stage:
            df = evaluate_two_stage_e2e(args.model)
        else:
            df = evaluate_single(args.model, args.task)
        out = RESULTS_DIR / "metrics.csv"
        df.to_csv(out, index=False)
        print(f"\nResultados salvos em {out}")
        print(df.to_string(index=False))
        return

    # Modo completo: spawna subprocesso por combo para isolar VRAM
    print("Avaliando todos os modelos via subprocessos isolados...\n")

    for model_name in MODELS:
        for task in TASKS:
            out_json = json_dir / f"{model_name}_{task}.json"
            if out_json.exists():
                print(f"  Pulando {model_name}/{task} (já existe)")
                continue
            _run_subprocess(["--model", model_name, "--task", task], out_json)

    for model_name in MODELS:
        out_json = json_dir / f"{model_name}_two_stage_e2e.json"
        if out_json.exists():
            print(f"  Pulando two-stage {model_name} (já existe)")
            continue
        _run_subprocess(["--two-stage", "--model", model_name], out_json)

    # Combina tudo
    final = _combine_jsons(json_dir)
    if final.empty:
        print("Nenhum resultado encontrado.")
        return

    out = RESULTS_DIR / "metrics.csv"
    final.to_csv(out, index=False)
    print(f"\nResultados salvos em {out}")

    summary = (
        final[final["group"].isin(["overall", "macro", "micro"])]
        .pivot_table(index=["model", "task", "group"], columns="metric", values="value")
        .reset_index()
    )
    summary_out = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Resumo salvo em {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
