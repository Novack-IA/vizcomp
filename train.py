"""
Treina um modelo YOLO-cls com weighted CrossEntropyLoss.

Uso:
    python train.py --model yolov8m --task one_stage
    python train.py --model yolov11m --task binary
    python train.py --model yolo26m  --task stage2
"""

import argparse
import functools
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils.torch_utils import unwrap_model

DATA_ROOT = Path("/home/ceia-nuc-1/vizcomp/data")
RUNS_ROOT = Path("/home/ceia-nuc-1/vizcomp/runs")

TRAIN_CFG = dict(
    epochs    = 200,
    patience  = 50,
    batch     = 32,
    imgsz     = 320,
    optimizer = "AdamW",
    lr0       = 1e-3,
    lrf       = 0.01,
    # augmentation — reduz hue shift (cor é diagnóstica em lesões orais)
    hsv_h     = 0.005,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    fliplr    = 0.5,
    degrees   = 10.0,
    translate = 0.1,
    scale     = 0.3,
    workers   = 8,
    device    = 0,
    amp       = False,  # cuDNN sublibrary version mismatch com CUDA 13.0
    verbose   = True,
    save      = True,
    plots     = True,
    exist_ok  = True,
)

MODEL_WEIGHTS = {
    "yolov8m":  "yolov8m-cls.pt",
    "yolov11m": "yolo11m-cls.pt",
    "yolo26m":  "yolo26m-cls.pt",
}


def compute_class_weights(data_dir: Path) -> list[float]:
    """Frequência inversa normalizada (média=1) sobre o conjunto de treino."""
    train_dir  = data_dir / "train"
    class_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir())
    counts     = [len(list(d.iterdir())) for d in class_dirs]
    n_total    = sum(counts)
    n_cls      = len(counts)
    weights    = [n_total / (n_cls * c) for c in counts]
    mean_w     = sum(weights) / len(weights)
    weights    = [w / mean_w for w in weights]
    print("Pesos de classe (ordem alfabética):")
    for d, c, w in zip(class_dirs, counts, weights):
        print(f"  {d.name}: n={c}  w={w:.4f}")
    return weights


class WeightedClassificationLoss:
    """
    Replica a interface de v8ClassificationLoss com CrossEntropyLoss ponderada.
    O Ultralytics chama criterion(preds, batch) onde batch é um dict;
    precisamos extrair batch["cls"] manualmente.
    """

    def __init__(self, weight: torch.Tensor, label_smoothing: float = 0.0):
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = self.loss_fn(preds, batch["cls"])
        return loss, loss.detach()


class WeightedCLSTrainer(ClassificationTrainer):
    """
    ClassificationTrainer com CrossEntropyLoss ponderada.

    Os pesos são injetados via set_class_weights(), que é o hook
    oficial do Ultralytics chamado após o model estar no device e
    antes do loop de treino.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, class_weights=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self._class_weights = class_weights

    def set_class_weights(self):
        if self._class_weights is None:
            return
        w = torch.tensor(self._class_weights, dtype=torch.float32).to(self.device)
        label_smoothing = getattr(self.args, "label_smoothing", 0.0)
        unwrap_model(self.model).criterion = WeightedClassificationLoss(w, label_smoothing)
        print(f"WeightedCrossEntropyLoss: {[f'{x:.3f}' for x in self._class_weights]}")


def train(model_name: str, task: str):
    data_dir   = DATA_ROOT / task
    weights_pt = MODEL_WEIGHTS[model_name]
    run_name   = f"{model_name}_{task}"

    print(f"\n{'='*60}")
    print(f"Modelo: {model_name}   Task: {task}")
    print(f"Dataset: {data_dir}")
    print(f"{'='*60}\n")

    class_weights = compute_class_weights(data_dir)

    # functools.partial injeta class_weights no __init__ do trainer.
    # O Ultralytics instancia o trainer como:
    #   trainer(overrides=args, _callbacks=callbacks)
    # O partial transforma isso em:
    #   WeightedCLSTrainer(overrides=args, _callbacks=callbacks, class_weights=class_weights)
    TrainerCls = functools.partial(WeightedCLSTrainer, class_weights=class_weights)

    model = YOLO(weights_pt)
    model.train(
        trainer = TrainerCls,
        data    = str(data_dir),
        name    = run_name,
        project = str(RUNS_ROOT),
        **TRAIN_CFG,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_WEIGHTS.keys()), required=True)
    parser.add_argument("--task", choices=["one_stage", "binary", "stage2"], required=True)
    args = parser.parse_args()
    train(args.model, args.task)


if __name__ == "__main__":
    main()
