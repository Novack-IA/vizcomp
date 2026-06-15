"""
Ablação: treina YOLO26m sem weighted loss (treinamento padrão Ultralytics).
Roda binary e one_stage para comparar com os modelos ponderados.
"""

from pathlib import Path
from ultralytics import YOLO

DATA_ROOT = Path("/home/ceia-nuc-1/vizcomp/data")
RUNS_ROOT = Path("/home/ceia-nuc-1/vizcomp/runs")

CFG = dict(
    epochs    = 100,
    batch     = 32,
    imgsz     = 320,
    optimizer = "AdamW",
    lr0       = 1e-3,
    lrf       = 0.01,
    hsv_h     = 0.005,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    fliplr    = 0.5,
    degrees   = 10.0,
    translate = 0.1,
    scale     = 0.3,
    workers   = 8,
    device    = 0,
    amp       = False,
    verbose   = True,
    save      = True,
    plots     = True,
    exist_ok  = True,
)

for task in ["binary", "one_stage"]:
    print(f"\n{'='*60}\nTask: {task}\n{'='*60}")
    model = YOLO("yolo26m-cls.pt")
    model.train(
        data    = str(DATA_ROOT / task),
        name    = f"yolo26m_{task}_noweight",
        project = str(RUNS_ROOT),
        **CFG,
    )
