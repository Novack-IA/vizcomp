"""
Prepara as 3 estruturas de dataset para o Ultralytics cls:
  - data/one_stage/   → 4 classes (normal, variation, opmd, oral_cancer)
  - data/binary/      → 2 classes (normal, abnormal)
  - data/stage2/      → 3 classes (variation, opmd, oral_cancer)

Split: 80% train / 20% val (seed=42, sem estratificação por paciente).
Oversampling das classes minoritárias no train por duplicação aleatória.
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter

SEED = 42
SPLIT = 0.8
DATASET_ROOT = Path("/home/ceia-nuc-1/vizcomp/SMART-OM")
OUTPUT_ROOT  = Path("/home/ceia-nuc-1/vizcomp/data")

# Alvo de oversampling no train (mínimo de imagens por classe)
OVERSAMPLE_TARGET = {
    "normal":      1716,   # já tem suficiente
    "variation":    300,
    "opmd":         300,
    "oral_cancer":  160,
}

CLASS_DIRS = {
    "normal":      DATASET_ROOT / "01. Normal"      / "01. Unannotated",
    "variation":   DATASET_ROOT / "02. Variation from normal" / "01. Unannotated",
    "opmd":        DATASET_ROOT / "03. OPMD"        / "01. Unannotated",
    "oral_cancer": DATASET_ROOT / "04. Oral Cancer" / "01. Unannotated",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def collect_images(class_dir: Path) -> list[Path]:
    imgs = []
    for p in class_dir.rglob("*"):
        if p.suffix in IMAGE_EXTS and p.is_file():
            imgs.append(p)
    return imgs


def split_images(imgs: list[Path], seed: int, ratio: float):
    rng = random.Random(seed)
    shuffled = imgs[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut], shuffled[cut:]


def oversample(imgs: list[Path], target: int, rng: random.Random) -> list[Path]:
    if len(imgs) >= target:
        return imgs
    extra = rng.choices(imgs, k=target - len(imgs))
    return imgs + extra


def copy_images(imgs: list[Path], dest_dir: Path, prefix: str = ""):
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: Counter = Counter()
    for src in imgs:
        name = src.stem
        ext  = src.suffix
        key  = f"{prefix}{name}"
        seen[key] += 1
        count = seen[key]
        fname = f"{key}_{count:03d}{ext}" if count > 1 else f"{key}{ext}"
        dst = dest_dir / fname
        shutil.copy2(src, dst)


def build_dataset(
    splits: dict[str, tuple[list[Path], list[Path]]],
    class_map: dict[str, str],
    output_dir: Path,
    oversample_targets: dict[str, int] | None = None,
):
    """
    splits       : {orig_class: (train_imgs, val_imgs)}
    class_map    : {orig_class: dest_class_name}
    output_dir   : raiz do dataset
    """
    rng = random.Random(SEED)
    for orig_cls, (train_imgs, val_imgs) in splits.items():
        dest_cls = class_map.get(orig_cls, orig_cls)

        # --- val (sem oversampling) ---
        copy_images(val_imgs, output_dir / "val" / dest_cls, prefix="")

        # --- train (com oversampling opcional) ---
        train = train_imgs[:]
        if oversample_targets and dest_cls in oversample_targets:
            target = oversample_targets[dest_cls]
            train = oversample(train, target, rng)
        copy_images(train, output_dir / "train" / dest_cls, prefix="")


def main():
    random.seed(SEED)

    print("Coletando imagens...")
    all_splits: dict[str, tuple[list[Path], list[Path]]] = {}
    for cls, d in CLASS_DIRS.items():
        imgs = collect_images(d)
        train, val = split_images(imgs, SEED, SPLIT)
        all_splits[cls] = (train, val)
        print(f"  {cls}: {len(imgs)} total → {len(train)} train / {len(val)} val")

    # ── 1. ONE-STAGE: 4 classes ──────────────────────────────────────────────
    print("\nCriando data/one_stage/ ...")
    shutil.rmtree(OUTPUT_ROOT / "one_stage", ignore_errors=True)
    build_dataset(
        splits=all_splits,
        class_map={k: k for k in all_splits},
        output_dir=OUTPUT_ROOT / "one_stage",
        oversample_targets=OVERSAMPLE_TARGET,
    )

    # ── 2. BINARY: normal vs abnormal ────────────────────────────────────────
    print("Criando data/binary/ ...")
    shutil.rmtree(OUTPUT_ROOT / "binary", ignore_errors=True)

    # agrupa variation + opmd + oral_cancer → abnormal
    binary_train_normal = all_splits["normal"][0]
    binary_val_normal   = all_splits["normal"][1]
    binary_train_abnormal = (
        all_splits["variation"][0] +
        all_splits["opmd"][0] +
        all_splits["oral_cancer"][0]
    )
    binary_val_abnormal = (
        all_splits["variation"][1] +
        all_splits["opmd"][1] +
        all_splits["oral_cancer"][1]
    )
    rng = random.Random(SEED)
    binary_train_abnormal_os = oversample(
        binary_train_abnormal,
        target=int(len(binary_train_normal) * 0.40),  # ~40% do normal → mais balanceado
        rng=rng,
    )

    binary_splits = {
        "normal":   (binary_train_normal,      binary_val_normal),
        "abnormal": (binary_train_abnormal_os, binary_val_abnormal),
    }
    build_dataset(
        splits=binary_splits,
        class_map={k: k for k in binary_splits},
        output_dir=OUTPUT_ROOT / "binary",
        oversample_targets=None,  # já aplicado acima
    )

    # ── 3. STAGE2: 3 classes (só anômalos) ───────────────────────────────────
    print("Criando data/stage2/ ...")
    shutil.rmtree(OUTPUT_ROOT / "stage2", ignore_errors=True)
    stage2_splits = {k: all_splits[k] for k in ("variation", "opmd", "oral_cancer")}
    stage2_targets = {
        "variation":  300,
        "opmd":       300,
        "oral_cancer": 160,
    }
    build_dataset(
        splits=stage2_splits,
        class_map={k: k for k in stage2_splits},
        output_dir=OUTPUT_ROOT / "stage2",
        oversample_targets=stage2_targets,
    )

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n=== Resumo final ===")
    for task in ("one_stage", "binary", "stage2"):
        task_dir = OUTPUT_ROOT / task
        print(f"\n[{task}]")
        for split in ("train", "val"):
            split_dir = task_dir / split
            if not split_dir.exists():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                n = len(list(cls_dir.iterdir()))
                print(f"  {split}/{cls_dir.name}: {n}")


if __name__ == "__main__":
    main()
