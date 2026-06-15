"""
Orquestra os 9 treinamentos sequencialmente, chamando train() diretamente
(sem subprocess, para não conflitar com o multiprocessing do dataloader).
"""

from train import train

MODELS = ["yolov8m", "yolov11m", "yolo26m"]
TASKS  = ["one_stage", "binary", "stage2"]


def main():
    combos = [(m, t) for t in TASKS for m in MODELS]
    total  = len(combos)

    for i, (model, task) in enumerate(combos, 1):
        print(f"\n{'#'*60}")
        print(f"[{i}/{total}] {model} | {task}")
        print(f"{'#'*60}")
        train(model, task)

    print("\nTodos os 9 treinamentos concluídos.")


if __name__ == "__main__":
    main()
