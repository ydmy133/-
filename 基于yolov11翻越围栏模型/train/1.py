import os
from ultralytics import YOLO

def train_yolov11(data_yaml="data.yaml", pretrained_weights="yolo11s.pt",
                epochs=100, batch_size=32, img_size=640,
                lr=0.01, cos_lr=True, warmup_epochs=3.0, patience=10,
                device=0, project="runs/train", name="yolo11s_improved"):
    model = YOLO(pretrained_weights)

    train_params = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "lr0": lr,
        "device": device,
        "project": project,
        "name": name,
        "workers": 8,
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "verbose": True,
        "cos_lr": cos_lr,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "close_mosaic": 10,
        "patience": patience,
        "save_period": 5,
        "resume": False,
        "amp": True,
        "cache": False
    }

    results = model.train(**train_params)


def evaluate_yolov11(model_weights, data_yaml="data.yaml", device=0):
    model = YOLO(model_weights)
    metrics = model.val(data=data_yaml, device=device, split="val")
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    mp = metrics.box.mp
    mr = metrics.box.mr
    f1 = (2 * mp * mr) / (mp + mr + 1e-16)
    print("验证结果（best.pt）:")
    print(f"  mAP@0.5      = {map50:.4f}")
    print(f"  mAP@0.5:0.95 = {map5095:.4f}")
    print(f"  Precision    = {mp:.4f}")
    print(f"  Recall       = {mr:.4f}")
    print(f"  F1 Score     = {f1:.4f}")
    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        names = model.names
        print("各类别 AP@0.5:0.95:")
        for idx, ap in enumerate(metrics.box.maps):
            print(f"  Class {idx} ({names[idx]}): AP = {ap:.4f}")


def find_latest_model_dir(project="runs/train", prefix="yolo11s_improved"):
    subdirs = [d for d in os.listdir(project) if d.startswith(prefix)]
    if not subdirs:
        raise FileNotFoundError("未找到任何训练输出目录")
    latest = max(subdirs, key=lambda x: os.path.getmtime(os.path.join(project, x)))
    return os.path.join(project, latest)


if __name__ == "__main__":
    DATA_YAML = "data.yaml"
    PRETRAINED = "yolo11s.pt"
    PROJECT_DIR = "../runs/train"
    RUN_NAME = "yolo11s_improved"

    train_yolov11(data_yaml=DATA_YAML, pretrained_weights=PRETRAINED,
                 epochs=100, batch_size=16, img_size=640, lr=0.01,
                 project=PROJECT_DIR, name=RUN_NAME,cos_lr=False)#batch_size:2n次幂，img_size:32的倍数

    latest_dir = find_latest_model_dir(project=PROJECT_DIR, prefix=RUN_NAME)
    best_weights_path = os.path.join(latest_dir, "weights", "best.pt")
    print(f"\n检测到最新 best.pt 路径: {best_weights_path}\n")
    evaluate_yolov11(best_weights_path, data_yaml=DATA_YAML)
