"""
predict_from_yolo.py
YOLO txt 결과와 원본 이미지를 입력받아,
각 요소 crop을 ui_classifier.pt 로 분류하는 모듈.

출력: elements.json
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import json
from pathlib import Path


# -------------------------------
# 1️⃣ 모델 로드
# -------------------------------
def load_ui_classifier(model_path: "models/ui_classifier.pt", num_classes: int, device: str):
    weights = ResNet34_Weights.IMAGENET1K_V1
    model = resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load(model_path, map_location=device)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


# -------------------------------
# 2️⃣ crop → 224×224 패딩 + 정규화
# -------------------------------
def preprocess_crop(image_bgr: np.ndarray, target_size: int = 224) -> torch.Tensor:
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    x_off, y_off = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    arr = rgb.astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr = (arr - mean) / std

    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


# -------------------------------
# 3️⃣ YOLO TXT 기반 CNN 분류
# -------------------------------
def predict_crops_from_txt(
    img_path: str,
    txt_dir: str,
    model_path: "models/ui_classifier.pt",
    class_names: list[str],
    output_path: str
):
    """
    img_path: 원본 스케치 이미지 경로
    txt_dir: YOLO bbox txt들이 저장된 폴더 경로
    model_path: ui_classifier.pt 파일 경로
    class_names: CNN 클래스 리스트
    output_path: 저장할 elements.json 경로
    """

    # 원본 이미지 로드
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ui_classifier(model_path, len(class_names), device)

    results = []

    txt_dir = Path(txt_dir)
    txt_files = sorted(list(txt_dir.glob("*.txt")))

    for i, txt_file in enumerate(txt_files):
        line = txt_file.read_text().strip()
        if not line:
            continue

        # YOLO 포맷: class x_center y_center width height
        parts = line.split()
        if len(parts) != 5:
            continue

        _, xc, yc, w, h = map(float, parts)

        # YOLO 좌표 → 절대 픽셀 단위 변환
        x1 = int((xc - w / 2) * W)
        y1 = int((yc - h / 2) * H)
        x2 = int((xc + w / 2) * W)
        y2 = int((yc + h / 2) * H)

        # crop
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        tensor = preprocess_crop(crop).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs))

        # 결과 저장
        results.append({
            "id": i,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "type": class_names[pred_idx],
            "score": float(probs[pred_idx])
        })

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved elements.json → {output_path}")