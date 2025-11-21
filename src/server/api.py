from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from yolo_detector import UIDetector
from algorithms import run_algorithms, generate_message
import requests

app = FastAPI()

# CORS 허용 (프론트와 연결 위해 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI 모델 로드 (서버 시작할 때 1번만 수행)
detector = UIDetector()

# Backend URL
BACKEND_URL = "http://YOUR_BACKEND/api/upload"


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 1) 이미지 읽기
    img_bytes = await file.read()

    # 2) YOLO 탐지
    detections = detector.run(img_bytes)

    # 3) 접근성 알고리즘 실행
    analysis = run_algorithms(detections)
    message = generate_message(analysis)

    # 4) Backend로 전달
    payload = {
        "detections": detections,
        "analysis": analysis,
        "message": message
    }

    try:
        requests.post(BACKEND_URL, json=payload)
    except Exception as e:
        print("[WARN] Backend not reachable:", e)

    # 5) 프론트에 반환
    return {
        "message": message,
        "detections": detections,
        "analysis": analysis
    }
