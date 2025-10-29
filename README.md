# Accessibility-ML

손그림 기반 모바일 UI 스케치에서
1) UI 구성요소를 분류하고 (CNN 요소 분류기),
2) 합성 레이아웃을 자동 생성하고,
3) 접근성 규칙(터치 타깃 크기, 간격, 대비 등)을 통해 위험 라벨을 만들고,
4) 화면 단위 접근성 위험을 예측하는 모델(DNN/CNN)을 학습하는 저장소.

## 폴더 구조
ai/
 ├─ notebooks/
 │    ├─ train_element_classifier_seyoung.ipynb     # 세영 Colab 노트북
 │    ├─ train_element_classifier_teammate.ipynb     # 원하 Colab 노트북
 │
 ├─ training/
 │    ├─ dataloader.py       # fastai DataBlock, 전처리 정의
 │    ├─ train.py            # 학습 루프 (fine_tune 등)
 │
 ├─ synth_layout/
 │    ├─ generate_layout.py  # 합성 레이아웃 생성기 (추후 사용)
 │    ├─ rules.py            # 접근성 규칙 엔진 (추후 사용)
 │
 ├─ exports/
 │    └─ README.md           # 모델 설명 (pth 파일은 업로드 금지)
 │
 ├─ requirements.txt
 ├─ .gitignore
 └─ README.md   ← 본 문서

## Colab 사용 흐름
1. Colab에서 레포를 clone한다.
2. requirements.txt 설치한다.
3. 개인 데이터(archive.zip)를 업로드하고 unzip해서 `data/archive/` 로 둔다.
4. notebooks 안에서 학습을 돌리고 결과를 저장한다.
5. 수정된 노트북을 commit/push 해서 기록을 공유한다.
