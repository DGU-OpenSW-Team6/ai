# Accessibility-ML

손그림 기반 모바일 UI 스케치에서
1) UI 구성요소를 분류하고 (CNN 요소 분류기),
2) 합성 레이아웃을 자동 생성하고,
3) 접근성 규칙(터치 타깃 크기, 간격, 대비 등)을 통해 위험 라벨을 만들고,
4) 화면 단위 접근성 위험을 예측하는 모델(DNN/CNN)을 학습하는 저장소.

## 폴더 구조
- `notebooks/`
  - Colab에서 돌리는 실험 노트북.
  - 사람별로 파일을 나눠서 충돌 안 나게 관리 (예: `_seyoung.ipynb`, `_teammate.ipynb`).
- `training/`
  - 공용 학습 로직 (fastai dataloader, train loop 등).
  - 노트북에서 공통으로 import해서 재사용.
- `synth_layout/`
  - 합성 레이아웃 생성 코드와 접근성 규칙 엔진 코드.
- `exports/`
  - 최종 학습된 모델 가중치(.pth)에 대한 설명만 저장.
  - 실제 .pth 파일은 .gitignore 처리되어 깃허브에 올라가지 않음.

## Colab 사용 흐름
1. Colab에서 레포를 clone한다.
2. requirements.txt 설치한다.
3. 개인 데이터(archive.zip)를 업로드하고 unzip해서 `data/archive/` 로 둔다.
4. notebooks 안에서 학습을 돌리고 결과를 저장한다.
5. 수정된 노트북을 commit/push 해서 기록을 공유한다.
