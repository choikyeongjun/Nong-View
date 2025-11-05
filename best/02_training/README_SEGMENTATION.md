# Model3 Greenhouse Segmentation Training Guide

## 개요

`optimized_training.py`가 YOLOv11-seg 세그멘테이션 모델을 지원하도록 수정되었습니다.

### 주요 변경사항

1. **Segmentation 모델 지원**
   - YOLOv11n-seg, YOLOv11s-seg, YOLOv11m-seg, YOLOv11l-seg, YOLOv11x-seg

2. **데이터셋 추가**
   - MODEL3_GREENHOUSE_SEG: model3_greenhouse_seg_processed
   - 클래스: Greenhouse_single (단동), Greenhouse_multi (연동)

3. **Segmentation 전용 기능**
   - Mask loss weight 추가
   - overlap_mask, mask_ratio 파라미터 추가
   - Segmentation 메트릭 추적 (mask_mAP50, mask_mAP50-95)

4. **기존 최적화 유지**
   - Progressive resizing
   - Curriculum learning
   - Advanced learning rate scheduling
   - Hardware optimization

---

## 데이터 구조

```
model3_greenhouse_seg_processed/
├── data.yaml
├── images/
│   ├── train/    (1186 images)
│   ├── val/      (147 images)
│   └── test/     (150 images)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### data.yaml

```yaml
path: C:\Users\LX\Nong-View\model3_greenhouse_seg_processed
train: images/train
val: images/val
test: images/test
nc: 2
names:
  - Greenhouse_single
  - Greenhouse_multi
task: segment
```

---

## 사용 방법

### 방법 1: train_model3_seg.py 실행 (권장)

```bash
cd best/02_training
python train_model3_seg.py
```

### 방법 2: optimized_training.py 직접 실행

```bash
cd best/02_training
python optimized_training.py --task segment --data "C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml" --epochs 100 --batch 16
```

### 방법 3: Python 코드에서 사용

```python
import sys
sys.path.append('../configs')

from best_config import ModelType, DatasetType
from optimized_training import (
    OptimizedModelTrainer,
    create_training_config,
    TrainingStrategy
)

# 설정 생성
config = create_training_config(
    model_type=ModelType.YOLO11N_SEG,
    dataset_type=DatasetType.MODEL3_GREENHOUSE_SEG,
    strategy=TrainingStrategy.PROGRESSIVE
)

# 설정 조정
config.epochs = 100
config.batch_size = 16
config.imgsz = 640

# 학습 실행
trainer = OptimizedModelTrainer(config)
results = trainer.train(r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml")

print(f"Best Mask mAP50: {results['best_metrics']['mask_mAP50']:.4f}")
```

---

## 모델 선택

### 모델 크기별 특성

| 모델 | 배치 크기 | 학습률 | 속도 | 정확도 | 메모리 |
|------|-----------|--------|------|--------|--------|
| YOLO11N_SEG | 16 | 0.001 | 빠름 | 보통 | 낮음 |
| YOLO11S_SEG | 12 | 0.001 | 보통 | 좋음 | 보통 |
| YOLO11M_SEG | 8 | 0.0008 | 느림 | 우수 | 높음 |
| YOLO11L_SEG | 6 | 0.0005 | 매우 느림 | 최고 | 매우 높음 |
| YOLO11X_SEG | 4 | 0.0003 | 극도로 느림 | 최고 | 극도로 높음 |

### 추천

- **빠른 실험/프로토타입**: YOLO11N_SEG
- **균형잡힌 성능**: YOLO11S_SEG
- **높은 정확도**: YOLO11M_SEG
- **최고 성능**: YOLO11L_SEG 또는 YOLO11X_SEG

---

## 주요 설정

### 손실 가중치 (Segmentation 최적화)

```python
box_loss_weight = 7.5    # Bounding box loss
cls_loss_weight = 0.5    # Classification loss
dfl_loss_weight = 1.5    # Distribution focal loss
mask_loss_weight = 2.5   # Mask segmentation loss (중요!)
```

### Segmentation 전용 설정

```python
overlap_mask = True      # 겹치는 마스크 허용
mask_ratio = 4          # 마스크 다운샘플링 비율
```

### 데이터 증강

```python
mosaic = 1.0            # 모자이크 증강
mixup = 0.15           # 믹스업 증강
copy_paste = 0.3       # 복사-붙여넣기 증강
degrees = 10.0         # 회전 (-10° ~ +10°)
translate = 0.2        # 이동 (20%)
scale = 0.9           # 스케일 (0.9 ~ 1.1)
```

---

## 출력 결과

### 디렉토리 구조

```
results/training_YYYYMMDD_HHMMSS/
├── train/
│   ├── weights/
│   │   ├── best.pt           # 최고 성능 모델
│   │   └── last.pt           # 마지막 모델
│   ├── results.png           # 학습 곡선
│   ├── confusion_matrix.png  # 혼동 행렬
│   ├── val_batch*.jpg        # 검증 결과
│   └── ...
├── training_history.json     # 학습 히스토리
└── training_report.json      # 종합 리포트
```

### 메트릭

#### Box Metrics (Bounding Box)
- `box_mAP50`: Box mAP @ IoU=0.5
- `box_mAP50-95`: Box mAP @ IoU=0.5:0.95
- `box_precision`: Box Precision
- `box_recall`: Box Recall

#### Mask Metrics (Segmentation)
- `mask_mAP50`: Mask mAP @ IoU=0.5
- `mask_mAP50-95`: Mask mAP @ IoU=0.5:0.95
- `mask_precision`: Mask Precision
- `mask_recall`: Mask Recall

---

## 학습 전략

### Progressive Resizing (기본)

```python
strategy = TrainingStrategy.PROGRESSIVE
```

- 단계적으로 이미지 크기 증가
- 초기: 320px → 중기: 416px → 후기: 640px
- 학습 속도 향상 및 과적합 방지

### Curriculum Learning

```python
strategy = TrainingStrategy.CURRICULUM
```

- 쉬운 샘플부터 어려운 샘플로
- 점진적인 증강 강도 증가

---

## 하드웨어 최적화

### GPU 최적화
- TF32 자동 활성화 (Ampere GPU 이상)
- cuDNN 벤치마크 활성화
- Automatic Mixed Precision (AMP) 지원
- 메모리 효율 최적화

### CPU 최적화
- 멀티코어 데이터 로딩 (workers=8)
- 효율적인 전처리 파이프라인

---

## 성능 목표

### Model3 Greenhouse Segmentation

| 메트릭 | 목표 | 우수 |
|--------|------|------|
| Mask mAP50 | > 85% | > 90% |
| Mask mAP50-95 | > 70% | > 80% |
| Box mAP50 | > 88% | > 93% |
| 추론 속도 | < 50ms | < 30ms |

---

## 문제 해결

### 메모리 부족 오류

```python
# 배치 크기 감소
config.batch_size = 8

# 이미지 크기 감소
config.imgsz = 512

# Workers 감소
config.workers = 4
```

### 과적합 (Overfitting)

```python
# 증강 강화
config.mosaic = 1.0
config.mixup = 0.3
config.copy_paste = 0.5

# 정규화 추가
config.label_smoothing = 0.1
config.weight_decay = 0.001
```

### 학습 속도 개선

```python
# 작은 모델 사용
model_type = ModelType.YOLO11N_SEG

# 배치 크기 증가 (GPU 메모리 허용 시)
config.batch_size = 32

# 캐시 활성화
train_args['cache'] = 'ram'  # or 'disk'
```

---

## 추가 정보

### 관련 파일
- `best/configs/best_config.py`: 통합 설정 시스템
- `best/02_training/optimized_training.py`: 최적화 학습 시스템
- `best/02_training/train_model3_seg.py`: 실행 스크립트

### 참고 문서
- [YOLO11 Segmentation 가이드](https://docs.ultralytics.com/tasks/segment/)
- [데이터 증강 전략](../docs/augmentation_strategies.md)
- [하드웨어 최적화](../docs/hardware_optimization.md)

---

## 예상 학습 시간

### RTX A6000 (48GB) 기준

| 모델 | 배치 크기 | 100 에포크 | 200 에포크 |
|------|-----------|-----------|-----------|
| YOLO11N_SEG | 16 | ~2시간 | ~4시간 |
| YOLO11S_SEG | 12 | ~3시간 | ~6시간 |
| YOLO11M_SEG | 8 | ~5시간 | ~10시간 |
| YOLO11L_SEG | 6 | ~8시간 | ~16시간 |
| YOLO11X_SEG | 4 | ~12시간 | ~24시간 |

---

## 연락처

문제가 발생하거나 질문이 있으시면 개발팀에 문의하세요.

**개발팀**: Claude Opus (Architecture) + Claude Sonnet (Implementation)
**버전**: 2.0.0 (Segmentation Support)
**날짜**: 2025-11-04


