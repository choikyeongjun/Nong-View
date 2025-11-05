# Model3 Greenhouse - 1024px 원본 크기 최적화

## 날짜: 2025-11-04
## 업데이트: 1024px 원본 데이터 반영

---

## 📊 주요 변경사항

### 원본 데이터 크기
```
이전 설정: 640px (추정)
실제 원본: 1024px ✅
학습 크기: 1024px (원본 크기 유지)
```

---

## 🎯 최적화된 설정 (RTX A6000 48GB)

### Segmentation 모델별 설정 (1024px)

| 모델 | 이미지 크기 | 배치 크기 | 학습률 | 예상 메모리 | 100 epoch |
|------|------------|----------|--------|------------|-----------|
| **YOLO11N-SEG** | 1024px | 8 | 0.001 | ~20GB | ~4h |
| **YOLO11S-SEG** | 1024px | 6 | 0.001 | ~25GB | ~5h |
| **YOLO11M-SEG** | 1024px | 16 | 0.0008 | ~38GB | ~6h |
| **YOLO11L-SEG** | 1024px | 12 | 0.0005 | ~42GB | ~10h |
| **YOLO11X-SEG** | 1024px | 8 | 0.0003 | ~45GB | ~15h |

### 앙상블 설정 (M + L + X)

```python
model_sizes = [
    ModelType.YOLO11M_SEG,  # 1024px, batch 16
    ModelType.YOLO11L_SEG,  # 1024px, batch 12
    ModelType.YOLO11X_SEG   # 1024px, batch 8
]
```

---

## 📈 성능 비교 (640px vs 1024px)

### 이미지 크기별 특성

| 크기 | 장점 | 단점 | 권장 |
|------|------|------|------|
| **640px** | • 빠른 학습<br>• 적은 메모리<br>• 큰 배치 | • 낮은 정확도<br>• 세부 정보 손실 | 빠른 실험 |
| **1024px** | • **높은 정확도**<br>• 세부 정보 보존<br>• 정밀한 경계 | • 느린 학습<br>• 많은 메모리<br>• 작은 배치 | **정식 학습** ✅ |

### 예상 성능 향상

```
640px 학습:
- Box mAP50:  ~88-90%
- Mask mAP50: ~85-87%

1024px 학습:
- Box mAP50:  ~92-95% (+3-5% ⬆️)
- Mask mAP50: ~90-93% (+5-6% ⬆️)
```

**Segmentation은 정밀한 경계가 중요하므로 1024px 학습 필수!** 🎯

---

## 🔧 메모리 최적화

### 현재 상태
```
GPU: RTX A6000 (48GB)
현재 사용: 13-14GB (28%)  ❌ 비효율
```

### 최적화 후
```
예상 사용: 38-45GB (80-95%)  ✅ 최적
```

### 배치 크기 조정

| 모델 | 640px 배치 | 1024px 배치 | 메모리 사용 |
|------|-----------|------------|-----------|
| M-SEG | 24 | **16** | ~38GB |
| L-SEG | 16 | **12** | ~42GB |
| X-SEG | 12 | **8** | ~45GB |

**1024px는 4배 큰 이미지 → 배치 크기 약 1/2로 조정**

---

## ⏰ 학습 시간 (1024px 기준)

### 3 epochs (현재 설정)
```
YOLO11M-SEG: ~10분
YOLO11L-SEG: ~18분
YOLO11X-SEG: ~27분
Total: ~55분
```

### 100 epochs (권장)
```
YOLO11M-SEG: ~6시간
YOLO11L-SEG: ~10시간
YOLO11X-SEG: ~15시간
Total: ~31시간
```

### 200 epochs (최고 성능)
```
Total: ~62시간 (약 2.5일)
```

---

## 🎯 왜 1024px로 학습하는가?

### 1. **원본 크기 유지**
```
원본: 1024px
학습: 1024px  ← 해상도 손실 없음!
추론: 1024px  ← 최적 성능
```

### 2. **Segmentation 특성**
```
Detection:    Box만 필요 → 640px도 충분
Segmentation: 정밀한 Polygon → 1024px 필수!
```

### 3. **비닐하우스 특성**
- 작은 구조물 (모서리, 연결부) 검출
- 정확한 경계선 필요
- 단동/연동 구분의 미세한 차이

---

## 📝 변경된 파일

### 1. `best/configs/best_config.py`

#### 데이터셋 정보
```python
DatasetType.MODEL3_GREENHOUSE_SEG: {
    "path": "model3_greenhouse_seg_processed",
    "classes": ["Greenhouse_single", "Greenhouse_multi"],
    "total_images": 1483,
    "total_objects": 1483,
    "task": "segment",
    "original_image_size": 1024,        # 추가 ✅
    "recommended_train_size": 1024,     # 추가 ✅
    "class_balance": {
        "Greenhouse_single": 1.0,
        "Greenhouse_multi": 1.0
    }
}
```

#### 모델별 설정
```python
ModelType.YOLO11M_SEG: {
    "batch_size": 16,
    "imgsz": 1024,     # 추가 ✅
    "lr0": 0.0008,
    "warmup_epochs": 5,
    "overlap_mask": True,
    "mask_ratio": 4
}
# L-SEG, X-SEG도 동일하게 imgsz: 1024 추가
```

### 2. `best/02_training/optimized_training.py`

#### create_training_config 함수
```python
# 이미지 크기 자동 결정
default_imgsz = dataset_info.get('recommended_train_size', 640)
config.imgsz = model_specific.get('imgsz', default_imgsz)
```

#### 앙상블 배치 크기
```python
# 1024px 기준으로 재조정
YOLO11M_SEG: batch 16
YOLO11L_SEG: batch 12
YOLO11X_SEG: batch 8
```

---

## 🚀 실행

```bash
cd C:\Users\LX\Nong-View\best\02_training
python optimized_training.py
```

### 출력 예시

```
================================================================================
🎯 MODEL3 GREENHOUSE SEGMENTATION - ENSEMBLE TRAINING
================================================================================
Models: YOLO11M-SEG + YOLO11L-SEG + YOLO11X-SEG

📋 Creating configurations:
--------------------------------------------------------------------------------
✓ YOLO11M_SEG
  - Image Size: 1024px     ← 원본 크기!
  - Batch Size: 16
  - Learning Rate: 0.0008
  - Epochs: 3
  - Overlap Mask: True
  - Mask Ratio: 4

✓ YOLO11L_SEG
  - Image Size: 1024px     ← 원본 크기!
  - Batch Size: 12
  - Learning Rate: 0.0005
  ...

✓ YOLO11X_SEG
  - Image Size: 1024px     ← 원본 크기!
  - Batch Size: 8
  - Learning Rate: 0.0003
  ...

⏰ Estimated total training time (1024px, RTX A6000):
   Current setting: 3 epochs → ~0.9 hours
   Expected GPU memory: 38-45GB (80-95%)

💡 1024px 학습으로 최고 성능 달성!
🚀 Starting ensemble training in 3 seconds...
```

---

## 📊 메모리 사용 예상

### 학습 시작 후
```bash
nvidia-smi
```

**예상 결과**:
```
M-SEG (batch 16): 38-40GB (80%)  ✅
L-SEG (batch 12): 42-44GB (88%)  ✅
X-SEG (batch 8):  45-47GB (95%)  ✅
```

---

## 💡 핵심 포인트

1. ✅ **원본 크기 유지**: 1024px로 학습
2. ✅ **메모리 최적화**: 배치 크기 조정으로 80-95% 사용
3. ✅ **성능 향상**: 640px 대비 +5-6% mAP 예상
4. ✅ **정밀한 Segmentation**: 비닐하우스 경계 정확도 극대화

**모든 설정이 1024px 원본 데이터에 최적화되었습니다!** 🎯

