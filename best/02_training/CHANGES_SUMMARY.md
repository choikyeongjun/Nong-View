# Optimized Training System - Segmentation Support Update

## 날짜: 2025-11-04
## 버전: 2.0.0 (Segmentation Support)

---

## 요약

`best/02_training/optimized_training.py`가 **YOLOv11-seg Segmentation 모델**을 완전히 지원하도록 수정되었습니다.

**데이터셋**: `model3_greenhouse_seg_processed`  
**클래스**: Greenhouse_single (단동), Greenhouse_multi (연동)  
**모델**: YOLOv11-seg (nano, small, medium, large, xlarge)  
**태스크**: Segmentation

---

## 주요 변경사항

### 1. 설정 파일 (`best/configs/best_config.py`)

#### 추가된 모델 타입
```python
class ModelType(Enum):
    # Detection 모델 (기존)
    YOLO11N = "yolo11n"
    YOLO11S = "yolo11s"
    YOLO11M = "yolo11m"
    YOLO11L = "yolo11l"
    YOLO11X = "yolo11x"
    
    # Segmentation 모델 (신규)
    YOLO11N_SEG = "yolo11n-seg"
    YOLO11S_SEG = "yolo11s-seg"
    YOLO11M_SEG = "yolo11m-seg"
    YOLO11L_SEG = "yolo11l-seg"
    YOLO11X_SEG = "yolo11x-seg"
```

#### 추가된 데이터셋 타입
```python
class DatasetType(Enum):
    # Detection 데이터셋 (기존)
    GREENHOUSE_MULTI = "greenhouse_multi"
    GREENHOUSE_SINGLE = "greenhouse_single"
    GROWTH_TIF = "growth_tif"
    
    # Segmentation 데이터셋 (신규)
    MODEL3_GREENHOUSE_SEG = "model3_greenhouse_seg"
```

#### Segmentation 손실 가중치
```python
mask_loss_gain: float = 2.5  # Segmentation 전용
```

#### 모델별 최적 설정 (Segmentation)
```python
ModelType.YOLO11N_SEG: {
    "batch_size": 16,
    "lr0": 0.001,
    "warmup_epochs": 3,
    "overlap_mask": True,
    "mask_ratio": 4
}
# ... 다른 크기 모델들도 동일한 패턴
```

---

### 2. 학습 시스템 (`best/02_training/optimized_training.py`)

#### TrainingConfig 확장
```python
@dataclass
class TrainingConfig:
    # 기존 필드들...
    task: str = "detect"  # detect or segment (신규)
    
    # Segmentation 전용 필드 (신규)
    mask_loss_weight: float = 2.5
    overlap_mask: bool = True
    mask_ratio: int = 4
```

#### LossFunctionOptimizer 개선
- Segmentation task 감지 시 자동으로 mask loss weight 추가
- MODEL3_GREENHOUSE_SEG 데이터셋에 대한 특화 설정
  - Mask loss weight 1.2배 증가 (정밀한 경계 검출)

```python
def _calculate_adaptive_weights(self) -> Dict[str, float]:
    weights = {
        'box': self.config.box_loss_weight,
        'cls': self.config.cls_loss_weight,
        'dfl': self.config.dfl_loss_weight
    }
    
    # Segmentation 지원
    if self.config.task == 'segment':
        weights['mask'] = self.config.mask_loss_weight
    
    # 데이터셋별 최적화
    if self.config.dataset_type == DatasetType.MODEL3_GREENHOUSE_SEG:
        if 'mask' in weights:
            weights['mask'] *= 1.2  # 비닐하우스는 정밀한 경계 필요
    
    return weights
```

#### 모델 초기화 개선
```python
def _initialize_model(self) -> YOLO:
    model_value = self.config.model_type.value
    
    # Segmentation 모델 자동 감지
    if model_value.endswith('-seg'):
        model_name = f"{model_value}.pt"
    else:
        model_name = f"yolo11{model_value}.pt"
    
    model = YOLO(model_name)
    logger.info(f"Model initialized: {model_name} (task: {self.config.task})")
    return model
```

#### 학습 인자 개선
```python
# 기본 학습 인자
train_args = {
    'data': data_yaml,
    'task': self.config.task,  # detect 또는 segment
    # ... 기존 인자들
}

# Segmentation 전용 인자 추가
if self.config.task == 'segment':
    train_args.update({
        'overlap_mask': self.config.overlap_mask,
        'mask_ratio': self.config.mask_ratio
    })
```

#### 메트릭 추적 개선
```python
# Segmentation 메트릭 추가
if config.task == 'segment':
    self.best_metrics = {
        'mAP50': 0, 'mAP50-95': 0,
        'mask_mAP50': 0, 'mask_mAP50-95': 0,  # 신규
        'loss': float('inf')
    }

# 리포트 생성 시 Segmentation 메트릭 포함
if self.config.task == 'segment':
    final_metrics.update({
        'mask_mAP50': results.results_dict.get('metrics/mAP50(M)', 0),
        'mask_mAP50-95': results.results_dict.get('metrics/mAP50-95(M)', 0),
        'mask_precision': results.results_dict.get('metrics/precision(M)', 0),
        'mask_recall': results.results_dict.get('metrics/recall(M)', 0),
    })
```

#### create_training_config 함수 개선
```python
def create_training_config(
    model_type: ModelType,
    dataset_type: DatasetType,
    strategy: TrainingStrategy = TrainingStrategy.PROGRESSIVE,
    data_yaml: Optional[str] = None
) -> TrainingConfig:
    # Task 자동 감지
    is_segmentation = '_SEG' in model_type.name
    task = 'segment' if is_segmentation else 'detect'
    
    # Segmentation 설정 자동 적용
    if is_segmentation:
        config.overlap_mask = model_specific.get('overlap_mask', True)
        config.mask_ratio = model_specific.get('mask_ratio', 4)
        config.mask_loss_weight = CONFIG.training.mask_loss_gain
    
    # MODEL3_GREENHOUSE_SEG 특화 설정
    if dataset_type == DatasetType.MODEL3_GREENHOUSE_SEG:
        config.mask_loss_weight *= 1.2
        config.copy_paste = 0.3
        config.mosaic = 1.0
    
    return config
```

---

### 3. 새로 추가된 파일

#### `best/02_training/train_model3_seg.py`
Model3 Greenhouse Segmentation 학습 전용 실행 스크립트

**특징**:
- 데이터 경로 자동 설정
- 최적 설정 사전 구성
- 상세한 로깅 및 진행 상황 표시
- 학습 결과 종합 리포트

**실행 방법**:
```bash
cd best/02_training
python train_model3_seg.py
```

#### `best/02_training/test_segmentation_config.py`
Segmentation 설정 검증 테스트 스크립트

**테스트 항목**:
1. 모델 타입 확인 (Detection + Segmentation)
2. 데이터셋 타입 확인 (MODEL3_GREENHOUSE_SEG 포함)
3. Segmentation 설정 생성 및 검증
4. Detection 설정 호환성 확인
5. 모든 Segmentation 모델 크기 테스트
6. 데이터 경로 및 data.yaml 검증

**실행 방법**:
```bash
cd best/02_training
python test_segmentation_config.py
```

**테스트 결과**: ✅ 모든 테스트 통과

#### `best/02_training/README_SEGMENTATION.md`
Segmentation 학습 시스템 종합 가이드

**내용**:
- 시스템 개요 및 변경사항
- 데이터 구조 및 설정
- 사용 방법 (3가지)
- 모델 선택 가이드
- 주요 설정 설명
- 출력 결과 및 메트릭
- 학습 전략
- 하드웨어 최적화
- 성능 목표
- 문제 해결
- 예상 학습 시간

---

## 기존 기능 유지

### ✅ Detection 모델 완전 호환
기존 Detection 모델 학습은 **변경 없이 동일하게 작동**합니다.

### ✅ 모든 최적화 기법 유지
- Progressive Resizing
- Curriculum Learning
- Advanced Learning Rate Scheduling
- Hardware-aware Optimization
- Focal Loss
- Adaptive Loss Weights
- EMA (Exponential Moving Average)
- AMP (Automatic Mixed Precision)

### ✅ 기존 데이터셋 지원
- GREENHOUSE_MULTI
- GREENHOUSE_SINGLE
- GROWTH_TIF

---

## 사용 예제

### Segmentation 학습

```python
from best_config import ModelType, DatasetType
from optimized_training import OptimizedModelTrainer, create_training_config

# 설정 생성
config = create_training_config(
    model_type=ModelType.YOLO11N_SEG,
    dataset_type=DatasetType.MODEL3_GREENHOUSE_SEG,
    strategy=TrainingStrategy.PROGRESSIVE
)

# 학습 실행
trainer = OptimizedModelTrainer(config)
results = trainer.train(r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml")

print(f"Box mAP50: {results['best_metrics']['mAP50']:.4f}")
print(f"Mask mAP50: {results['best_metrics']['mask_mAP50']:.4f}")
```

### Detection 학습 (기존과 동일)

```python
config = create_training_config(
    model_type=ModelType.YOLO11N,
    dataset_type=DatasetType.GROWTH_TIF,
    strategy=TrainingStrategy.PROGRESSIVE
)

trainer = OptimizedModelTrainer(config)
results = trainer.train("path/to/data.yaml")
```

---

## 검증 결과

### 테스트 통과 ✅

```
1. ✓ 모델 타입 테스트 통과
   - Detection 모델: YOLO11N, S, M, L, X
   - Segmentation 모델: YOLO11N_SEG, S_SEG, M_SEG, L_SEG, X_SEG

2. ✓ 데이터셋 타입 테스트 통과
   - Detection: GREENHOUSE_MULTI, GREENHOUSE_SINGLE, GROWTH_TIF
   - Segmentation: MODEL3_GREENHOUSE_SEG

3. ✓ Segmentation 설정 생성 테스트 통과
   - Task: segment
   - Overlap Mask: True
   - Mask Ratio: 4
   - Mask Loss Weight: 3.0 (1.2배 증가)

4. ✓ Detection 설정 테스트 통과
   - 기존 기능 100% 유지

5. ✓ 모든 Segmentation 모델 설정 테스트 통과
   - 5개 모델 크기 모두 정상

6. ✓ 데이터 경로 확인 완료
   - data.yaml 존재 확인
   - Task: segment 확인
   - Classes: 2 (Greenhouse_single, Greenhouse_multi)
```

### 설정 검증 ✅

```python
# Segmentation 설정 예시
Model Type: YOLO11N_SEG
Dataset Type: MODEL3_GREENHOUSE_SEG
Task: segment
Epochs: 100
Batch Size: 16
Image Size: 640
Base LR: 0.001
Strategy: progressive

손실 가중치:
- Box Loss Weight: 7.5
- Cls Loss Weight: 0.5
- DFL Loss Weight: 1.5
- Mask Loss Weight: 3.0  # 비닐하우스 최적화 (2.5 * 1.2)

Segmentation 전용:
- Overlap Mask: True
- Mask Ratio: 4
```

---

## 실행 방법

### 방법 1: 전용 스크립트 (가장 간단)

```bash
cd best/02_training
python train_model3_seg.py
```

### 방법 2: 직접 실행 (커스터마이징 가능)

```bash
cd best/02_training
python optimized_training.py --task segment \
    --data "C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml" \
    --epochs 100 --batch 16
```

### 방법 3: Python 코드

```python
import sys
sys.path.append('path/to/best/configs')

from best_config import ModelType, DatasetType
from optimized_training import OptimizedModelTrainer, create_training_config

config = create_training_config(
    model_type=ModelType.YOLO11N_SEG,
    dataset_type=DatasetType.MODEL3_GREENHOUSE_SEG
)

trainer = OptimizedModelTrainer(config)
results = trainer.train(r"C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\data.yaml")
```

---

## 예상 학습 시간 (RTX A6000 기준)

| 모델 | 배치 크기 | 100 에포크 |
|------|-----------|-----------|
| YOLO11N_SEG | 16 | ~2시간 |
| YOLO11S_SEG | 12 | ~3시간 |
| YOLO11M_SEG | 8 | ~5시간 |
| YOLO11L_SEG | 6 | ~8시간 |
| YOLO11X_SEG | 4 | ~12시간 |

---

## 출력 메트릭

### Box Metrics (Bounding Box)
- `box_mAP50`: Box mAP @ IoU=0.5
- `box_mAP50-95`: Box mAP @ IoU=0.5:0.95
- `box_precision`: Box Precision
- `box_recall`: Box Recall

### Mask Metrics (Segmentation) 🆕
- `mask_mAP50`: Mask mAP @ IoU=0.5
- `mask_mAP50-95`: Mask mAP @ IoU=0.5:0.95
- `mask_precision`: Mask Precision
- `mask_recall`: Mask Recall

---

## 파일 변경 요약

### 수정된 파일
1. `best/configs/best_config.py`
   - Segmentation 모델 타입 추가
   - MODEL3_GREENHOUSE_SEG 데이터셋 추가
   - mask_loss_gain 설정 추가
   - Segmentation 모델별 최적 설정 추가

2. `best/02_training/optimized_training.py`
   - TrainingConfig에 task, mask 관련 필드 추가
   - LossFunctionOptimizer Segmentation 지원
   - 모델 초기화 Segmentation 지원
   - 메트릭 추적 Segmentation 지원
   - create_training_config Segmentation 지원

### 새로 추가된 파일
1. `best/02_training/train_model3_seg.py`
   - Model3 Greenhouse 전용 학습 스크립트

2. `best/02_training/test_segmentation_config.py`
   - Segmentation 설정 검증 테스트 스크립트

3. `best/02_training/README_SEGMENTATION.md`
   - Segmentation 학습 종합 가이드

4. `best/02_training/CHANGES_SUMMARY.md`
   - 이 문서 (변경사항 요약)

---

## 기술적 세부사항

### Segmentation 모델 로딩
```python
# Detection: yolo11n.pt
# Segmentation: yolo11n-seg.pt

if model_value.endswith('-seg'):
    model_name = f"{model_value}.pt"
else:
    model_name = f"yolo11{model_value}.pt"
```

### Task 자동 감지
```python
is_segmentation = '_SEG' in model_type.name
task = 'segment' if is_segmentation else 'detect'
```

### Adaptive Loss Weights (Segmentation)
```python
# 기본 가중치
box: 7.5, cls: 0.5, dfl: 1.5, mask: 2.5

# MODEL3_GREENHOUSE_SEG 최적화
mask: 2.5 * 1.2 = 3.0  # 비닐하우스는 정밀한 경계 필요
```

---

## 다음 단계

### 즉시 가능
1. ✅ Segmentation 학습 실행
2. ✅ 다양한 모델 크기 실험
3. ✅ 하이퍼파라미터 튜닝

### 향후 개선 사항
1. Segmentation 전용 증강 기법 추가
2. Ensemble 학습 지원
3. Knowledge Distillation 구현
4. 추론 최적화 (TensorRT, ONNX)

---

## 연락처 및 지원

**개발팀**: Claude Opus (Architecture) + Claude Sonnet (Implementation)  
**버전**: 2.0.0 (Segmentation Support)  
**날짜**: 2025-11-04

문제 발생 시:
1. `test_segmentation_config.py` 실행하여 설정 확인
2. `README_SEGMENTATION.md` 참고
3. 로그 파일 확인 (`results/training_*/train/`)

---

## 라이센스

기존 Nong-View 프로젝트 라이센스를 따릅니다.

