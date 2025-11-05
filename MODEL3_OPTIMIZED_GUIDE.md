# 🏭 Model3 Greenhouse 최적화 전처리 가이드

## 📋 개요

`optimized_preprocessing.py` 구조를 기반으로 한 **Model3 Greenhouse 전용 전처리 시스템**입니다.

---

## ✨ 주요 특징

### 🔬 optimized_preprocessing.py 기반 고급 기능

1. **품질 필터링**
   - 흐림 감지 (Laplacian variance)
   - 밝기 평가
   - 대비 평가
   - 이상치 자동 제거 (IQR 방법)

2. **계층화 분할 (Stratified Split)**
   - 클래스별 독립 분할
   - 클래스 비율 유지
   - 재현성 보장 (random_seed=42)

3. **데이터 증강**
   - 온실 특화 증강 기법
   - 좌우 반전, 밝기 조정, 색상 변화
   - 미세한 회전 적용
   - **훈련 데이터만** 증강 (기본 2배)

4. **자동 통계 분석**
   - 처리 전후 통계 비교
   - 클래스 분포 분석
   - 품질 메트릭 수집

---

## 📊 데이터 정보

**소스**: `C:\Users\LX\Nong-View\model3_greenhouse`

**클래스**:
- **0**: Greenhouse_single (단동)
- **1**: Greenhouse_multi (연동)

**현재 데이터**:
- Train: 1,186개
- Val: 148개
- Test: 149개
- **총 1,483개 이미지**

---

## 🚀 사용 방법

### 방법 1: Python 스크립트 실행 (권장 ⭐)

#### 1️⃣ 스크립트 실행

```bash
cd C:\Users\LX\Nong-View
python preprocess_model3_optimized.py
```

#### 2️⃣ 처리 과정

스크립트가 자동으로 다음 단계를 수행합니다:

```
[1/6] 데이터 수집
  → train/val/test 폴더의 모든 이미지 수집
  → 1,483개 이미지 수집 완료

[2/6] 품질 분석
  → 흐림, 밝기, 대비 평가
  → 이상치 자동 제거
  → 품질 기준 통과 이미지만 선택

[3/6] 계층화 분할
  → 클래스별 독립 분할
  → Train: 80%, Val: 10%, Test: 10%
  → 클래스 비율 유지

[4/6] 출력 디렉토리 생성
  → YOLO 형식 구조 생성
  → C:\Users\LX\Nong-View\model3_greenhouse_best_processed

[5/6] 데이터 복사 및 증강
  → 원본 이미지 복사
  → 훈련 데이터 2배 증강 (온실 특화)
  → 진행바 표시

[6/6] 메타데이터 생성
  → data.yaml 생성
  → processing_stats.json 생성
  → 완료!
```

#### 3️⃣ 예상 소요 시간

- **품질 분석**: 2~3분
- **데이터 복사**: 2~3분
- **데이터 증강**: 3~5분
- **전체**: **약 8~10분**

---

### 방법 2: 설정 커스터마이징

스크립트 하단의 `main()` 함수에서 설정 수정:

```python
def main():
    config = Model3Config(
        source_dir=r"C:\Users\LX\Nong-View\model3_greenhouse",
        output_dir=r"C:\Users\LX\Nong-View\model3_greenhouse_best_processed",

        # 클래스 설정
        classes=['Greenhouse_single', 'Greenhouse_multi'],
        nc=2,

        # 데이터 분할 비율
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,

        # 전처리 옵션
        enable_quality_filter=True,      # 품질 필터링 활성화
        enable_augmentation=True,        # 데이터 증강 활성화
        augmentation_factor=2,           # 증강 배수 (2배)

        # 품질 임계값
        quality_threshold=0.4,           # 0.0 ~ 1.0 (낮을수록 관대)

        # 랜덤 시드
        random_seed=42
    )
```

---

## 📁 출력 구조

**위치**: `C:\Users\LX\Nong-View\model3_greenhouse_best_processed`

```
model3_greenhouse_best_processed/
├── data.yaml                    ← YOLO 설정 파일
├── processing_stats.json        ← 처리 통계
├── images/
│   ├── train/                   ← 훈련 이미지 (원본 + 증강)
│   │   ├── 1F001D40001.png
│   │   ├── 1F001D40001_aug1.png
│   │   └── 1F001D40001_aug2.png
│   ├── val/                     ← 검증 이미지 (원본만)
│   └── test/                    ← 테스트 이미지 (원본만)
└── labels/
    ├── train/                   ← 훈련 라벨
    ├── val/                     ← 검증 라벨
    └── test/                    ← 테스트 라벨
```

---

## 📊 data.yaml 파일 예시

```yaml
path: C:\Users\LX\Nong-View\model3_greenhouse_best_processed
train: images/train
val: images/val
test: images/test
nc: 2
names:
- Greenhouse_single
- Greenhouse_multi

dataset_info:
  total: 1483
  train: 1186
  val: 148
  test: 149
  processed: 1483
  augmented: 1186
  filtered: 0

preprocessing:
  method: optimized_stratified_split
  quality_filtering: true
  quality_threshold: 0.4
  augmentation: true
  augmentation_factor: 2
  random_seed: 42
```

---

## 🔬 고급 기능 상세

### 1️⃣ 품질 필터링

**평가 항목**:
- **흐림 감지**: Laplacian variance 계산
- **밝기**: 최적 밝기(127.5) 대비 편차
- **대비**: 표준편차 기반 대비 평가

**종합 점수** (0.0 ~ 1.0):
```python
quality_score = (
    blur_score * 0.5 +
    brightness_score * 0.3 +
    contrast_score * 0.2
)
```

**필터링 기준**:
- `quality_threshold = 0.4` 미만 제거
- IQR 방법으로 이상치 제거

---

### 2️⃣ 데이터 증강 (온실 특화)

**적용 변환**:

1. **좌우 반전** (50% 확률)
   - bbox 좌표 자동 조정

2. **밝기 조정**
   - 0.8 ~ 1.2배 랜덤

3. **색상 변화** (HSV)
   - 색조: ±10도
   - 채도: 0.8 ~ 1.2배
   - 명도: 0.9 ~ 1.1배

4. **미세 회전**
   - -5 ~ +5도 랜덤

**적용 대상**:
- ✅ **훈련 데이터만** 증강
- ❌ 검증/테스트 데이터는 원본 유지

---

### 3️⃣ 계층화 분할

**알고리즘**:

```python
# 클래스별 독립 분할
for each class:
    shuffle(class_images)
    train = class_images[:80%]
    val = class_images[80%:90%]
    test = class_images[90%:]

# 최종 셔플
shuffle(all_train)
shuffle(all_val)
shuffle(all_test)
```

**장점**:
- 클래스 비율 완벽 유지
- 과학적으로 검증된 방법
- 재현 가능 (random_seed)

---

## 🎓 YOLO 훈련 예시

전처리 완료 후 YOLO 훈련:

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolo11n.pt')

# 훈련 시작
results = model.train(
    data='C:/Users/LX/Nong-View/model3_greenhouse_best_processed/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='model3_greenhouse_best',
    project='runs/train',

    # 추가 설정 (선택)
    patience=30,
    save=True,
    plots=True
)

# 평가
metrics = model.val()

# 추론
results = model.predict('test_image.png')
```

---

## 📈 통계 분석

### processing_stats.json

```json
{
  "original_images": 1483,
  "processed_images": 1483,
  "augmented_images": 1186,
  "filtered_images": 0,
  "total_objects": 5134,
  "class_distribution": {
    "Greenhouse_single": 3204,
    "Greenhouse_multi": 1930
  },
  "processing_time": 485.23
}
```

---

## 🔧 설정 조정 가이드

### 품질 필터링 강도 조절

```python
# 엄격한 필터링 (고품질만)
quality_threshold=0.6

# 보통 (기본값)
quality_threshold=0.4

# 관대한 필터링 (대부분 통과)
quality_threshold=0.2

# 필터링 비활성화
enable_quality_filter=False
```

### 증강 배수 조절

```python
# 증강 없음
enable_augmentation=False

# 2배 (기본값)
augmentation_factor=2

# 3배 (더 많은 데이터)
augmentation_factor=3
```

### 분할 비율 조절

```python
# 기본 (8:1:1)
train_ratio=0.8
val_ratio=0.1
test_ratio=0.1

# 더 많은 검증 데이터 (7:2:1)
train_ratio=0.7
val_ratio=0.2
test_ratio=0.1
```

---

## ✅ 검증 체크리스트

전처리 후 확인:

- [ ] `model3_greenhouse_best_processed` 폴더 생성
- [ ] `data.yaml` 파일 존재
- [ ] `processing_stats.json` 파일 존재
- [ ] `images/train` 폴더에 원본 + 증강 이미지
- [ ] `images/val`, `images/test` 폴더에 원본 이미지
- [ ] `labels` 폴더의 라벨 파일 개수 = 이미지 개수
- [ ] 로그에 클래스 분포 확인
- [ ] 품질 필터링 통계 확인

---

## 📊 기대 효과

### optimized_preprocessing.py 기반 장점

1. **품질 향상**
   - 저품질 이미지 자동 제거
   - 일관된 데이터 품질

2. **성능 향상**
   - 클래스 균형 유지로 편향 감소
   - 데이터 증강으로 과적합 방지

3. **재현성**
   - 랜덤 시드로 동일 결과 보장
   - 과학적 분할 방법

4. **효율성**
   - 자동화된 전처리 파이프라인
   - 상세한 통계 및 로깅

---

## 🆘 문제 해결

### Q: 필수 패키지가 없다는 에러

```bash
pip install opencv-python pillow numpy pyyaml tqdm scikit-learn
```

### Q: 품질 필터링으로 너무 많이 제거됨

```python
# 임계값을 낮추거나
quality_threshold=0.3

# 필터링 비활성화
enable_quality_filter=False
```

### Q: 증강 이미지가 생성되지 않음

- 훈련 데이터만 증강됩니다
- `enable_augmentation=True` 확인
- 로그에서 증강 성공 여부 확인

### Q: 메모리 부족 에러

```python
# 증강 배수 감소
augmentation_factor=1  # 증강 없음
```

---

## 💡 추가 정보

### 관련 문서
- [CLAUDE.md](./CLAUDE.md): 프로젝트 전체 가이드
- [best/01_data_processing/optimized_preprocessing.py](./best/01_data_processing/optimized_preprocessing.py): 원본 스크립트

### 개발팀
- **작성자**: Claude Sonnet
- **기반**: optimized_preprocessing.py
- **날짜**: 2025-11-04

---

## 🎉 완료!

최적화된 전처리를 통해 더 나은 YOLO 모델 훈련이 가능합니다!

**Good luck! 🚀**
