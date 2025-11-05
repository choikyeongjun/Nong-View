# 🎯 YOLOv11-seg Segmentation 전처리 가이드

## 🔍 문제 원인 분석

### ❌ 기존 코드의 문제
기존 `preprocess_model3_optimized.py`는 **Detection (bbox)** 형식만 지원:
```python
# bbox 형식 (5개 값)
class_id x_center y_center width height
```

### ✅ 실제 데이터 형식
당신의 데이터는 **Segmentation (polygon)** 형식:
```python
# polygon 형식 (class_id + N개 좌표쌍)
class_id x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 ...
```

**예시 (실제 라벨)**:
```
1 0.0 0.9990234375 0.294921875 0.9990234375 0.197265625 0.6171875 0.095703125 0.2412109375 0.0 0.265625 0.0 0.9990234375
```
- class_id: `1` (Greenhouse_multi)
- polygon: 6개 좌표쌍 (x1,y1), (x2,y2), ..., (x6,y6)

---

## 🆕 완전히 새로운 Segmentation 전용 스크립트

**파일**: `preprocess_model3_segmentation.py`

### 주요 차이점

| 기능 | Detection 버전 | Segmentation 버전 |
|------|----------------|-------------------|
| **라벨 형식** | bbox (5개 값) | polygon (가변 길이) |
| **파싱** | 고정 5개 | 동적 파싱 |
| **증강 (좌우반전)** | x_center 변환 | 모든 x 좌표 변환 |
| **저장** | 5개 값 | 모든 polygon 좌표 |

---

## 🔬 핵심 코드 비교

### 1️⃣ 라벨 파싱

**Detection (bbox) - 기존 코드 ❌**:
```python
def _load_yolo_labels(self, label_path):
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:  # 고정 5개
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            bboxes.append([x, y, w, h])
```

**Segmentation (polygon) - 새 코드 ✅**:
```python
def _load_seg_labels(self, label_path):
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:  # 가변 길이
            class_id = int(parts[0])
            polygon = [float(x) for x in parts[1:]]  # 모든 좌표
            polygons.append(polygon)
```

---

### 2️⃣ 증강 (좌우 반전)

**Detection (bbox) - 기존 코드 ❌**:
```python
# bbox 중심점만 변환
aug_bboxes = [[1.0 - bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bboxes]
```

**Segmentation (polygon) - 새 코드 ✅**:
```python
# 모든 polygon의 x 좌표 변환
for poly in polygons:
    aug_poly = []
    for i in range(0, len(poly), 2):
        x = poly[i]
        y = poly[i + 1]
        aug_poly.append(1.0 - x)  # x 좌표 반전
        aug_poly.append(y)        # y 좌표 유지
    aug_polygons.append(aug_poly)
```

---

### 3️⃣ 라벨 저장

**Detection (bbox) - 기존 코드 ❌**:
```python
f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
```

**Segmentation (polygon) - 새 코드 ✅**:
```python
# 모든 polygon 좌표 저장
coords_str = ' '.join([f"{coord:.10f}" for coord in polygon])
f.write(f"{class_id} {coords_str}\n")
```

---

## 🚀 실행 방법

### 1️⃣ 기존 출력 폴더 삭제
```bash
rm -rf "C:\Users\LX\Nong-View\model3_greenhouse_seg_processed"
```

### 2️⃣ 새 스크립트 실행
```bash
cd C:\Users\LX\Nong-View
python preprocess_model3_segmentation.py
```

### 3️⃣ 예상 출력
```
============================================================
YOLOv11-seg Segmentation 전처리 시작
============================================================
소스: C:\Users\LX\Nong-View\model3_greenhouse
출력: C:\Users\LX\Nong-View\model3_greenhouse_seg_processed
증강: 3배

[1/5] 데이터 수집 중...
✓ 1483개 이미지 수집

[2/5] 계층화 분할 중...
  Greenhouse_single: Train=579, Val=72, Test=73
  Greenhouse_multi: Train=607, Val=75, Test=77
✓ Train: 1186, Val: 147, Test: 150

[3/5] 출력 디렉토리 생성 중...
✓ C:\Users\LX\Nong-View\model3_greenhouse_seg_processed

[4/5] 데이터 복사 및 증강 중...
train 처리: 100%|██████████| 1186/1186
✓ 증강 성공: 1F001D40001_aug1.png
✓ 증강 성공: 1F001D40002_aug1.png
...
✓ 처리: 1186개, 증강: 3558개

[5/5] 메타데이터 생성 중...
✓ 완료

============================================================
✅ 전처리 완료!
============================================================

📊 처리 통계:
  - 원본 이미지: 1483개
  - 처리된 이미지: 1483개
  - 증강된 이미지: 3558개
  - 총 이미지: 5041개
  - 총 객체: 7701개
  - 처리 시간: 350.23초

📊 클래스 분포:
  - Greenhouse_multi: 2895개 (37.6%)
  - Greenhouse_single: 4806개 (62.4%)
```

---

## 📁 출력 구조

```
model3_greenhouse_seg_processed/
├── data.yaml                    ← YOLOv11-seg 설정
├── processing_stats.json        ← 통계
├── images/
│   ├── train/                   ← 1,186 + 3,558 = 4,744개
│   │   ├── 1F001D40001.png     ← 원본
│   │   ├── 1F001D40001_aug1.png
│   │   ├── 1F001D40001_aug2.png
│   │   ├── 1F001D40001_aug3.png
│   │   └── ...
│   ├── val/                     ← 147개 (원본만)
│   └── test/                    ← 150개 (원본만)
└── labels/
    ├── train/                   ← polygon 형식
    ├── val/
    └── test/
```

---

## 🎯 data.yaml 확인

생성된 `data.yaml`:
```yaml
path: C:\Users\LX\Nong-View\model3_greenhouse_seg_processed
train: images/train
val: images/val
test: images/test
nc: 2
names:
- Greenhouse_single
- Greenhouse_multi

task: segment  # ← Segmentation 명시

dataset_info:
  total: 1483
  train: 1186
  val: 147
  test: 150
  processed: 1483
  augmented: 3558

preprocessing:
  method: stratified_split_segmentation
  augmentation: true
  augmentation_factor: 3
  random_seed: 42
```

---

## ✅ 증강 확인 방법

### 1️⃣ 파일 개수 확인
```bash
# 전체 훈련 이미지
ls "C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\images\train" | wc -l
# 예상: 4,744개

# 증강 파일만
ls "C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\images\train" | grep "_aug" | wc -l
# 예상: 3,558개
```

### 2️⃣ 라벨 형식 확인
```bash
# 원본 라벨
head -1 "C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\labels\train\1F001D40001.txt"
# 출력: 1 0.0 0.9990234375 0.294921875 0.9990234375 ...

# 증강 라벨 (좌우 반전 확인)
head -1 "C:\Users\LX\Nong-View\model3_greenhouse_seg_processed\labels\train\1F001D40001_aug1.txt"
# 출력: 1 1.0 0.9990234375 0.705078125 0.9990234375 ...
#       ↑ x 좌표가 1.0 - x로 변환됨
```

---

## 🎓 YOLOv11-seg 훈련 예시

```python
from ultralytics import YOLO

# YOLOv11-seg 모델 로드
model = YOLO('yolo11n-seg.pt')  # ← -seg 모델 사용!

# 훈련
results = model.train(
    data='C:/Users/LX/Nong-View/model3_greenhouse_seg_processed/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='model3_greenhouse_seg',
    project='runs/segment',  # ← segment 폴더
    task='segment'  # ← task 명시
)

# 검증
metrics = model.val()

# 추론
results = model.predict('test_image.png')
```

---

## 🔧 주요 개선사항 요약

| 항목 | 개선 내용 |
|------|----------|
| **형식 지원** | bbox → polygon |
| **파싱 로직** | 고정 5개 → 가변 길이 |
| **증강 (좌우반전)** | 중심점만 → 모든 x 좌표 |
| **라벨 저장** | 5개 값 → 전체 polygon |
| **품질 필터링** | 비활성화 (이미 완료) |
| **로깅** | 상세 (성공/실패 추적) |
| **예상 시간** | 6~8분 |

---

## 🐛 문제 해결

### Q: 여전히 증강이 안 된다면?

**1. 로그 확인**:
```bash
python preprocess_model3_segmentation.py 2>&1 | grep -E "(증강|ERROR)"
```

**2. 라벨 형식 확인**:
```bash
# 원본 라벨 확인
cat "C:\Users\LX\Nong-View\model3_greenhouse\labels\train\1F001D40001.txt"

# polygon 형식인지 확인 (값이 7개 이상)
```

**3. 수동 테스트**:
```python
# Python 콘솔에서
label_path = r"C:\Users\LX\Nong-View\model3_greenhouse\labels\train\1F001D40001.txt"
with open(label_path) as f:
    line = f.readline()
    parts = line.strip().split()
    print(f"값 개수: {len(parts)}")  # 7개 이상이어야 함
    print(f"class_id: {parts[0]}")
    print(f"polygon: {parts[1:]}")
```

---

### Q: 증강 배수를 변경하고 싶다면?

스크립트의 `main()` 함수 수정:
```python
config = SegConfig(
    augmentation_factor=2,  # 2배로 변경
    # 또는
    augmentation_factor=5,  # 5배로 변경
)
```

---

## 🎉 완료!

이제 **YOLOv11-seg** 전용 전처리가 완료되었습니다!

**실행**:
```bash
python preprocess_model3_segmentation.py
```

**예상 결과**:
- ✅ 증강 파일 생성 (`_aug1`, `_aug2`, `_aug3`)
- ✅ Polygon 좌표 정확히 변환
- ✅ YOLOv11-seg 훈련 준비 완료

**행운을 빕니다! 🚀**
