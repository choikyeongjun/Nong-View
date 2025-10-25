# POD별 개발 가이드

## POD 1: 파일 가져오기 (데이터 관리)

### 담당: Opus (아키텍처) + Sonnet (구현)

#### Opus 작업 영역
```python
# 데이터 레지스트리 설계
class DataRegistry:
    """
    전체 파이프라인에서 참조할 데이터 관리 시스템
    - 유니크 ID 체계
    - 메타데이터 스키마
    - 버전 관리
    """
    
# 좌표계 정합 엔진
class CoordinateValidator:
    """
    - EPSG:5186 (Korean 2000)으로 통일
    - 좌표계 변환 로직
    - Geometry 검증
    """
```

#### Sonnet 작업 영역
```python
# REST API 구현
@app.post("/api/v1/images/upload")
async def upload_image(file: UploadFile):
    """TIF/ECW 파일 업로드"""
    
@app.post("/api/v1/shapes/upload")
async def upload_shapefile(file: UploadFile):
    """SHP 파일 업로드 및 파싱"""
```

---

## POD 2: 크로핑 (ROI 추출)

### 담당: Sonnet (전담)

#### 구현 체크리스트
- [ ] Convex Hull 계산 함수
- [ ] GDAL clip_by_mask 구현
- [ ] 크롭 결과 저장 로직
- [ ] 크롭 메타데이터 관리

#### 코드 예시
```python
import rasterio
from rasterio.mask import mask
import geopandas as gpd

def crop_raster_by_roi(raster_path, roi_geometry):
    """
    래스터를 ROI로 크롭
    """
    with rasterio.open(raster_path) as src:
        cropped, transform = mask(src, [roi_geometry], crop=True)
        # 크롭된 래스터 저장
        
def calculate_convex_hull(geometries):
    """
    필지 집합의 Convex Hull 계산
    """
    union = geometries.unary_union
    return union.convex_hull
```

---

## POD 3: 타일링

### 담당: Opus (전담)

#### 구현 요구사항
- 640x640 픽셀 타일
- 20% 겹침 (overlap)
- 타일 인덱스 관리
- 병렬 처리 지원

#### 알고리즘
```python
class TilingEngine:
    def __init__(self, tile_size=640, overlap=0.2):
        self.tile_size = tile_size
        self.overlap = overlap
        
    def generate_tiles(self, image_array, geo_transform):
        """
        이미지를 타일로 분할
        Returns: [(tile_array, tile_bbox, tile_id), ...]
        """
        stride = int(tile_size * (1 - overlap))
        # 타일 생성 로직
        
    def create_tile_index(self, tiles):
        """
        공간 인덱스 생성 (R-tree)
        """
```

---

## POD 4: AI 분석

### 담당: Opus (엔진) + Sonnet (API)

#### Opus - 추론 엔진
```python
class InferenceEngine:
    def __init__(self):
        self.models = {
            'crop': self.load_model('yolov11_crop.pt'),
            'facility': self.load_model('yolov11_facility.pt'),
            'landuse': self.load_model('yolov11_landuse.pt')
        }
    
    def predict(self, tile, model_type):
        """
        타일 단위 추론
        Returns: detections (boxes, classes, scores)
        """
        
class ModelVersionManager:
    """
    모델 버전 관리
    A/B 테스트 지원
    """
```

#### Sonnet - API & 결과 저장
```python
@app.post("/api/v1/inference/batch")
async def batch_inference(tiles: List[TileData]):
    """배치 추론 API"""
    
def save_inference_results(results, tile_id):
    """
    추론 결과 DB 저장
    - Detection 정보
    - 메타데이터
    - 성능 메트릭
    """
```

---

## POD 5: 병합

### 담당: Opus (알고리즘) + Sonnet (통계)

#### Opus - 병합 알고리즘
```python
class MergeEngine:
    def merge_tile_results(self, tile_results, roi_geometry):
        """
        타일 결과를 ROI 단위로 병합
        - NMS (Non-Maximum Suppression)
        - Boundary 처리
        """
        
    def resolve_overlaps(self, detections):
        """
        겹치는 detection 해결
        """
```

#### Sonnet - 통계 산출
```python
def calculate_statistics(merged_results, parcel_info):
    """
    필지별 통계 계산
    - 작물 면적
    - 시설물 개수
    - 휴경지 비율
    """
    
def join_with_admin_data(spatial_results, admin_db):
    """
    행정 데이터와 조인
    - PNU 기준
    - 소유자 정보
    - 보조금 이력
    """
```

---

## POD 6: GPKG 발행

### 담당: Sonnet (전담)

#### 구현 체크리스트
- [ ] GPKG 생성 함수
- [ ] 레이어 구조 정의
- [ ] 민감정보 마스킹
- [ ] 다운로드 API

#### 코드 구조
```python
import geopandas as gpd
from shapely.geometry import shape

class GPKGExporter:
    def __init__(self):
        self.layers = {}
        
    def add_layer(self, name, gdf):
        """레이어 추가"""
        self.layers[name] = gdf
        
    def export(self, output_path):
        """GPKG 파일 생성"""
        for name, gdf in self.layers.items():
            gdf.to_file(output_path, layer=name, driver="GPKG")
            
    def mask_sensitive_data(self, gdf):
        """민감정보 제거/마스킹"""
        # 개인정보 필드 처리
```

---

## 공통 개발 규칙

### 1. 코드 스타일
- PEP 8 준수
- Type hints 사용
- Docstring 필수

### 2. 테스트
- Unit test coverage > 80%
- Integration test 필수
- Performance test

### 3. 로깅
```python
import logging

logger = logging.getLogger(__name__)

# 로그 레벨
logger.debug("디버그 정보")
logger.info("처리 상태")
logger.warning("경고")
logger.error("오류")
```

### 4. 에러 처리
```python
class NongViewException(Exception):
    """Base exception"""
    
class DataValidationError(NongViewException):
    """데이터 검증 실패"""
    
class ProcessingError(NongViewException):
    """처리 중 오류"""
```

### 5. 설정 관리
```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    minio_endpoint: str
    gpu_device: int = 0
    
    class Config:
        env_file = ".env"
```

---

## 개발 진행 순서

### Phase 1: 기반 구축 (Week 1-2)
1. 프로젝트 구조 생성
2. Docker 환경 구성
3. DB 스키마 생성
4. 기본 API 프레임워크

### Phase 2: 데이터 파이프라인 (Week 3-4)
1. 파일 업로드 구현
2. 크로핑 모듈 개발
3. 타일링 엔진 구현
4. 데이터 레지스트리 구축

### Phase 3: AI 시스템 (Week 5-6)
1. 모델 로딩 시스템
2. 추론 엔진 구현
3. 결과 저장 로직
4. 성능 최적화

### Phase 4: 통합 및 출력 (Week 7-8)
1. 병합 알고리즘 구현
2. 통계 산출 기능
3. GPKG Export
4. 대시보드 개발

---

## 테스트 시나리오

### 1. 단위 테스트
- 각 POD 모듈별 테스트
- Mock 데이터 사용

### 2. 통합 테스트
- 전체 파이프라인 테스트
- 실제 샘플 데이터 사용

### 3. 성능 테스트
- 10km² 영역 처리
- 1000개 타일 동시 처리
- GPU 사용률 모니터링

### 4. 정확도 테스트
- Ground truth와 비교
- Precision/Recall 측정
- 오탐/미탐 분석

---

*Last Updated: 2025-10-26*