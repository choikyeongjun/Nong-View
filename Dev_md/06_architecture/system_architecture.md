# Nong-View 시스템 아키텍처

## 1. 시스템 개요

### 1.1 목적
- 드론 정사영상을 활용한 농업 모니터링 자동화
- AI 기반 작물/시설물 자동 탐지
- 행정 보고용 공간정보 자동 생성

### 1.2 주요 구성요소
```
[드론 영상] → [전처리] → [AI 분석] → [후처리] → [GPKG/리포트]
     ↓           ↓           ↓           ↓           ↓
[데이터 레이크] ← [메타데이터 관리] → [결과 DB]
```

## 2. 기술 스택

### 2.1 백엔드
- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Task Queue**: Celery + Redis
- **Database**: PostgreSQL + PostGIS

### 2.2 AI/ML
- **Framework**: PyTorch
- **Model**: YOLOv11
- **Inference**: TorchServe / Triton
- **GPU**: CUDA 11.8+

### 2.3 GIS/영상처리
- **Raster**: GDAL, Rasterio
- **Vector**: GeoPandas, Shapely
- **Tiling**: rasterio + numpy
- **GPKG**: Fiona, GDAL

### 2.4 인프라
- **Container**: Docker
- **Orchestration**: Kubernetes
- **Storage**: MinIO (S3 compatible)
- **Monitoring**: Prometheus + Grafana

## 3. POD별 상세 아키텍처

### POD 1: 데이터 관리
```python
# 데이터 구조
DataRegistry:
  - image_id: UUID
  - file_path: str
  - capture_date: datetime
  - crs: str (EPSG:5186)
  - resolution: float
  - metadata: JSON
  
ShapeRegistry:
  - shape_id: UUID
  - pnu: str
  - geometry: Geometry
  - properties: JSON
```

### POD 2: 크로핑
```python
CropEngine:
  - input: raster + roi_geometry
  - process: clip_by_mask()
  - output: cropped_raster
  - index: spatial_index
```

### POD 3: 타일링
```python
TilingEngine:
  - tile_size: 640x640
  - overlap: 0.2 (20%)
  - output_format: 'TIFF'
  - naming: f"{region}_{date}_{row}_{col}.tif"
```

### POD 4: AI 분석
```python
InferenceEngine:
  - models: {
      'crop_detection': YOLOv11_crop,
      'facility_detection': YOLOv11_facility,
      'landuse_classification': YOLOv11_landuse
    }
  - batch_size: 16
  - confidence_threshold: 0.5
```

### POD 5: 병합
```python
MergeEngine:
  - nms_threshold: 0.5
  - aggregation: 'union'
  - statistics: ['count', 'area', 'length']
  - join_key: 'pnu'
```

### POD 6: GPKG 발행
```python
GPKGExporter:
  - layers: [
      'parcels',
      'crops',
      'facilities',
      'statistics'
    ]
  - crs: 'EPSG:5186'
  - format: 'GPKG 1.2'
```

## 4. 데이터 플로우

### 4.1 입력 데이터
```
[드론 촬영]
    ↓
[정사영상 (TIF/ECW)]
    ↓
[메타데이터 추출]
    ↓
[데이터 레지스트리 등록]
```

### 4.2 처리 파이프라인
```
[원본 영상]
    ↓
[ROI 추출 (Crop)]
    ↓
[타일 분할 (640x640)]
    ↓
[AI 추론 (YOLOv11)]
    ↓
[결과 병합]
    ↓
[통계 산출]
    ↓
[GPKG 생성]
```

### 4.3 출력 데이터
```
[GPKG 파일]
    ├── Layer: 필지 (parcels)
    ├── Layer: 작물 (crops)
    ├── Layer: 시설물 (facilities)
    └── Layer: 통계 (statistics)
```

## 5. API 설계

### 5.1 RESTful API
```
POST   /api/v1/images/upload
GET    /api/v1/images/{image_id}
POST   /api/v1/analysis/start
GET    /api/v1/analysis/{job_id}/status
GET    /api/v1/results/{result_id}
GET    /api/v1/export/gpkg/{area_id}
```

### 5.2 WebSocket (실시간 모니터링)
```
WS /ws/processing/{job_id}
  - progress: 0-100%
  - current_step: string
  - estimated_time: seconds
```

## 6. 데이터베이스 스키마

### 6.1 Core Tables
```sql
-- 영상 메타데이터
CREATE TABLE images (
    id UUID PRIMARY KEY,
    file_path VARCHAR(500),
    capture_date TIMESTAMP,
    crs VARCHAR(20),
    resolution FLOAT,
    metadata JSONB
);

-- 필지 정보
CREATE TABLE parcels (
    pnu VARCHAR(19) PRIMARY KEY,
    geometry GEOMETRY(Polygon, 5186),
    owner_info JSONB,
    crop_info JSONB
);

-- AI 분석 결과
CREATE TABLE detections (
    id UUID PRIMARY KEY,
    image_id UUID REFERENCES images(id),
    parcel_pnu VARCHAR(19) REFERENCES parcels(pnu),
    class_name VARCHAR(50),
    confidence FLOAT,
    geometry GEOMETRY,
    area FLOAT,
    created_at TIMESTAMP
);
```

## 7. 보안 및 권한

### 7.1 인증/인가
- JWT 기반 인증
- Role-based access control (RBAC)
- API Key for external systems

### 7.2 데이터 보안
- 민감정보 암호화 (AES-256)
- 개인정보 마스킹
- Audit logging

## 8. 성능 요구사항

### 8.1 처리량
- 영상 업로드: 100MB/s
- 타일 생성: 1000 tiles/min
- AI 추론: 100 tiles/min (per GPU)
- GPKG 생성: < 30s

### 8.2 응답시간
- API 응답: < 200ms (p95)
- 파일 업로드: < 5s (100MB)
- 분석 완료: < 10min (10km²)

## 9. 확장성

### 9.1 수평 확장
- Kubernetes HPA
- GPU node autoscaling
- Database read replicas

### 9.2 수직 확장
- GPU upgrade path
- Storage expansion
- Memory optimization

## 10. 모니터링

### 10.1 메트릭
- Processing queue depth
- GPU utilization
- API response time
- Error rate

### 10.2 알림
- Processing failure
- Queue overflow
- System resource alert

---

*Version: 1.0.0*
*Last Updated: 2025-10-26*