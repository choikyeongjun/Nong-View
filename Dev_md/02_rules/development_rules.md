# Nong-View 개발 규칙 및 가이드라인

## 1. 프로젝트 규칙

### 1.1 브랜치 전략
```
main
├── develop
│   ├── feature/pod1-data-ingestion
│   ├── feature/pod2-cropping
│   ├── feature/pod3-tiling
│   ├── feature/pod4-ai-inference
│   ├── feature/pod5-merging
│   └── feature/pod6-gpkg-export
└── release/v1.0.0
```

### 1.2 커밋 메시지 규칙
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- feat: 새로운 기능
- fix: 버그 수정
- docs: 문서 수정
- style: 코드 포맷팅
- refactor: 리팩토링
- test: 테스트 코드
- chore: 빌드, 패키지 관련

**Examples:**
```
feat(pod1): Add TIF file upload handler
fix(pod4): Resolve GPU memory leak in inference
docs(api): Update API documentation for v1.0
```

## 2. 코딩 컨벤션

### 2.1 Python
```python
# 파일명: snake_case
data_registry.py

# 클래스명: PascalCase
class DataRegistry:
    pass

# 함수명: snake_case
def process_image():
    pass

# 상수: UPPER_SNAKE_CASE
MAX_TILE_SIZE = 640
DEFAULT_OVERLAP = 0.2

# Private 메서드: underscore prefix
def _validate_input():
    pass
```

### 2.2 Type Hints
```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def process_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2
) -> List[Tuple[np.ndarray, Dict[str, float]]]:
    """
    Process image into tiles
    
    Args:
        image: Input image array
        tile_size: Size of each tile
        overlap: Overlap ratio between tiles
        
    Returns:
        List of (tile_array, metadata) tuples
    """
    pass
```

### 2.3 Docstring
```python
def complex_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Detailed explanation of what the function does,
    including any important algorithms or logic.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        TypeError: When param2 is not integer
        
    Example:
        >>> result = complex_function("test", 123)
        >>> print(result)
        True
    """
    pass
```

## 3. 디렉토리 구조

### 3.1 프로젝트 구조
```
Nong-View/
├── src/
│   ├── pod1_data_ingestion/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── validators.py
│   │   └── tests/
│   ├── pod2_cropping/
│   ├── pod3_tiling/
│   ├── pod4_ai_inference/
│   ├── pod5_merging/
│   ├── pod6_gpkg_export/
│   └── common/
│       ├── config.py
│       ├── database.py
│       ├── exceptions.py
│       └── utils.py
├── api/
│   ├── v1/
│   │   ├── endpoints/
│   │   ├── schemas/
│   │   └── dependencies.py
│   └── main.py
├── models/
│   ├── weights/
│   └── configs/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
├── docs/
├── docker/
├── k8s/
└── requirements/
    ├── base.txt
    ├── dev.txt
    └── prod.txt
```

## 4. 데이터 규칙

### 4.1 파일 명명 규칙
```
# 원본 영상
{region}_{date}_{type}.{ext}
예: namwon_20250115_ortho.tif

# 크롭된 영상
{region}_{date}_{pnu}_crop.tif
예: namwon_20250115_4513010100100010000_crop.tif

# 타일
{region}_{date}_{row}_{col}.tif
예: namwon_20250115_001_001.tif

# 결과 파일
{region}_{date}_{analysis_type}_result.gpkg
예: namwon_20250115_crop_detection_result.gpkg
```

### 4.2 좌표계 표준
- **기본 좌표계**: EPSG:5186 (Korea 2000 / Central Belt 2010)
- **WGS84 변환**: 필요시에만 EPSG:4326으로 변환
- **단위**: 미터(m)

### 4.3 메타데이터 스키마
```json
{
  "image_id": "uuid",
  "capture_date": "2025-01-15T10:30:00Z",
  "crs": "EPSG:5186",
  "resolution": 0.25,
  "bounds": {
    "minx": 123456.78,
    "miny": 234567.89,
    "maxx": 345678.90,
    "maxy": 456789.01
  },
  "source": {
    "drone_model": "DJI Matrice 300",
    "camera": "Zenmuse P1",
    "altitude": 150,
    "overlap": 0.8
  }
}
```

## 5. API 규칙

### 5.1 REST API 설계
```
# 리소스 명명: 복수형 명사
/api/v1/images
/api/v1/analyses
/api/v1/results

# HTTP 메서드
GET    : 조회
POST   : 생성
PUT    : 전체 수정
PATCH  : 부분 수정
DELETE : 삭제

# 상태 코드
200 : OK
201 : Created
204 : No Content
400 : Bad Request
401 : Unauthorized
403 : Forbidden
404 : Not Found
422 : Unprocessable Entity
500 : Internal Server Error
```

### 5.2 응답 포맷
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "attributes": {}
  },
  "meta": {
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "1.0.0"
  }
}

// 에러 응답
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": []
  }
}
```

## 6. 테스트 규칙

### 6.1 테스트 파일 명명
```python
# 테스트 파일: test_ prefix
test_data_registry.py

# 테스트 클래스: Test prefix
class TestDataRegistry:
    pass

# 테스트 메서드: test_ prefix
def test_upload_image():
    pass
```

### 6.2 테스트 구조
```python
import pytest

class TestFeature:
    """Feature 테스트"""
    
    @pytest.fixture
    def setup_data(self):
        """테스트 데이터 설정"""
        return {"test": "data"}
    
    def test_happy_path(self, setup_data):
        """정상 케이스"""
        assert True
        
    def test_edge_case(self):
        """경계 케이스"""
        pass
        
    def test_error_handling(self):
        """에러 처리"""
        with pytest.raises(ValueError):
            raise ValueError("Test error")
```

## 7. 보안 규칙

### 7.1 민감정보 처리
- 환경변수로 관리 (.env 파일)
- 절대 하드코딩 금지
- Git에 커밋 금지

### 7.2 개인정보 마스킹
```python
def mask_personal_info(data: dict) -> dict:
    """개인정보 마스킹"""
    masked = data.copy()
    if 'owner_name' in masked:
        masked['owner_name'] = masked['owner_name'][:1] + '**'
    if 'phone' in masked:
        masked['phone'] = masked['phone'][:7] + '****'
    return masked
```

## 8. 로깅 규칙

### 8.1 로그 레벨
```python
import logging

# DEBUG: 상세 디버그 정보
logger.debug(f"Processing tile {tile_id}")

# INFO: 일반 정보
logger.info(f"Analysis started for {image_id}")

# WARNING: 경고 (처리는 계속)
logger.warning(f"Low confidence detection: {confidence}")

# ERROR: 에러 (복구 가능)
logger.error(f"Failed to process tile: {error}")

# CRITICAL: 심각한 에러 (시스템 중단)
logger.critical(f"Database connection lost")
```

### 8.2 로그 포맷
```python
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
```

## 9. 문서화 규칙

### 9.1 README 구조
```markdown
# Project Name

## Overview
간단한 프로젝트 설명

## Features
- 주요 기능 목록

## Installation
설치 방법

## Usage
사용 방법

## API Documentation
API 문서 링크

## Contributing
기여 가이드

## License
라이선스 정보
```

### 9.2 코드 주석
```python
# TODO: 구현 필요
# FIXME: 버그 수정 필요
# NOTE: 중요 참고사항
# HACK: 임시 해결책
# XXX: 주의 필요
```

## 10. 성능 규칙

### 10.1 최적화 우선순위
1. 정확성 (Correctness)
2. 명확성 (Clarity)
3. 성능 (Performance)

### 10.2 벤치마크 기준
- 타일 생성: < 100ms/tile
- AI 추론: < 500ms/tile
- 병합 처리: < 5s/parcel
- API 응답: < 200ms (p95)

---

*Version: 1.0.0*
*Last Updated: 2025-10-26*
*Next Review: 2025-11-02*