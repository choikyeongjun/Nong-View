# Nong-View 통합 Jupyter 노트북 사용 가이드

**문서 버전**: v1.0  
**작성일**: 2025-10-27  
**대상**: 개발자, 테스터, 관리자  

---

## 📖 개요

### 목적
이 가이드는 Nong-View 프로젝트의 통합 Jupyter 노트북 `nongview_v1.ipynb` 사용법을 설명합니다.

### 노트북 특징
- **완전한 통합**: 모든 POD 모듈과 API가 하나의 파일에 포함
- **실시간 테스트**: 코드 변경 즉시 테스트 가능
- **단계별 실행**: 섹션별로 독립적 실행 지원
- **문서화 통합**: 코드와 설명이 함께 제공

---

## 🚀 시작하기

### 1. 환경 준비
```bash
# 필수 패키지 설치 (첫 번째 셀 실행)
pip install fastapi uvicorn sqlalchemy alembic psycopg2-binary
pip install rasterio geopandas shapely ultralytics torch
pip install pillow opencv-python numpy pandas
```

### 2. 노트북 실행 순서
```python
# 권장 실행 순서
1. 🔧 환경 설정 및 의존성 설치
2. ⚙️ 설정 및 구성
3. 🗄️ 데이터베이스 모델 정의
4. 🔗 데이터베이스 연결 관리  
5. 🔌 POD 모듈 통합
6. 📋 API 스키마 정의
7. 🚀 FastAPI 애플리케이션
8. 📸 Images API 구현
9. 🔬 Analysis API 구현
10. 🖥️ 서버 시작 및 테스트
11. 🧪 API 테스트
12. 📊 샘플 데이터 생성
13. 📈 개발 완료 상태 체크
```

---

## 📁 노트북 구조

### 주요 섹션별 기능

#### 1. 환경 설정 (1-2번 셀)
```python
# 기능
- 필수 패키지 자동 설치
- 라이브러리 임포트
- 로깅 설정
```

#### 2. 설정 관리 (3번 셀)
```python
# Settings 클래스
class Settings:
    PROJECT_NAME: str = "Nong-View API"
    VERSION: str = "1.0.0"
    DATABASE_URL: str = "sqlite:///./nongview_test.db"
    # ... 기타 설정
```

#### 3. 데이터베이스 (4-5번 셀)
```python
# SQLAlchemy 모델
- User: 사용자 관리
- Image: 이미지 메타데이터
- Analysis: 분석 작업
- CropResult: 크로핑 결과  
- Export: 내보내기 작업
```

#### 4. POD 모듈 (6-8번 셀)
```python
# 통합된 모듈
- DataRegistry: 데이터 수집
- CroppingEngine: 이미지 크로핑
- AIInferenceEngine: AI 추론
```

#### 5. API 구현 (9-11번 셀)
```python
# RESTful API 엔드포인트
- /api/v1/images/upload
- /api/v1/images
- /api/v1/analyses
- /health
```

#### 6. 테스트 및 검증 (12-13번 셀)
```python
# 자동 테스트
- API 엔드포인트 테스트
- 샘플 데이터 생성
- 상태 체크
```

---

## 🔧 사용법

### 1. 전체 실행
```python
# 모든 셀을 순서대로 실행
Cell → Run All
```

### 2. 단계별 실행
```python
# 각 섹션별로 실행
Shift + Enter (현재 셀 실행)
Ctrl + Enter (현재 셀 실행 후 다음으로 이동)
```

### 3. 서버 시작
```python
# 10번 셀 실행 후 자동으로 서버 시작
# 접속 URL
http://127.0.0.1:8000        # 메인 페이지
http://127.0.0.1:8000/health # 헬스 체크
http://127.0.0.1:8000/api/docs # API 문서
```

### 4. API 테스트
```python
# 11번 셀에서 자동 테스트 실행
test_health_check()
test_image_list() 
test_analysis_list()
```

---

## 🧪 개발 워크플로우

### 1. 새로운 기능 개발
```python
# 단계
1. 해당 섹션 셀 수정
2. 셀 실행으로 즉시 테스트
3. API 테스트 셀로 검증
4. 문제 발견 시 즉시 수정
```

### 2. 디버깅 프로세스
```python
# 디버깅 방법
1. 에러 발생 셀 확인
2. 변수 상태 print로 체크
3. 로그 메시지 확인
4. 단계적 실행으로 원인 추적
```

### 3. 데이터베이스 초기화
```python
# DB 리셋이 필요한 경우
import os
if os.path.exists("./nongview_test.db"):
    os.remove("./nongview_test.db")

# 4번 셀 다시 실행 (데이터베이스 모델 정의)
```

---

## 📊 모니터링 및 상태 확인

### 1. 실시간 로그 확인
```python
# 로그 레벨별 확인
logger.info("정보 메시지")
logger.warning("경고 메시지") 
logger.error("에러 메시지")
```

### 2. 데이터베이스 상태
```python
# 13번 셀에서 자동 체크
check_development_status()

# 출력 예시
📊 데이터베이스 상태:
  👥 사용자: 1명
  🖼️ 이미지: 0개
  🔬 분석: 0개
  ✂️ 크롭 결과: 0개
```

### 3. API 서버 상태
```python
# 헬스 체크로 서버 상태 확인
GET /health

# 응답 예시
{
  "success": true,
  "message": "서비스가 정상 작동 중입니다",
  "data": {
    "timestamp": "2025-10-27T...",
    "version": "1.0.0",
    "environment": "development"
  }
}
```

---

## 🛠️ 커스터마이징

### 1. 설정 변경
```python
# 3번 셀의 Settings 클래스 수정
class Settings:
    HOST: str = "0.0.0.0"        # 외부 접속 허용
    PORT: int = 8001             # 포트 변경
    DATABASE_URL: str = "postgresql://..." # PostgreSQL 사용
```

### 2. POD 모듈 확장
```python
# 새로운 POD 모듈 추가
class NewPODModule:
    def __init__(self):
        # 초기화 코드
        pass
    
    def process(self, data):
        # 처리 로직
        return result
```

### 3. API 엔드포인트 추가
```python
# 새로운 API 엔드포인트
@app.get("/api/v1/new-endpoint")
async def new_endpoint():
    return {"message": "새로운 엔드포인트"}
```

---

## 🚨 문제 해결

### 일반적인 문제들

#### 1. 패키지 설치 실패
```bash
# 해결방법
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. 포트 충돌
```python
# 다른 포트 사용
settings.PORT = 8001  # 또는 다른 포트
```

#### 3. 데이터베이스 연결 오류
```python
# SQLite 파일 권한 확인
import os
os.chmod("./nongview_test.db", 0o666)
```

#### 4. AI 모델 로딩 실패
```python
# 기본 모델 다운로드 확인
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # 자동 다운로드
```

### 성능 최적화

#### 1. 메모리 사용량 줄이기
```python
# 대용량 이미지 처리 시
import gc
gc.collect()  # 가비지 컬렉션 강제 실행
```

#### 2. 데이터베이스 최적화
```python
# 연결 풀 크기 조정
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10
)
```

---

## 📚 참고 자료

### 노트북 내 문서
- 각 셀의 마크다운 설명
- 코드 주석
- 실행 결과 예시

### 외부 문서
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)
- [Rasterio 문서](https://rasterio.readthedocs.io/)
- [YOLO 문서](https://docs.ultralytics.com/)

### 프로젝트 문서
- `Dev_md/03_guides/api_design_guide.md`
- `Dev_md/03_guides/database_usage_guide.md`
- `Dev_md/06_architecture/system_architecture.md`

---

## 💡 팁과 요령

### 개발 효율성
1. **자주 저장**: Ctrl+S로 정기적 저장
2. **단계적 실행**: 전체 실행보다 섹션별 실행 권장
3. **로그 활용**: print 대신 logger 사용
4. **에러 핸들링**: try-except로 안전한 실행

### 협업
1. **버전 관리**: 파일명에 버전 표기 (v1.0, v1.1)
2. **주석 추가**: 수정 사항은 주석으로 기록
3. **백업**: 중요한 변경 전 파일 백업
4. **문서화**: 새로운 기능은 마크다운으로 설명 추가

---

## 🔄 업데이트 이력

| 버전 | 날짜 | 변경사항 |
|------|------|----------|
| v1.0 | 2025-10-27 | 초기 버전 생성 |

---

**작성자**: Claude Sonnet  
**검토자**: -  
**승인자**: -  
**다음 업데이트**: 필요시