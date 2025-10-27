# Nong-View 데이터베이스 로컬 테스트 환경 가이드

## 📋 개요
Nong-View 프로젝트의 데이터베이스 구현을 위한 로컬 테스트 환경이 구축되었습니다.

## 🗂️ 생성된 파일 구조
```
D:\Nong-View\
├── .env                      # 환경 설정 파일
├── alembic.ini              # Alembic 마이그레이션 설정
├── alembic/                 # 마이그레이션 스크립트 디렉토리
│   ├── env.py
│   ├── script.py.mako
│   └── README
├── src/
│   └── database/
│       ├── __init__.py
│       ├── database.py      # DB 연결 및 세션 관리
│       └── models.py        # SQLAlchemy 모델 정의
├── test_db.py              # DB 테스트 스크립트
├── run_test.bat            # 테스트 실행 배치 파일
└── README_DB.md            # 본 문서
```

## 📊 데이터베이스 모델 구조

### 1. **Image** (영상 정보)
- 드론 정사영상 메타데이터 저장
- 좌표계, 크기, 촬영 정보 등

### 2. **Analysis** (분석 작업)
- AI 분석 작업 관리
- 진행 상황 및 상태 추적

### 3. **Result** (분석 결과)
- 객체 탐지/분류 결과
- 신뢰도, 지오메트리, 속성 정보

### 4. **Tile** (타일 정보)
- 640x640 타일 관리
- 타일 위치 및 인덱스

### 5. **TileResult** (타일별 분석 결과)
- 각 타일의 AI 추론 결과

### 6. **Parcel** (필지 정보)
- PNU 기반 필지 정보
- 소유자, 경작 상태 등

### 7. **ParcelStatistics** (필지별 통계)
- 작물 면적, 시설물 수 등

### 8. **Export** (내보내기 작업)
- GPKG 발행 작업 관리

## 🚀 테스트 환경 실행 방법

### 방법 1: 배치 파일 사용 (권장)
```bash
D:\Nong-View\run_test.bat
```

### 방법 2: 수동 실행
```bash
# 1. 필요한 패키지 설치
C:\Users\ASUS\anaconda3\python.exe -m pip install sqlalchemy python-dotenv alembic

# 2. 테스트 스크립트 실행
C:\Users\ASUS\anaconda3\python.exe D:\Nong-View\test_db.py
```

## ✅ 테스트 항목

테스트 스크립트는 다음 항목들을 검증합니다:

1. **데이터베이스 연결 테스트**
   - SQLite 데이터베이스 생성
   - 연결 확인

2. **CRUD 작업 테스트**
   - Image: 생성, 조회, 수정
   - Analysis: 작업 생성 및 진행률 업데이트
   - Result: 다중 결과 저장
   - Parcel: 필지 정보 관리
   - Tile: 타일 생성 및 조회

3. **관계 테스트**
   - Image ↔ Analysis
   - Analysis ↔ Result
   - Image ↔ Tile

## 🔧 환경 설정 (.env)

```env
# SQLite (개발용)
DATABASE_URL=sqlite:///D:/Nong-View/nongview.db

# PostgreSQL (운영용 - 필요시 변경)
# DATABASE_URL=postgresql://user:password@localhost:5432/nongview
```

## 📝 데이터베이스 마이그레이션

### 초기 마이그레이션 생성
```bash
cd D:\Nong-View
alembic revision --autogenerate -m "Initial migration"
```

### 마이그레이션 적용
```bash
alembic upgrade head
```

### 마이그레이션 롤백
```bash
alembic downgrade -1
```

## 🎯 다음 단계

1. **API 엔드포인트 구현**
   - FastAPI 라우터 작성
   - CRUD 서비스 레이어

2. **비동기 처리**
   - Celery 워커 설정
   - Redis 연동

3. **성능 최적화**
   - 인덱스 추가
   - 쿼리 최적화

## ⚠️ 주의사항

- 현재 SQLite를 사용 중이며, 운영 환경에서는 PostgreSQL 사용 권장
- 민감한 정보(개인정보)는 마스킹 처리 필요
- 대용량 이미지 처리 시 별도 스토리지 고려

## 📞 문제 발생 시

- 데이터베이스 파일 위치: `D:\Nong-View\nongview.db`
- 로그 확인: `test_db.py` 실행 시 콘솔 출력 확인
- 환경 변수 확인: `.env` 파일 설정 검토