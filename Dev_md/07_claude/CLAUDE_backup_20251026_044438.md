# Nong-View AI 영상분석 파이프라인 개발 계획

## 프로젝트 개요
- **목적**: 드론 정사영상 기반 농업 모니터링 및 행정 자동화 시스템
- **핵심기능**: 조사료/시설물/농지이용 상태 AI 자동 분석
- **최종산출물**: 행정보고용 GPKG 및 통계 리포트
- **진행상황**: Opus 구현 완료 (50%), Sonnet 구현 대기 (50%)

---

## 📋 TODO LIST - Claude Opus

### ✅ 완료된 작업 (2025-10-26)

#### POD 1: 데이터 관리
- [x] 데이터 레지스트리 아키텍처 설계 및 구현
- [x] 메타데이터 스키마 정의 (ImageMetadata, ShapeMetadata)
- [x] 버전 관리 시스템 구현 (시계열 스냅샷)
- [x] 좌표계 검증 엔진 구현 (CoordinateValidator)
- [x] 공간 데이터 검증 구현 (GeometryValidator)

#### POD 3: 타일링
- [x] 640x640 타일 생성 엔진 구현 (TilingEngine)
- [x] 겹침(Overlap) 전략 알고리즘 구현 (20% 기본값)
- [x] 타일 인덱싱 시스템 구현 (R-tree 기반)
- [x] 병렬 처리 시스템 구현 (ThreadPoolExecutor)

#### POD 4: AI 분석
- [x] YOLOv11 추론 엔진 구현 (InferenceEngine)
- [x] 멀티모델 관리 시스템 구현 (ModelManager)
  - [x] 조사료/사료작물 분류 모델 지원
  - [x] 비닐하우스 탐지 모델 지원
  - [x] 경작/휴경 판별 모델 지원
- [x] 모델 버전 관리 시스템 구현
- [x] A/B 테스트 및 롤백 기능 구현
- [x] GPU/CPU 자동 선택 및 최적화

#### POD 5: 병합
- [x] 타일 결과 병합 알고리즘 구현 (MergeEngine)
- [x] IOU 기반 중복 제거 로직 구현
- [x] 4가지 병합 전략 구현 (weighted_avg, max_confidence, union, intersection)
- [x] 필지별 통계 산출 기능 구현

#### 테스트
- [x] 각 POD별 단위 테스트 작성
- [x] pytest 기반 테스트 구조 구축

### 🔄 향후 작업 (지원 역할)
- [ ] 성능 최적화 컨설팅
- [ ] 복잡한 알고리즘 문제 해결 지원
- [ ] 시스템 아키텍처 리뷰
- [ ] 코드 리뷰 및 개선 제안

---

## 📋 TODO LIST - Claude Sonnet

### 🚀 즉시 시작 필요 (Priority 1)

#### POD 2: 크로핑 (ROI 추출) - 신규 개발
- [ ] 크로핑 엔진 구현 (`src/pod2_cropping/`)
  - [ ] ROI 추출 핵심 로직
  - [ ] Convex Hull 계산 함수
  - [ ] GDAL/Rasterio 클리핑 구현
  - [ ] 최소 분석 경계 계산
- [ ] 크로핑 스키마 정의
  - [ ] CropConfig 스키마
  - [ ] ROIBounds 스키마
  - [ ] CropResult 스키마
- [ ] 크로핑 API 엔드포인트
  - [ ] POST `/api/v1/images/{id}/crop`
  - [ ] GET `/api/v1/crops/{crop_id}`
- [ ] 테스트 코드 작성

#### POD 6: GPKG 발행 - 신규 개발
- [ ] GPKG Export 엔진 (`src/pod6_gpkg_export/`)
  - [ ] GeoPackage 생성 로직
  - [ ] 레이어 구조 정의 (parcels, crops, facilities, statistics)
  - [ ] 좌표계/필드명 표준화
- [ ] 민감정보 처리
  - [ ] 개인정보 마스킹 함수
  - [ ] 필드 제거/익명화 로직
- [ ] Export API 구현
  - [ ] POST `/api/v1/export/gpkg`
  - [ ] GET `/api/v1/export/{export_id}/status`
  - [ ] GET `/api/v1/export/{export_id}/download`
- [ ] 리포트 템플릿 생성기

### 📡 API 개발 (Priority 2)

#### FastAPI 서버 구축
- [ ] 프로젝트 구조 설정 (`api/`)
  - [ ] `main.py` - 메인 애플리케이션
  - [ ] `routers/` - 라우터 모듈
  - [ ] `schemas/` - Pydantic 스키마
  - [ ] `dependencies.py` - 의존성 주입

#### 핵심 API 엔드포인트
- [ ] 이미지 관리 API
  - [ ] POST `/api/v1/images` - 업로드
  - [ ] GET `/api/v1/images` - 목록
  - [ ] GET `/api/v1/images/{id}` - 상세
  - [ ] DELETE `/api/v1/images/{id}` - 삭제

- [ ] 분석 API
  - [ ] POST `/api/v1/analyses` - 분석 시작
  - [ ] GET `/api/v1/analyses/{job_id}` - 상태 조회
  - [ ] GET `/api/v1/analyses/{job_id}/results` - 결과

- [ ] 타일 API
  - [ ] POST `/api/v1/images/{id}/tiles` - 타일 생성
  - [ ] GET `/api/v1/tiles/{tile_id}` - 타일 조회

- [ ] 통계 API
  - [ ] GET `/api/v1/parcels/{pnu}/statistics`
  - [ ] GET `/api/v1/statistics/regional`

### 🗄️ 데이터베이스 (Priority 2)

#### DB 설정 및 모델
- [ ] SQLAlchemy 모델 정의
  - [ ] Image 모델
  - [ ] Analysis 모델
  - [ ] Result 모델
  - [ ] Parcel 모델
- [ ] Alembic 마이그레이션 설정
- [ ] DB 연결 풀 구성
- [ ] 트랜잭션 관리

### 🎨 UI/UX 개발 (Priority 3)

#### 관리자 대시보드
- [ ] 프론트엔드 프레임워크 선택 (React/Vue)
- [ ] 대시보드 레이아웃
  - [ ] 사이드바 네비게이션
  - [ ] 메인 컨텐츠 영역
  - [ ] 상태바
- [ ] 핵심 페이지
  - [ ] 이미지 관리 페이지
  - [ ] 분석 모니터링 페이지
  - [ ] 결과 조회 페이지
  - [ ] 통계 대시보드

#### 시각화 컴포넌트
- [ ] 지도 뷰어 (Leaflet/OpenLayers)
- [ ] 진행률 표시기
- [ ] 차트 컴포넌트 (Chart.js)
- [ ] 테이블 컴포넌트

### 🚀 배포 및 인프라 (Priority 4)

#### Docker 구성
- [ ] Dockerfile 작성
  - [ ] API 서버 이미지
  - [ ] 워커 이미지
  - [ ] 프론트엔드 이미지
- [ ] docker-compose.yml 작성
- [ ] 환경변수 설정

#### CI/CD
- [ ] GitHub Actions 워크플로우
- [ ] 자동 테스트
- [ ] 자동 배포

### 📝 문서화 (Ongoing)

- [ ] API 문서 자동 생성 (Swagger)
- [ ] 사용자 가이드 작성
- [ ] 설치 가이드 업데이트
- [ ] 트러블슈팅 가이드

---

## 🎯 이번 주 목표 (2025-10-27 ~ 11-02)

### Sonnet 우선순위
1. **10-27 (월)**: POD2 크로핑 엔진 구현
2. **10-28 (화)**: POD6 GPKG Export 구현
3. **10-29 (수)**: FastAPI 서버 기본 구조
4. **10-30 (목)**: 이미지/분석 API 구현
5. **10-31 (금)**: 통계/Export API 구현
6. **11-01 (토)**: 테스트 및 디버깅
7. **11-02 (일)**: 문서화 및 리뷰

### Opus 지원
- 코드 리뷰 및 피드백
- 복잡한 문제 해결 지원
- 성능 최적화 조언

---

## 📊 진행 상황 추적

### 전체 진행률
```
POD1 (데이터): ████████████████████ 100% (Opus ✅)
POD2 (크로핑): ░░░░░░░░░░░░░░░░░░░░ 0%   (Sonnet 🔄)
POD3 (타일링): ████████████████████ 100% (Opus ✅)
POD4 (AI분석): ████████████████████ 100% (Opus ✅)
POD5 (병합):   ████████████████████ 100% (Opus ✅)
POD6 (GPKG):   ░░░░░░░░░░░░░░░░░░░░ 0%   (Sonnet 🔄)
API:           ░░░░░░░░░░░░░░░░░░░░ 0%   (Sonnet 🔄)
UI:            ░░░░░░░░░░░░░░░░░░░░ 0%   (Sonnet 🔄)

전체: ████████████░░░░░░░░ 50%
```

### 주요 마일스톤
- ✅ 2025-10-26: Opus 담당 모듈 완료
- 🎯 2025-11-02: Sonnet API 개발 완료 (목표)
- 🎯 2025-11-09: MVP 출시 준비 완료 (목표)

---

## 🤝 협업 규칙

### 코드 컨벤션
- Python: PEP 8 준수
- API: RESTful 원칙
- 문서: Markdown 형식
- 커밋: Conventional Commits

### 브랜치 전략
```
main
├── develop
│   ├── feature/pod2-cropping (Sonnet)
│   ├── feature/pod6-gpkg (Sonnet)
│   └── feature/api-server (Sonnet)
└── release/v1.0.0
```

### 커뮤니케이션
- 일일 진행 상황 업데이트
- 블로킹 이슈 즉시 공유
- API 변경 사전 협의
- 코드 리뷰 상호 진행

---

*Last Updated: 2025-10-26 04:45*
*Next Review: 2025-10-27 09:00*
*Status: Opus Complete, Sonnet Starting*