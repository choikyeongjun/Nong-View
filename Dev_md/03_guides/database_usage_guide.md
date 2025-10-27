# Nong-View 데이터베이스 사용 가이드

**문서 버전**: 1.0.0  
**최종 수정**: 2025-10-27  
**대상 독자**: 개발자, 시스템 관리자

---

## 목차
1. [환경 설정](#1-환경-설정)
2. [데이터베이스 초기화](#2-데이터베이스-초기화)
3. [기본 CRUD 작업](#3-기본-crud-작업)
4. [고급 쿼리 패턴](#4-고급-쿼리-패턴)
5. [마이그레이션 관리](#5-마이그레이션-관리)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. 환경 설정

### 1.1 필수 패키지 설치

```bash
# Anaconda Python 사용 시
C:\Users\ASUS\anaconda3\python.exe -m pip install -r requirements.txt

# 최소 설치 (DB만)
pip install sqlalchemy==2.0.23 alembic==1.12.1 python-dotenv==1.0.0
```

### 1.2 환경 변수 설정

`.env` 파일 생성:
```env
# 개발 환경 (SQLite)
DATABASE_URL=sqlite:///D:/Nong-View/nongview.db
DEBUG_MODE=True

# 운영 환경 (PostgreSQL)
# DATABASE_URL=postgresql://user:password@localhost:5432/nongview
# DEBUG_MODE=False
```

### 1.3 빠른 시작

```bash
# Windows 환경
D:\Nong-View\run_test.bat

# Python 직접 실행
python D:\Nong-View\test_db.py
```

---

## 2. 데이터베이스 초기화

### 2.1 테이블 생성

```python
from src.database.database import engine, Base
from src.database.models import *

# 모든 테이블 생성
Base.metadata.create_all(bind=engine)
```

### 2.2 초기 데이터 삽입

```python
from src.database.database import get_db
from src.database.models import Parcel

db = next(get_db())

# 필지 데이터 삽입
parcel = Parcel(
    pnu="3627010100100010000",
    address="전라북도 남원시 도통동 100-1",
    area=2500.0,
    land_use="농지",
    crop_type="벼"
)
db.add(parcel)
db.commit()
```

---

## 3. 기본 CRUD 작업

### 3.1 Create (생성)

```python
from src.database.models import Image
from datetime import datetime

# 이미지 레코드 생성
image = Image(
    filename="drone_ortho_001.tif",
    filepath="/data/images/drone_ortho_001.tif",
    width=5000,
    height=4000,
    bands=4,
    crs="EPSG:5186",
    bounds={
        "minx": 127.123,
        "miny": 35.456,
        "maxx": 127.789,
        "maxy": 35.890
    }
)

db.add(image)
db.commit()
db.refresh(image)  # ID 등 자동 생성 필드 갱신
print(f"Created image with ID: {image.id}")
```

### 3.2 Read (조회)

```python
# 단일 레코드 조회
image = db.query(Image).filter(Image.id == image_id).first()

# 조건부 조회
processing_images = db.query(Image).filter(
    Image.status == "processing"
).all()

# 페이징
page = 1
page_size = 10
images = db.query(Image).offset((page-1)*page_size).limit(page_size).all()
```

### 3.3 Update (수정)

```python
# 방법 1: 직접 수정
image = db.query(Image).filter(Image.id == image_id).first()
image.status = "completed"
image.metadata = {"processed": True, "quality": "high"}
db.commit()

# 방법 2: bulk update
db.query(Image).filter(Image.status == "uploaded").update(
    {"status": "queued"},
    synchronize_session=False
)
db.commit()
```

### 3.4 Delete (삭제)

```python
# 단일 삭제
image = db.query(Image).filter(Image.id == image_id).first()
db.delete(image)
db.commit()

# 조건부 삭제
db.query(Result).filter(Result.confidence < 0.3).delete()
db.commit()
```

---

## 4. 고급 쿼리 패턴

### 4.1 조인 쿼리

```python
from sqlalchemy.orm import joinedload

# Eager Loading으로 N+1 문제 해결
analyses_with_results = db.query(Analysis).options(
    joinedload(Analysis.results)
).filter(Analysis.status == "completed").all()

# 명시적 조인
query = db.query(Image, Analysis).join(
    Analysis, Image.id == Analysis.image_id
).filter(Analysis.model_name == "YOLOv11")
```

### 4.2 집계 쿼리

```python
from sqlalchemy import func

# 분석별 결과 수 집계
result_counts = db.query(
    Analysis.id,
    func.count(Result.id).label("result_count")
).join(Result).group_by(Analysis.id).all()

# 필지별 통계
stats = db.query(
    Parcel.pnu,
    func.avg(ParcelStatistics.crop_coverage_percent).label("avg_coverage"),
    func.sum(ParcelStatistics.facility_count).label("total_facilities")
).join(ParcelStatistics).group_by(Parcel.pnu).all()
```

### 4.3 JSON 필드 쿼리

```python
from sqlalchemy import cast, String

# JSON 필드 검색
high_altitude_images = db.query(Image).filter(
    cast(Image.metadata["altitude"], String) > "100"
).all()

# JSON 배열 처리
results_with_rice = db.query(Result).filter(
    Result.attributes["crop_type"].astext == "rice"
).all()
```

### 4.4 트랜잭션 관리

```python
from sqlalchemy import exc

try:
    # 트랜잭션 시작
    with db.begin():
        # 여러 작업 수행
        image = Image(...)
        db.add(image)
        
        analysis = Analysis(image_id=image.id, ...)
        db.add(analysis)
        
        # 자동 커밋 또는 롤백
except exc.IntegrityError as e:
    print(f"Integrity error: {e}")
    # 트랜잭션 자동 롤백
```

---

## 5. 마이그레이션 관리

### 5.1 Alembic 초기화

```bash
# 프로젝트 루트에서 실행
cd D:\Nong-View
alembic init alembic
```

### 5.2 마이그레이션 생성

```bash
# 자동 생성
alembic revision --autogenerate -m "Add new column to images"

# 수동 생성
alembic revision -m "Custom migration"
```

### 5.3 마이그레이션 적용

```bash
# 최신 버전으로 업그레이드
alembic upgrade head

# 특정 버전으로
alembic upgrade +1  # 한 단계 위로
alembic downgrade -1  # 한 단계 아래로

# 현재 버전 확인
alembic current
```

### 5.4 마이그레이션 스크립트 예시

```python
# alembic/versions/xxx_add_column.py
def upgrade():
    op.add_column('images', 
        sa.Column('processing_time', sa.Float(), nullable=True)
    )
    
def downgrade():
    op.drop_column('images', 'processing_time')
```

---

## 6. 트러블슈팅

### 6.1 일반적인 오류 해결

#### SQLite "database is locked" 오류
```python
# 해결책: 타임아웃 증가
engine = create_engine(
    "sqlite:///nongview.db",
    connect_args={"timeout": 30}
)
```

#### Foreign Key 제약 오류
```python
# SQLite에서 Foreign Key 활성화
from sqlalchemy import event

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

#### 대용량 JSON 처리
```python
# JSON 필드 크기 제한 설정
from sqlalchemy.dialects.postgresql import JSONB

class LargeResult(Base):
    __tablename__ = "large_results"
    data = Column(JSONB)  # PostgreSQL JSONB 타입 사용
```

### 6.2 성능 최적화 팁

#### 인덱스 추가
```python
from sqlalchemy import Index

# 모델에 인덱스 정의
class Image(Base):
    __tablename__ = "images"
    status = Column(String, index=True)  # 단일 컬럼 인덱스
    
    __table_args__ = (
        Index('ix_image_status_date', 'status', 'created_at'),  # 복합 인덱스
    )
```

#### 쿼리 최적화
```python
# Bad: N+1 문제
for analysis in db.query(Analysis).all():
    print(analysis.results)  # 각각 추가 쿼리 발생

# Good: Eager loading
analyses = db.query(Analysis).options(
    joinedload(Analysis.results)
).all()
```

#### 벌크 작업
```python
# Bad: 개별 삽입
for data in large_dataset:
    obj = Model(**data)
    db.add(obj)
    db.commit()

# Good: 벌크 삽입
db.bulk_insert_mappings(Model, large_dataset)
db.commit()
```

### 6.3 데이터베이스 모니터링

```python
# 쿼리 로깅 활성화
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# 실행 시간 측정
from sqlalchemy import event
import time

@event.listens_for(engine, "before_execute")
def receive_before_execute(conn, clauseelement, multiparams, params):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(engine, "after_execute")
def receive_after_execute(conn, clauseelement, multiparams, params, result):
    total_time = time.time() - conn.info['query_start_time'].pop()
    if total_time > 1.0:  # 1초 이상 걸린 쿼리 로깅
        logger.warning(f"Slow query ({total_time:.2f}s): {clauseelement}")
```

---

## 부록 A: 유용한 SQL 쿼리

### 데이터베이스 상태 확인
```sql
-- 테이블 크기 확인 (PostgreSQL)
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(tablename::regclass) DESC;

-- 활성 연결 확인
SELECT count(*) FROM pg_stat_activity;
```

### 데이터 정리
```sql
-- 오래된 분석 결과 삭제
DELETE FROM analyses 
WHERE completed_at < NOW() - INTERVAL '30 days' 
  AND status = 'completed';

-- 고아 레코드 정리
DELETE FROM results 
WHERE analysis_id NOT IN (SELECT id FROM analyses);
```

---

## 부록 B: 참고 자료

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [PostGIS Documentation](https://postgis.net/documentation/)

---

**지원 및 문의**
- 이슈 트래커: GitHub Issues
- 이메일: dev-team@nongview.com
- 내부 위키: /wiki/database