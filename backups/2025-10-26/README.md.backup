# Nong-View (농뷰) - AI 기반 농업 영상 분석 플랫폼

## 📋 Overview
Nong-View는 드론으로 촬영한 정사영상을 AI로 분석하여 농업 모니터링 및 행정 보고를 자동화하는 시스템입니다.

### 주요 목적
- 🚁 드론 정사영상 기반 농지 자동 분석
- 🌾 조사료/사료작물 생육 상태 모니터링
- 🏠 농업 시설물(비닐하우스 등) 자동 탐지
- 📊 행정 보고용 공간정보 자동 생성

## ✨ Features

### Core Features
- **POD 1**: 영상/공간데이터 수집 및 관리
- **POD 2**: ROI(관심영역) 추출
- **POD 3**: 640x640 타일 자동 생성
- **POD 4**: YOLOv11 기반 AI 분석
- **POD 5**: 분석 결과 병합 및 통계
- **POD 6**: GPKG 포맷 자동 발행

### AI Models
1. 🌱 조사료/사료작물 분류 및 면적 추정
2. 🏗️ 비닐하우스(단동/연동) 시설물 탐지  
3. 🚜 경작/휴경 상태 자동 판별

## 🛠️ Tech Stack

### Backend
- Python 3.10+
- FastAPI
- PostgreSQL + PostGIS
- Celery + Redis

### AI/ML
- PyTorch
- YOLOv11
- CUDA 11.8+

### GIS
- GDAL / Rasterio
- GeoPandas / Shapely
- Fiona

### Infrastructure
- Docker / Kubernetes
- MinIO (S3 compatible)
- Prometheus + Grafana

## 📦 Installation

### Prerequisites
```bash
# Python 3.10+
python --version

# GDAL
gdal-config --version

# PostgreSQL with PostGIS
psql --version

# Redis
redis-server --version
```

### Setup
```bash
# Clone repository
git clone https://github.com/aebonlee/Nong-View.git
cd Nong-View

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/init_db.py

# Run migrations
alembic upgrade head
```

## 🚀 Quick Start

### 1. Start Services
```bash
# Start Redis
redis-server

# Start Celery worker
celery -A src.worker worker --loglevel=info

# Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Upload Image
```bash
curl -X POST "http://localhost:8000/api/v1/images/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_ortho.tif"
```

### 3. Start Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/start" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "uuid-here",
    "roi_pnu": "4513010100100010000",
    "models": ["crop", "facility", "landuse"]
  }'
```

### 4. Export Results
```bash
curl -X GET "http://localhost:8000/api/v1/export/gpkg/{result_id}" \
  -o result.gpkg
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs)
- [Architecture Guide](./Dev_md/06_architecture/system_architecture.md)
- [Development Guide](./Dev_md/03_guides/pod_development_guide.md)
- [Development Rules](./Dev_md/02_rules/development_rules.md)

## 🗂️ Project Structure
```
Nong-View/
├── src/              # Source code
│   ├── pod1_data_ingestion/
│   ├── pod2_cropping/
│   ├── pod3_tiling/
│   ├── pod4_ai_inference/
│   ├── pod5_merging/
│   └── pod6_gpkg_export/
├── api/              # REST API
├── models/           # AI models
├── tests/            # Test suites
├── scripts/          # Utility scripts
├── Dev_md/           # Development documentation
└── docker/           # Docker configurations
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/unit/test_tiling.py

# Run integration tests
pytest tests/integration/
```

## 🐳 Docker

```bash
# Build image
docker build -t nong-view:latest .

# Run container
docker run -p 8000:8000 nong-view:latest

# Docker Compose
docker-compose up -d
```

## 📈 Performance

### Benchmarks
- 📸 Image Upload: 100MB/s
- 🔲 Tile Generation: 1000 tiles/min
- 🤖 AI Inference: 100 tiles/min (per GPU)
- 📦 GPKG Export: < 30s

### Requirements
- RAM: 16GB minimum
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: 500GB+ SSD recommended

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Commit Convention
```
feat: New feature
fix: Bug fix
docs: Documentation
style: Code style
refactor: Code refactoring
test: Test code
chore: Build/package
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Architecture & AI**: Claude Opus
- **API & Integration**: Claude Sonnet

## 📞 Support

- GitHub Issues: [https://github.com/aebonlee/Nong-View/issues](https://github.com/aebonlee/Nong-View/issues)
- Email: support@nong-view.kr

## 🙏 Acknowledgments

- 남원시 스마트빌리지 사업단
- 한국농촌경제연구원
- 농림축산식품부

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-26  
**Status**: 🚧 Under Development