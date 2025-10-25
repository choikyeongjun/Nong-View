"""
Nong-View API 서버 메인 애플리케이션

농업 영상 분석 파이프라인의 REST API 서버
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import os
import uvicorn

from v1.endpoints import images, analyses, crops, exports, statistics
from v1.dependencies import get_db
from config import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI 기반 농업 영상 분석 플랫폼 API",
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/v1/openapi.json",
    debug=settings.DEBUG
)

# 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅 미들웨어"""
    start_time = time.time()
    
    # 요청 로깅
    logger.info(f"요청 시작: {request.method} {request.url}")
    
    # 요청 처리
    response = await call_next(request)
    
    # 응답 로깅
    process_time = time.time() - start_time
    logger.info(
        f"요청 완료: {request.method} {request.url} "
        f"상태: {response.status_code} 시간: {process_time:.3f}s"
    )
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


# 라우터 등록
app.include_router(
    images.router,
    prefix="/api/v1/images",
    tags=["images"],
    dependencies=[]
)

app.include_router(
    analyses.router,
    prefix="/api/v1/analyses",
    tags=["analyses"],
    dependencies=[]
)

app.include_router(
    crops.router,
    prefix="/api/v1/crops",
    tags=["crops"],
    dependencies=[]
)

app.include_router(
    exports.router,
    prefix="/api/v1/exports",
    tags=["exports"],
    dependencies=[]
)

app.include_router(
    statistics.router,
    prefix="/api/v1/statistics",
    tags=["statistics"],
    dependencies=[]
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Nong-View API Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs"
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "api_docs": "/api/docs"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리기"""
    logger.error(f"예상치 못한 오류: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "내부 서버 오류가 발생했습니다.",
                "details": str(exc) if app.debug else None
            }
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )