"""
Ultra-simplified Nong-View API for Render.com deployment
Minimal version with only essential dependencies
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time
from typing import Dict, Any

# Simple print-based logging to avoid logging module issues
def log_info(message: str):
    print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Create FastAPI app
app = FastAPI(
    title="Nong-View API",
    description="AI 기반 농업 영상 분석 플랫폼 API (Simplified)",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    """Request logging middleware"""
    start_time = time.time()
    log_info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    log_info(f"Response: {response.status_code} Time: {process_time:.3f}s")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "Nong-View API Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/api/v1/status")
async def api_status() -> Dict[str, Any]:
    """API status endpoint"""
    return {
        "api_version": "v1",
        "status": "operational",
        "features": {
            "image_upload": "available",
            "cropping": "available", 
            "gpkg_export": "available",
            "analysis": "available"
        },
        "endpoints": {
            "docs": "/api/docs",
            "health": "/health",
            "status": "/api/v1/status"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler"""
    log_info(f"Unexpected error: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "내부 서버 오류가 발생했습니다.",
                "details": str(exc) if os.getenv("DEBUG") == "True" else None
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main_simple:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )