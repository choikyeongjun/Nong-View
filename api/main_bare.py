"""
Bare minimum API using only Starlette (no FastAPI/pydantic)
For Render.com deployment when everything else fails
"""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import os
import time

async def root(request):
    """Root endpoint"""
    return JSONResponse({
        "message": "Nong-View API Server (Bare Mode)",
        "version": "1.0.0",
        "status": "running",
        "mode": "bare_minimum",
        "port": os.getenv("PORT", "8000")
    })

async def health(request):
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "mode": "bare_minimum"
    })

async def test(request):
    """Test endpoint"""
    return JSONResponse({
        "test": "success",
        "message": "Bare minimum API is working!",
        "environment": os.getenv("ENVIRONMENT", "unknown")
    })

# Create Starlette app
app = Starlette(
    routes=[
        Route("/", root),
        Route("/health", health),
        Route("/test", test),
    ]
)

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main_bare:app",
        host=host,
        port=port,
        reload=False
    )