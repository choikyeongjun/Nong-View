"""
Emergency minimal Nong-View API for Render.com deployment
Only FastAPI and uvicorn - no other dependencies
"""

from fastapi import FastAPI
import os
import time

# Create FastAPI app
app = FastAPI(
    title="Nong-View API (Emergency Mode)",
    description="Ultra-minimal deployment test",
    version="1.0.0",
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Nong-View API Server (Emergency Mode)",
        "version": "1.0.0",
        "status": "running",
        "mode": "emergency",
        "port": os.getenv("PORT", "8000")
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "mode": "emergency"
    }

@app.get("/test")
async def test():
    """Test endpoint"""
    return {
        "test": "success",
        "message": "Emergency API is working!",
        "environment": os.getenv("ENVIRONMENT", "unknown")
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main_emergency:app",
        host=host,
        port=port,
        reload=False
    )