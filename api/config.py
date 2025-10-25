"""
Nong-View API Configuration
"""

import os
from typing import List
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    # Application settings
    PROJECT_NAME: str = "Nong-View API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./nongview.db"
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security settings
    SECRET_KEY: str = "change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    ALGORITHM: str = "HS256"
    
    # File storage settings
    UPLOAD_PATH: str = "/tmp/uploads"
    CROP_PATH: str = "/tmp/crops"
    EXPORT_PATH: str = "/tmp/exports"
    MAX_FILE_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB
    
    # CORS settings
    ALLOWED_HOSTS: str = "*"
    CORS_ORIGINS: str = "http://localhost:3000,https://localhost:3000"
    
    @validator('CORS_ORIGINS')
    def validate_cors_origins(cls, v: str) -> List[str]:
        return [origin.strip() for origin in v.split(',')]
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Render.com specific configurations
if os.getenv('RENDER'):
    # Override settings for Render.com deployment
    settings.ENVIRONMENT = "production"
    settings.DEBUG = False
    
    # Use Render's PORT environment variable
    if os.getenv('PORT'):
        settings.PORT = int(os.getenv('PORT'))
    
    # Database URL from Render
    if os.getenv('DATABASE_URL'):
        settings.DATABASE_URL = os.getenv('DATABASE_URL')
    
    # Redis URL from Render
    if os.getenv('REDIS_URL'):
        settings.REDIS_URL = os.getenv('REDIS_URL')
    
    # Secret key from Render
    if os.getenv('SECRET_KEY'):
        settings.SECRET_KEY = os.getenv('SECRET_KEY')
    
    # CORS origins for production
    if os.getenv('CORS_ORIGINS'):
        settings.CORS_ORIGINS = os.getenv('CORS_ORIGINS').split(',')
    else:
        settings.CORS_ORIGINS = ["https://nong-view.onrender.com"]