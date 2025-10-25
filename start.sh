#!/bin/bash
# Render.com Start Script for Nong-View

echo "🚀 Starting Nong-View API server..."

# Set environment variables
export GDAL_CONFIG=/usr/bin/gdal-config
export GDAL_DATA=/usr/share/gdal
export PROJ_LIB=/usr/share/proj

# Create data directories if they don't exist
mkdir -p /tmp/uploads
mkdir -p /tmp/crops
mkdir -p /tmp/exports

# Start the FastAPI server
cd api

# Try simplified version first for initial deployment
if [ -f "main_simple.py" ]; then
    echo "Starting with simplified API..."
    uvicorn main_simple:app --host 0.0.0.0 --port $PORT --workers 1
else
    echo "Starting with full API..."
    uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
fi