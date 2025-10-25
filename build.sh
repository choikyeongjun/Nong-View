#!/bin/bash
# Render.com Build Script for Nong-View

echo "🚀 Starting Nong-View build process..."

# Update package list
echo "📦 Updating package list..."
apt-get update

# Install GDAL and dependencies
echo "🗺️ Installing GDAL and geospatial dependencies..."
apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    gcc \
    libc6-dev \
    pkg-config \
    libproj-dev \
    proj-data \
    proj-bin

# Set GDAL environment variables
export GDAL_CONFIG=/usr/bin/gdal-config
export GDAL_DATA=/usr/share/gdal
export PROJ_LIB=/usr/share/proj

# Upgrade pip
echo "🐍 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p /tmp/uploads
mkdir -p /tmp/crops  
mkdir -p /tmp/exports

echo "✅ Build completed successfully!"