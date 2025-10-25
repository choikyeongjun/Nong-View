#!/bin/bash
# Render.com Build Script for Nong-View

echo "🚀 Starting Nong-View build process..."

# Update package list
echo "📦 Updating package list..."
apt-get update

# Install Rust for packages that need compilation
echo "🦀 Installing Rust toolchain..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

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
# Try Render-optimized requirements first, fallback to minimal if needed
if [ -f "requirements-render.txt" ]; then
    echo "Using Render-optimized requirements..."
    pip install -r requirements-render.txt
elif [ -f "requirements-minimal.txt" ]; then
    echo "Using minimal requirements..."
    pip install -r requirements-minimal.txt
else
    echo "Using standard requirements..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p /tmp/uploads
mkdir -p /tmp/crops  
mkdir -p /tmp/exports

echo "✅ Build completed successfully!"