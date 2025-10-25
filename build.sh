#!/bin/bash
# Render.com Build Script for Nong-View

echo "🚀 Starting Nong-View build process..."

# Update package list
echo "📦 Updating package list..."
apt-get update

# Install essential build tools
echo "🔧 Installing build tools..."
apt-get install -y \
    build-essential \
    g++ \
    gcc \
    libc6-dev \
    pkg-config

# Install GDAL and dependencies (minimal set)
echo "🗺️ Installing minimal GDAL dependencies..."
apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin

# Set environment variables
export GDAL_CONFIG=/usr/bin/gdal-config
export GDAL_DATA=/usr/share/gdal
export PROJ_LIB=/usr/share/proj

# Setup Rust with writable cache directory
echo "🦀 Setting up Rust environment..."
export CARGO_HOME=/tmp/cargo
export RUSTUP_HOME=/tmp/rustup
export PATH=/tmp/cargo/bin:$PATH
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
source /tmp/cargo/env

# Upgrade pip and install wheel
echo "🐍 Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install ultra-minimal Python dependencies
echo "📚 Installing Python dependencies (Ultra-minimal)..."
echo "✅ Using Render-optimized requirements.txt"

# Try ultra-minimal first, fallback to emergency if needed
if ! pip install -r requirements.txt --prefer-binary --no-cache-dir --only-binary=all; then
    echo "⚠️ Standard install failed, trying emergency minimal..."
    pip install -r requirements-emergency.txt --prefer-binary --no-cache-dir --only-binary=all
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p /tmp/uploads
mkdir -p /tmp/crops  
mkdir -p /tmp/exports

echo "✅ Build completed successfully!"