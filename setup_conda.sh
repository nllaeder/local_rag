#!/bin/bash

# Conda-based development setup script for Local RAG System

set -e

echo "🚀 Setting up Local RAG System with Conda..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment..."
if conda info --envs | grep -q "local-rag"; then
    echo "Environment 'local-rag' already exists. Updating..."
    conda activate local-rag
else
    conda create -n local-rag python=3.11 -y
    conda activate local-rag
fi

# Install core packages via conda-forge
echo "📋 Installing core dependencies via conda..."
conda install -c conda-forge \
    fastapi \
    uvicorn \
    streamlit \
    psycopg2 \
    numpy \
    requests \
    python-dotenv \
    pydantic \
    tqdm \
    pytest -y

# Install PyTorch via conda (much more reliable than pip)
echo "🔥 Installing PyTorch..."
conda install pytorch cpuonly -c pytorch -y

# Install remaining packages via pip
echo "🐍 Installing additional packages via pip..."
pip install \
    sentence-transformers \
    transformers \
    pgvector \
    PyPDF2 \
    PyMuPDF \
    anthropic \
    python-multipart \
    pytest-asyncio \
    black \
    flake8

# Setup environment file
echo "⚙️ Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "❗ Please edit .env and add your Claude API key"
else
    echo "✅ .env file already exists"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/documents data/processed

echo ""
echo "🎉 Conda environment setup complete!"
echo ""
echo "To activate the environment:"
echo "conda activate local-rag"
echo ""
echo "Next steps:"
echo "1. Install PostgreSQL with pgvector (if not done already)"
echo "2. Edit .env file and add your Claude API key"
echo "3. Initialize database: psql -h localhost -U rag_user -d rag_system -f sql/init_database.sql"
echo "4. Test setup: python test_setup.py"
echo "5. Start API: python -m src.api.main"
echo "6. Start UI: streamlit run src/ui/streamlit_app.py"
