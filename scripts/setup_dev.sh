#!/bin/bash

# Development setup script for Local RAG System

set -e

echo "🚀 Setting up Local RAG System for development..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📋 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

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

# Check PostgreSQL
echo "🐘 Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL found"
    
    # Check if pgvector extension is available
    if psql -c "SELECT 1" template1 &> /dev/null 2>&1; then
        echo "✅ PostgreSQL is accessible"
        
        # Try to create database and user
        echo "📊 Setting up database..."
        psql -c "CREATE DATABASE rag_system;" template1 2>/dev/null || echo "Database may already exist"
        psql -c "CREATE USER rag_user WITH PASSWORD 'rag_password';" template1 2>/dev/null || echo "User may already exist"
        psql -c "GRANT ALL PRIVILEGES ON DATABASE rag_system TO rag_user;" template1 2>/dev/null || echo "Privileges may already be granted"
        
        # Initialize schema
        psql -U rag_user -d rag_system -f sql/init_database.sql 2>/dev/null || echo "Schema may already exist"
        echo "✅ Database setup completed"
    else
        echo "❌ Cannot connect to PostgreSQL. Please ensure it's running and accessible."
    fi
else
    echo "❌ PostgreSQL not found. Please install PostgreSQL and pgvector extension."
    echo "Ubuntu/Debian: sudo apt install postgresql postgresql-contrib postgresql-15-pgvector"
fi

echo ""
echo "🎉 Development setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Edit .env file and add your Claude API key"
echo "3. Start the API server: python -m src.api.main"
echo "4. Start the UI: streamlit run src/ui/streamlit_app.py"
echo ""
echo "Or use Docker: docker-compose up -d"