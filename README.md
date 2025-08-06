# Local RAG System

A fully local Retrieval-Augmented Generation (RAG) system that runs without external infrastructure dependencies. Uses PostgreSQL with pgvector for vector storage, local embeddings via sentence-transformers, and only calls external APIs for final LLM responses via Claude.

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Document      │───▶│  Embedding   │───▶│   PostgreSQL    │
│   Processing    │    │  Generation  │    │  (pgvector)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                       │
┌─────────────────┐    ┌──────────────┐              │
│   Chat UI       │◀──▶│   FastAPI    │◀─────────────┘
│  (Streamlit)    │    │   Backend    │
└─────────────────┘    └──────────────┘
                              │
                       ┌──────────────┐
                       │   Claude API │
                       │   Calls      │
                       └──────────────┘
```

## Features

- **Local Document Processing**: PDF text extraction and intelligent chunking
- **Local Embeddings**: sentence-transformers for cost-effective vector generation
- **PostgreSQL Vector Search**: Fast semantic search with pgvector extension
- **Chat Interface**: Streamlit-based UI with conversation history
- **Source Citations**: Shows document sources with page numbers
- **Cost Efficient**: Only external API calls are to Claude for final responses
- **Docker Support**: Easy deployment with docker-compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Claude API key from Anthropic

### 1. Clone and Setup

```bash
git clone <repository>
cd local-rag
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Claude API key
```

### 3. Start with Docker

```bash
docker-compose up -d
```

This will start:
- PostgreSQL with pgvector (port 5432)
- FastAPI backend (port 8000)
- Streamlit UI (port 8501)

### 4. Access the Application

- **Chat Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## Manual Setup (Development)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup PostgreSQL

Install PostgreSQL and the pgvector extension:

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE EXTENSION vector;"

# Run the initialization script
psql -U rag_user -d rag_system -f sql/init_database.sql
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database and API configuration
```

### 4. Start Services

```bash
# Terminal 1: Start FastAPI backend
python -m src.api.main

# Terminal 2: Start Streamlit UI
streamlit run src/ui/streamlit_app.py
```

## Usage

### 1. Upload Documents

- Use the Streamlit interface to upload PDF documents
- Documents are automatically processed and chunked
- Embeddings are generated locally and stored in PostgreSQL

### 2. Chat with Documents

- Ask questions about your uploaded documents
- The system retrieves relevant chunks and provides contextualized responses
- Source citations show which documents and pages were referenced

### 3. Search Documents

- Use the search tab to find specific information
- Results show similarity scores and source locations

## Configuration

Key configuration options in `.env`:

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_system
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password

# Claude API
CLAUDE_API_KEY=your_key_here
CLAUDE_MODEL=claude-3-sonnet-20240229

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast, good quality
# EMBEDDING_MODEL=all-mpnet-base-v2  # Higher quality, slower

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Embedding Model Options

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose |
| all-mpnet-base-v2 | 768 | Medium | Best | High quality needed |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | Fast | Good | Multilingual |

## API Endpoints

### Document Management
- `POST /documents/upload` - Upload PDF document
- `GET /documents` - List all documents
- `GET /documents/stats` - Get document statistics
- `DELETE /documents/{id}` - Delete document

### Search and Chat
- `POST /search` - Search document chunks
- `POST /chat` - Chat with RAG system
- `GET /chat/sessions` - List chat sessions
- `GET /chat/sessions/{id}/history` - Get chat history

### System
- `GET /health` - Health check

## Development

### Project Structure

```
local-rag/
├── src/
│   ├── api/              # FastAPI backend
│   ├── database/         # PostgreSQL operations
│   ├── document_processing/  # PDF processing and chunking
│   ├── embeddings/       # Local embedding generation
│   ├── ui/              # Streamlit interface
│   └── config/          # Configuration management
├── sql/                 # Database schema
├── docker/              # Docker configurations
├── tests/               # Test files
└── data/               # Document storage
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

## Troubleshooting

### Common Issues

1. **pgvector Extension Not Found**
   ```bash
   # Install pgvector extension
   sudo apt install postgresql-15-pgvector
   ```

2. **CUDA Out of Memory**
   ```bash
   # Use CPU for embeddings
   export DEVICE=cpu
   ```

3. **Database Connection Issues**
   ```bash
   # Check PostgreSQL service
   sudo systemctl status postgresql
   ```

### Performance Optimization

1. **For Large Document Collections**:
   - Increase `lists` parameter in vector index
   - Use more powerful embedding models
   - Consider database connection pooling

2. **For Faster Embeddings**:
   - Use GPU if available (`DEVICE=cuda`)
   - Increase batch size for processing
   - Consider smaller embedding models

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request