"""
FastAPI backend for the RAG system.
Provides endpoints for document processing, search, and chat functionality.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from ..database import DatabaseManager
from ..embeddings import EmbeddingGenerator
from ..document_processing import PDFProcessor
from .claude_client import ClaudeClient
from ..config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (will be initialized in lifespan)
db_manager: Optional[DatabaseManager] = None
embedding_generator: Optional[EmbeddingGenerator] = None
pdf_processor: Optional[PDFProcessor] = None
claude_client: Optional[ClaudeClient] = None
settings: Optional[Settings] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_manager, embedding_generator, pdf_processor, claude_client, settings
    
    # Initialize components
    settings = Settings()
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_name=settings.embedding_model,
        device=settings.device
    )
    embedding_generator.warm_up()
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Initialize Claude client
    claude_client = ClaudeClient(
        api_key=settings.claude_api_key,
        model=settings.claude_model
    )
    
    # Initialize database manager
    db_manager = DatabaseManager(settings.database_params)
    
    logger.info("FastAPI application started")
    
    yield
    
    # Cleanup
    if db_manager:
        db_manager.disconnect()
    
    logger.info("FastAPI application shut down")

# Create FastAPI app
app = FastAPI(
    title="Local RAG System",
    description="RAG system with local embeddings and PostgreSQL vector search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    max_chunks: int = 5
    similarity_threshold: float = 0.5

class ChatResponse(BaseModel):
    response: str
    session_id: int
    retrieved_chunks: List[Dict[str, Any]]
    tokens_used: Optional[int] = None

class DocumentStatus(BaseModel):
    id: int
    filename: str
    status: str
    chunk_count: int
    upload_date: str

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    similarity_threshold: float = 0.5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_embedding_dim: int

# Dependency to get database manager
def get_db():
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db_manager

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "embedding_model": settings.embedding_model}

# Document management endpoints
@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: DatabaseManager = Depends(get_db)
):
    """Upload and process a PDF document"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    upload_dir = Path("data/documents")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Start background processing
    background_tasks.add_task(
        process_document_background,
        str(file_path),
        file.filename,
        len(content)
    )
    
    return {"message": f"Document {file.filename} uploaded and processing started"}

async def process_document_background(file_path: str, filename: str, file_size: int):
    """Background task for document processing"""
    try:
        with db_manager:
            # Process document
            file_hash, total_pages, chunks = pdf_processor.process_document(Path(file_path))
            
            # Check if document already exists
            if db_manager.document_exists(file_hash):
                logger.info(f"Document {filename} already exists in database")
                return
            
            # Insert document record
            doc_id = db_manager.insert_document(
                filename, file_path, file_hash, total_pages, file_size
            )
            
            # Update status to processing
            db_manager.update_document_status(doc_id, "processing")
            
            # Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedding_generator.generate_embeddings_batch(
                chunk_texts, show_progress=False
            )
            
            # Insert chunks with embeddings
            db_manager.insert_document_chunks_batch(doc_id, chunks, embeddings)
            
            # Mark as completed
            db_manager.update_document_status(doc_id, "completed")
            
            logger.info(f"Document {filename} processing completed")
            
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        if 'doc_id' in locals():
            with db_manager:
                db_manager.update_document_status(doc_id, "failed")

@app.get("/documents", response_model=List[DocumentStatus])
async def list_documents(db: DatabaseManager = Depends(get_db)):
    """Get list of all documents"""
    with db:
        documents = db.get_document_list()
        return [
            DocumentStatus(
                id=doc['id'],
                filename=doc['filename'],
                status=doc['processing_status'],
                chunk_count=doc['chunk_count'],
                upload_date=doc['upload_date'].isoformat()
            )
            for doc in documents
        ]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: DatabaseManager = Depends(get_db)):
    """Delete a document and all its chunks"""
    with db:
        db.delete_document(document_id)
    return {"message": f"Document {document_id} deleted"}

@app.get("/documents/stats")
async def get_document_stats(db: DatabaseManager = Depends(get_db)):
    """Get document and chunk statistics"""
    with db:
        return db.get_document_stats()

# Search endpoints
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest, db: DatabaseManager = Depends(get_db)):
    """Search for similar document chunks"""
    # Generate query embedding
    query_embedding = embedding_generator.generate_query_embedding(request.query)
    
    # Search similar chunks
    with db:
        results = db.search_similar_chunks(
            query_embedding,
            limit=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
    
    return SearchResponse(
        results=results,
        query_embedding_dim=len(query_embedding)
    )

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: DatabaseManager = Depends(get_db)):
    """Chat with the RAG system"""
    # Generate query embedding
    query_embedding = embedding_generator.generate_query_embedding(request.message)
    
    # Search for relevant chunks
    with db:
        relevant_chunks = db.search_similar_chunks(
            query_embedding,
            limit=request.max_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        # Create or get session
        if request.session_id is None:
            session_id = db.create_chat_session()
        else:
            session_id = request.session_id
        
        # Get chat history for context
        chat_history = db.get_chat_history(session_id, limit=10)
        
        # Add user message to history
        db.add_chat_message(session_id, "user", request.message)
    
    # Generate response using Claude
    context = "\n\n".join([
        f"[Source: {chunk['filename']}, Page {chunk['page_number']}]\n{chunk['chunk_text']}"
        for chunk in relevant_chunks
    ])
    
    response, tokens_used = await claude_client.generate_response(
        request.message, context, chat_history
    )
    
    # Save assistant response
    with db:
        db.add_chat_message(
            session_id, 
            "assistant", 
            response, 
            retrieved_chunks=[chunk['id'] for chunk in relevant_chunks],
            claude_tokens_used=tokens_used
        )
    
    return ChatResponse(
        response=response,
        session_id=session_id,
        retrieved_chunks=relevant_chunks,
        tokens_used=tokens_used
    )

@app.get("/chat/sessions")
async def get_chat_sessions(db: DatabaseManager = Depends(get_db)):
    """Get list of chat sessions"""
    with db:
        return db.get_chat_sessions()

@app.get("/chat/sessions/{session_id}/history")
async def get_chat_history(session_id: int, db: DatabaseManager = Depends(get_db)):
    """Get chat history for a session"""
    with db:
        return db.get_chat_history(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)