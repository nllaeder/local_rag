-- Initialize database with pgvector extension for semantic search
-- Run this script as a PostgreSQL superuser

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database schema for RAG system
CREATE SCHEMA IF NOT EXISTS rag;

-- Document table to store original document metadata
CREATE TABLE IF NOT EXISTS rag.documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath TEXT NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_pages INTEGER,
    file_size BIGINT,
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed'))
);

-- Document chunks table for storing processed text chunks with embeddings
CREATE TABLE IF NOT EXISTS rag.document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES rag.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384), -- sentence-transformers/all-MiniLM-L6-v2 produces 384-dimensional embeddings
    page_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Chat sessions table for maintaining conversation history
CREATE TABLE IF NOT EXISTS rag.chat_sessions (
    id SERIAL PRIMARY KEY,
    session_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages table for storing conversation history
CREATE TABLE IF NOT EXISTS rag.chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES rag.chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retrieved_chunks INTEGER[], -- Array of chunk IDs used for this response
    claude_tokens_used INTEGER
);

-- Create indexes for efficient vector similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON rag.document_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create standard indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_documents_hash ON rag.documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON rag.documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON rag.document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON rag.chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON rag.chat_sessions(updated_at DESC);

-- Function to update chat session timestamp
CREATE OR REPLACE FUNCTION rag.update_chat_session_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE rag.chat_sessions 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update session timestamp when messages are added
CREATE TRIGGER update_session_timestamp
    AFTER INSERT ON rag.chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION rag.update_chat_session_timestamp();

-- Grant permissions (adjust as needed for your setup)
GRANT USAGE ON SCHEMA rag TO PUBLIC;
GRANT ALL ON ALL TABLES IN SCHEMA rag TO PUBLIC;
GRANT ALL ON ALL SEQUENCES IN SCHEMA rag TO PUBLIC;