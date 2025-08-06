"""
Database manager for PostgreSQL with pgvector support.
Handles all database operations for the RAG system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
from dataclasses import asdict
from pathlib import Path

from ..document_processing import DocumentChunk

logger = logging.getLogger(__name__)

def adapt_numpy_array(numpy_array):
    """Adapter for numpy arrays to PostgreSQL vector format"""
    return AsIs(f"'{numpy_array.tolist()}'::vector")

# Register the adapter
register_adapter(np.ndarray, adapt_numpy_array)

class DatabaseManager:
    """Manages all database operations for the RAG system"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize database manager
        
        Args:
            connection_params: PostgreSQL connection parameters
        """
        self.connection_params = connection_params
        self.connection = None
        
    def connect(self) -> None:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.connection.autocommit = False
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
    
    def init_database(self, sql_file_path: Path) -> None:
        """
        Initialize database schema from SQL file
        
        Args:
            sql_file_path: Path to SQL initialization file
        """
        with open(sql_file_path, 'r') as f:
            sql_content = f.read()
        
        with self.connection.cursor() as cursor:
            cursor.execute(sql_content)
            self.connection.commit()
        
        logger.info("Database schema initialized")
    
    def document_exists(self, file_hash: str) -> bool:
        """Check if document with given hash already exists"""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM rag.documents WHERE file_hash = %s",
                (file_hash,)
            )
            return cursor.fetchone() is not None
    
    def insert_document(self, filename: str, filepath: str, file_hash: str, 
                       total_pages: int, file_size: int) -> int:
        """
        Insert new document record
        
        Returns:
            Document ID
        """
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO rag.documents (filename, filepath, file_hash, total_pages, file_size)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (filename, filepath, file_hash, total_pages, file_size))
            
            document_id = cursor.fetchone()[0]
            self.connection.commit()
            
            logger.info(f"Document inserted with ID: {document_id}")
            return document_id
    
    def update_document_status(self, document_id: int, status: str) -> None:
        """Update document processing status"""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "UPDATE rag.documents SET processing_status = %s WHERE id = %s",
                (status, document_id)
            )
            self.connection.commit()
    
    def insert_document_chunks_batch(self, document_id: int, chunks: List[DocumentChunk], 
                                   embeddings: List[np.ndarray]) -> None:
        """
        Insert document chunks with embeddings in batch
        
        Args:
            document_id: ID of the parent document
            chunks: List of document chunks
            embeddings: List of embedding vectors (same order as chunks)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings lists must have same length")
        
        # Prepare data for batch insert
        chunk_data = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_data.append((
                document_id,
                chunk.chunk_index,
                chunk.text,
                embedding,
                chunk.page_number,
                chunk.char_start,
                chunk.char_end,
                chunk.token_count
            ))
        
        with self.connection.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO rag.document_chunks 
                (document_id, chunk_index, chunk_text, embedding, page_number, char_start, char_end, token_count)
                VALUES %s
                """,
                chunk_data
            )
            self.connection.commit()
        
        logger.info(f"Inserted {len(chunks)} chunks for document {document_id}")
    
    def search_similar_chunks(self, query_embedding: np.ndarray, limit: int = 5, 
                            similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar chunks with metadata
        """
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    dc.id,
                    dc.chunk_text,
                    dc.page_number,
                    dc.chunk_index,
                    d.filename,
                    d.filepath,
                    1 - (dc.embedding <=> %s::vector) as similarity_score
                FROM rag.document_chunks dc
                JOIN rag.documents d ON dc.document_id = d.id
                WHERE d.processing_status = 'completed'
                    AND 1 - (dc.embedding <=> %s::vector) >= %s
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, similarity_threshold, query_embedding, limit))
            
            results = cursor.fetchall()
            
        logger.info(f"Found {len(results)} similar chunks")
        return [dict(row) for row in results]
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    SUM(total_pages) as total_pages,
                    SUM(file_size) as total_file_size,
                    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_documents,
                    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_documents
                FROM rag.documents
            """)
            doc_stats = dict(cursor.fetchone())
            
            cursor.execute("SELECT COUNT(*) as total_chunks FROM rag.document_chunks")
            chunk_stats = dict(cursor.fetchone())
            
        return {**doc_stats, **chunk_stats}
    
    def create_chat_session(self, session_name: Optional[str] = None) -> int:
        """Create new chat session"""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO rag.chat_sessions (session_name) VALUES (%s) RETURNING id",
                (session_name,)
            )
            session_id = cursor.fetchone()[0]
            self.connection.commit()
            
        logger.info(f"Created chat session {session_id}")
        return session_id
    
    def add_chat_message(self, session_id: int, role: str, content: str, 
                        retrieved_chunks: Optional[List[int]] = None,
                        claude_tokens_used: Optional[int] = None) -> int:
        """Add message to chat session"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO rag.chat_messages 
                (session_id, role, content, retrieved_chunks, claude_tokens_used)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (session_id, role, content, retrieved_chunks, claude_tokens_used))
            
            message_id = cursor.fetchone()[0]
            self.connection.commit()
            
        return message_id
    
    def get_chat_history(self, session_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT role, content, timestamp, retrieved_chunks, claude_tokens_used
                FROM rag.chat_messages 
                WHERE session_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (session_id, limit))
            
            results = cursor.fetchall()
            
        return [dict(row) for row in reversed(results)]  # Return in chronological order
    
    def get_chat_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of chat sessions"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, session_name, created_at, updated_at,
                       (SELECT COUNT(*) FROM rag.chat_messages WHERE session_id = cs.id) as message_count
                FROM rag.chat_sessions cs
                ORDER BY updated_at DESC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            
        return [dict(row) for row in results]
    
    def delete_document(self, document_id: int) -> None:
        """Delete document and all associated chunks"""
        with self.connection.cursor() as cursor:
            cursor.execute("DELETE FROM rag.documents WHERE id = %s", (document_id,))
            self.connection.commit()
            
        logger.info(f"Deleted document {document_id}")
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all documents"""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, filename, filepath, total_pages, file_size, 
                       upload_date, processing_status,
                       (SELECT COUNT(*) FROM rag.document_chunks WHERE document_id = d.id) as chunk_count
                FROM rag.documents d
                ORDER BY upload_date DESC
            """)
            
            results = cursor.fetchall()
            
        return [dict(row) for row in results]