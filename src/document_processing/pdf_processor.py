"""
PDF document processing module for RAG system.
Handles PDF text extraction, intelligent chunking, and preparation for embedding.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import PyPDF2
import fitz  # PyMuPDF - better text extraction
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a processed document chunk"""
    chunk_index: int
    text: str
    page_number: int
    char_start: int
    char_end: int
    token_count: int

class PDFProcessor:
    """Handles PDF document processing and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with chunking parameters
        
        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for deduplication"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def extract_text_pymupdf(self, file_path: Path) -> Tuple[str, int]:
        """
        Extract text from PDF using PyMuPDF (better quality extraction)
        
        Returns:
            Tuple of (full_text, total_pages)
        """
        try:
            doc = fitz.open(file_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                # Clean up text
                text = text.replace('\n', ' ').replace('\r', '')
                text = ' '.join(text.split())  # Remove extra whitespace
                full_text += f"\n[PAGE {page_num + 1}]\n{text}\n"
            
            doc.close()
            return full_text, len(doc)
            
        except Exception as e:
            logger.error(f"Failed to extract text with PyMuPDF: {e}")
            return self.extract_text_pypdf2(file_path)
    
    def extract_text_pypdf2(self, file_path: Path) -> Tuple[str, int]:
        """
        Fallback text extraction using PyPDF2
        
        Returns:
            Tuple of (full_text, total_pages)
        """
        try:
            full_text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text = text.replace('\n', ' ').replace('\r', '')
                    text = ' '.join(text.split())
                    full_text += f"\n[PAGE {page_num + 1}]\n{text}\n"
                
                return full_text, total_pages
                
        except Exception as e:
            logger.error(f"Failed to extract text with PyPDF2: {e}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (1 token ≈ 4 characters for English)
        """
        return len(text) // 4
    
    def intelligent_chunking(self, text: str) -> List[DocumentChunk]:
        """
        Create intelligent chunks that respect sentence and paragraph boundaries
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, finalize current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    chunk_index,
                    current_start,
                    current_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap from previous chunk
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(sentence) - 1
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                chunk_index,
                current_start,
                current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics"""
        # Simple sentence splitting - can be enhanced with NLTK/spaCy for better accuracy
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of current chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break at sentence boundary within overlap
        overlap_text = text[-self.chunk_overlap:]
        sentences = self._split_into_sentences(overlap_text)
        
        if len(sentences) > 1:
            # Return last complete sentence(s) within overlap
            return '. '.join(sentences[1:]) + '.'
        
        return overlap_text
    
    def _create_chunk(self, text: str, chunk_index: int, char_start: int, char_end: int) -> DocumentChunk:
        """Create a DocumentChunk object"""
        # Extract page number from text (simple heuristic)
        page_number = 1
        page_match = text.rfind('[PAGE ')
        if page_match != -1:
            end_match = text.find(']', page_match)
            if end_match != -1:
                try:
                    page_number = int(text[page_match + 6:end_match])
                except ValueError:
                    pass
        
        return DocumentChunk(
            chunk_index=chunk_index,
            text=text,
            page_number=page_number,
            char_start=char_start,
            char_end=char_end,
            token_count=self.estimate_tokens(text)
        )
    
    def process_document(self, file_path: Path) -> Tuple[str, int, List[DocumentChunk]]:
        """
        Complete document processing pipeline
        
        Returns:
            Tuple of (file_hash, total_pages, chunks)
        """
        logger.info(f"Processing document: {file_path}")
        
        # Calculate file hash for deduplication
        file_hash = self.calculate_file_hash(file_path)
        
        # Extract text
        full_text, total_pages = self.extract_text_pymupdf(file_path)
        
        if not full_text.strip():
            raise ValueError(f"No text extracted from {file_path}")
        
        # Create chunks
        chunks = self.intelligent_chunking(full_text)
        
        logger.info(f"Document processed: {len(chunks)} chunks created from {total_pages} pages")
        
        return file_hash, total_pages, chunks