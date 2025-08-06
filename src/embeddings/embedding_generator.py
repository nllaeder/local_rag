"""
Embedding generation module using sentence-transformers.
Handles local embedding generation without external API calls.
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles local embedding generation using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
            device: Device to run on ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        logger.info(f"Initializing embedding model {model_name} on {device}")
        
        # Load the model
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate embedding
        embedding = self.model.encode(text.strip(), convert_to_numpy=True)
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding arrays
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        logger.info(f"Generating embeddings for {len(valid_texts)} texts")
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return [emb for emb in embeddings]
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        Identical to generate_embedding but semantically different for clarity
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector for the query
        """
        return self.generate_embedding(query)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'Unknown')
        }
    
    def warm_up(self) -> None:
        """Warm up the model with a dummy inference"""
        logger.info("Warming up embedding model...")
        self.generate_embedding("This is a test sentence for model warm-up.")
        logger.info("Model warm-up complete")
    
    @staticmethod
    def get_recommended_models() -> dict:
        """Get dictionary of recommended models for different use cases"""
        return {
            "fast": "all-MiniLM-L6-v2",  # 384 dims, fast and good quality
            "balanced": "all-mpnet-base-v2",  # 768 dims, best overall quality
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dims, supports 50+ languages
            "large": "all-mpnet-base-v2"  # 768 dims, highest quality but slower
        }
    
    def verify_embedding_quality(self, test_texts: Optional[List[str]] = None) -> dict:
        """
        Verify embedding quality with test texts
        
        Args:
            test_texts: Optional test texts, uses defaults if None
            
        Returns:
            Dictionary with quality metrics
        """
        if test_texts is None:
            test_texts = [
                "The cat sits on the mat",
                "A feline rests on the rug",
                "Dogs are running in the park",
                "Python is a programming language"
            ]
        
        embeddings = self.generate_embeddings_batch(test_texts, show_progress=False)
        
        # Calculate similarities
        sim_similar = self.cosine_similarity(embeddings[0], embeddings[1])  # Should be high
        sim_different = self.cosine_similarity(embeddings[0], embeddings[2])  # Should be lower
        
        return {
            "similar_sentences_similarity": sim_similar,
            "different_sentences_similarity": sim_different,
            "embedding_dimension": self.embedding_dim,
            "quality_check": "PASS" if sim_similar > sim_different else "FAIL"
        }