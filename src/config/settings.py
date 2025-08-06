"""
Configuration settings for the RAG system.
Handles environment variables and default configurations.
"""

import os
from typing import Dict, Any
from pathlib import Path

class Settings:
    """Application settings and configuration"""
    
    def __init__(self):
        """Initialize settings from environment variables and defaults"""
        
        # Database configuration
        self.database_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'rag_system'),
            'user': os.getenv('POSTGRES_USER', 'rag_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'rag_password')
        }
        
        # Claude API configuration
        self.claude_api_key = os.getenv('CLAUDE_API_KEY')
        if not self.claude_api_key:
            raise ValueError("CLAUDE_API_KEY environment variable is required")
        
        self.claude_model = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')
        
        # Embedding model configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.device = os.getenv('DEVICE', None)  # None for auto-detection
        
        # Document processing configuration
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        
        # API configuration
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', 8000))
        
        # File storage paths
        self.data_dir = Path(os.getenv('DATA_DIR', 'data'))
        self.documents_dir = self.data_dir / 'documents'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Performance settings
        self.max_workers = int(os.getenv('MAX_WORKERS', 4))
        self.batch_size = int(os.getenv('BATCH_SIZE', 32))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding sensitive data)"""
        return {
            'database_host': self.database_params['host'],
            'database_port': self.database_params['port'],
            'database_name': self.database_params['database'],
            'claude_model': self.claude_model,
            'embedding_model': self.embedding_model,
            'device': self.device,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'api_host': self.api_host,
            'api_port': self.api_port,
            'log_level': self.log_level,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }
    
    @classmethod
    def load_from_env_file(cls, env_file: Path) -> 'Settings':
        """Load settings from .env file"""
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv(env_file)
        return cls()