import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple












# define default values
DEFAULT_SUPPORTED_FORMATS = ["JPEG", "PNG", "WEBP", "GIF"]
DEFAULT_MAX_IMAGE_SIZE = (1024, 1024)



@dataclass
class Config:
    # Gemini Models (Free Tier)
    CHAT_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/embedding-001"
    VISION_MODEL: str = "gemini-2.5-flash"
    
    # API Configuration
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "gemini_rag_collection"
    
    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_RESULTS: int = 5
    
    # Image Processing
    MAX_IMAGE_SIZE: Tuple[int, int] = DEFAULT_MAX_IMAGE_SIZE
    SUPPORTED_IMAGE_FORMATS: List[str] = field(default_factory=lambda: DEFAULT_SUPPORTED_FORMATS.copy())
    
    # Rate Limiting (Free Tier Limits)
    MAX_REQUESTS_PER_MINUTE: int = 10
    MAX_TOKENS_PER_REQUEST: int = 32768
    

    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get configuration for Gemini API"""
        return {
            "api_key": self.GEMINI_API_KEY,
            "chat_model": self.CHAT_MODEL,
            "embedding_model": self.EMBEDDING_MODEL,
            "vision_model": self.VISION_MODEL
        }
    
    def get_chroma_config(self) -> Dict[str, Any]:
        """Get configuration for ChromaDB"""
        return {
            "persist_directory": self.CHROMA_DB_PATH,
            "collection_name": self.COLLECTION_NAME
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        return True

