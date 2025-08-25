import logging
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class SentenceChunker(BaseChunker):
    """Chunker that splits text by sentences."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the sentence chunker.
        
        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        super().__init__(chunk_size, chunk_overlap)
        
        logger.info(f"Initialized SentenceChunker with model: {self.model_name}")
    
    def _create_text_splitter(self) -> SentenceTransformersTokenTextSplitter:
        """Create and return the sentence transformer text splitter."""
        return SentenceTransformersTokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model_name=self.model_name
        )
