import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class RecursiveCharacterChunker(BaseChunker):
    """Chunker that splits text recursively by different separators."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        """
        Initialize the recursive character chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators to use for splitting (in order of preference)
        """
        self.separators = separators or ["\n\n", "\n", " ", ""]
        super().__init__(chunk_size, chunk_overlap)
        
        logger.info(f"Initialized RecursiveCharacterChunker with separators: {self.separators}")
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create and return the recursive character text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
