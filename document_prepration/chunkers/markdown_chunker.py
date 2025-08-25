import logging
from langchain.text_splitter import MarkdownTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class MarkdownChunker(BaseChunker):
    """Chunker that splits markdown text while preserving structure."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
        Initialize the markdown chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        super().__init__(chunk_size, chunk_overlap)
        
        logger.info("Initialized MarkdownChunker")
    
    def _create_text_splitter(self) -> MarkdownTextSplitter:
        """Create and return the markdown text splitter."""
        return MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
