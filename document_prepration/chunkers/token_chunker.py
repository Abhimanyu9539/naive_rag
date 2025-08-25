import logging
from langchain.text_splitter import TokenTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class TokenChunker(BaseChunker):
    """Chunker that splits text by token count using tiktoken."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 encoding_name: str = "cl100k_base"):
        """
        Initialize the token chunker.
        
        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Name of the tiktoken encoding to use
        """
        self.encoding_name = encoding_name
        super().__init__(chunk_size, chunk_overlap)
        
        logger.info(f"Initialized TokenChunker with encoding: {self.encoding_name}")
    
    def _create_text_splitter(self) -> TokenTextSplitter:
        """Create and return the token text splitter."""
        return TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=self.encoding_name
        )
