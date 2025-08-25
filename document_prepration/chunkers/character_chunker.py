import logging
from langchain.text_splitter import CharacterTextSplitter

from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class CharacterChunker(BaseChunker):
    """Simple chunker that splits text by character count."""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 separator: str = "\n\n"):
        """
        Initialize the character chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separator: Separator to use for splitting
        """
        self.separator = separator
        super().__init__(chunk_size, chunk_overlap)
        
        logger.info(f"Initialized CharacterChunker with separator: '{self.separator}'")
    
    def _create_text_splitter(self) -> CharacterTextSplitter:
        """Create and return the character text splitter."""
        return CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator,
            length_function=len
        )
