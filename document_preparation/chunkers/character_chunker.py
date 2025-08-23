"""
Character-based document chunker using LangChain's RecursiveCharacterTextSplitter.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker


class CharacterChunker(BaseChunker):
    """Character-based document chunker using LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Initialize the character chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlapping characters between chunks
            separators: List of separators to use for splitting (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def _create_text_splitter(self):
        """Create a character-based text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )
    
    def get_character_count(self, text: str) -> int:
        """
        Get the character count for a text.
        
        Args:
            text: Text to count characters for
            
        Returns:
            Number of characters
        """
        return len(text)
    
    def get_chunk_character_counts(self, chunks: List[Document]) -> List[int]:
        """
        Get character counts for a list of chunks.
        
        Args:
            chunks: List of chunked documents
            
        Returns:
            List of character counts
        """
        return [self.get_character_count(chunk.page_content) for chunk in chunks]
