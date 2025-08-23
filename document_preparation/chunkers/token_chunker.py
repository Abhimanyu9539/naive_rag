"""
Token-based chunker for splitting documents by token count.
"""

import re
from typing import List, Optional

from .base_chunker import BaseChunker


class TokenChunker(BaseChunker):
    """Chunker that splits documents based on token count."""
    
    def __init__(self, 
                 max_tokens: int = 500,
                 overlap_tokens: int = 50,
                 tokenizer_name: str = "cl100k_base"):
        """
        Initialize the token chunker.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            tokenizer_name: Name of the tokenizer to use
        """
        super().__init__()
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Get or create the tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self.tokenizer_name)
            except ImportError:
                raise ImportError("tiktoken is required for token-based chunking. Install with: pip install tiktoken")
        return self._tokenizer
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Encode text to tokens
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.max_tokens:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Calculate end position for this chunk
            end = start + self.max_tokens
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(tokens) and self.overlap_tokens > 0:
                # Look for a good break point within the overlap region
                break_point = self._find_break_point(tokens, start, end)
                end = break_point
            
            # Extract tokens for this chunk
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Clean up the chunk text
            chunk_text = self._clean_chunk_text(chunk_text)
            
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            # Move start position for next chunk
            if end >= len(tokens):
                break
            
            start = end - self.overlap_tokens
            if start >= len(tokens):
                break
        
        return chunks
    
    def _find_break_point(self, tokens: List[int], start: int, end: int) -> int:
        """
        Find a good break point within the overlap region.
        
        Args:
            tokens: List of token IDs
            start: Start position of current chunk
            end: End position of current chunk
            
        Returns:
            Optimal break point position
        """
        # Look for sentence endings in the overlap region
        overlap_start = max(start, end - self.overlap_tokens)
        
        # Common sentence ending tokens (period, exclamation, question mark)
        sentence_endings = ['.', '!', '?', '\n', '\n\n']
        
        # Convert tokens back to text to find sentence boundaries
        overlap_text = self.tokenizer.decode(tokens[overlap_start:end])
        
        # Find the last sentence ending
        for ending in sentence_endings:
            if ending in overlap_text:
                # Find the position of the last occurrence
                last_pos = overlap_text.rfind(ending)
                if last_pos > 0:
                    # Convert character position back to token position
                    char_count = 0
                    for i, token in enumerate(tokens[overlap_start:end]):
                        char_count += len(self.tokenizer.decode([token]))
                        if char_count > last_pos:
                            return overlap_start + i + 1
        
        # If no good break point found, use the original end
        return end
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean up chunk text by removing incomplete sentences at boundaries.
        
        Args:
            text: Raw chunk text
            
        Returns:
            Cleaned chunk text
        """
        # Remove incomplete sentences at the beginning
        text = re.sub(r'^[^.!?]*[.!?]\s*', '', text)
        
        # Remove incomplete sentences at the end
        text = re.sub(r'\s*[^.!?]*$', '', text)
        
        return text.strip()
    
    def chunk_by_sentences(self, text: str, max_sentences: int = 10) -> List[str]:
        """
        Alternative chunking method based on sentence count.
        
        Args:
            text: Text to split into chunks
            max_sentences: Maximum number of sentences per chunk
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # Check if adding this sentence would exceed limits
            if (len(current_chunk) >= max_sentences or 
                current_tokens + sentence_tokens > self.max_tokens):
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
