"""
Utility classes and functions for document preparation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib
from datetime import datetime


@dataclass
class Document:
    """Represents a document with metadata and content."""
    
    content: str
    source: str  # File path or URL
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.title is None:
            self.title = Path(self.source).stem if self.source else "Unknown"
    
    @property
    def id(self) -> str:
        """Generate a unique ID for the document."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"{self.source}_{content_hash[:8]}"
    
    @property
    def word_count(self) -> int:
        """Get the word count of the document."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Get the character count of the document."""
        return len(self.content)


@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    
    content: str
    document_id: str
    chunk_id: str
    start_index: int
    end_index: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def word_count(self) -> int:
        """Get the word count of the chunk."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Get the character count of the chunk."""
        return len(self.content)


def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    return Path(file_path).suffix.lower()


def is_supported_file(file_path: str) -> bool:
    """Check if the file type is supported."""
    supported_extensions = {'.pdf', '.txt', '.docx', '.doc'}
    return get_file_extension(file_path) in supported_extensions


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing invalid characters."""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    return sanitized


def create_chunk_id(document_id: str, chunk_index: int) -> str:
    """Create a unique chunk ID."""
    return f"{document_id}_chunk_{chunk_index:04d}"


def merge_chunks(chunks: List[Chunk]) -> str:
    """Merge multiple chunks back into a single text."""
    return " ".join(chunk.content for chunk in chunks)


def calculate_overlap_tokens(text1: str, text2: str, tokenizer) -> int:
    """Calculate the number of overlapping tokens between two texts."""
    tokens1 = tokenizer.encode(text1)
    tokens2 = tokenizer.encode(text2)
    
    # Find common tokens at the end of text1 and beginning of text2
    min_len = min(len(tokens1), len(tokens2))
    overlap = 0
    
    for i in range(1, min_len + 1):
        if tokens1[-i:] == tokens2[:i]:
            overlap = i
    
    return overlap
