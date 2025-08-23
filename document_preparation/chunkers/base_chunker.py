"""
Base chunker class that defines the interface for document chunking.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..utils import Document, Chunk


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""
    
    def __init__(self):
        """Initialize the chunker."""
        pass
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        pass
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of Chunk objects
        """
        text_chunks = self.chunk(document.content)
        chunks = []
        
        current_index = 0
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{document.id}_chunk_{i:04d}"
            
            chunk = Chunk(
                content=chunk_text,
                document_id=document.id,
                chunk_id=chunk_id,
                start_index=current_index,
                end_index=current_index + len(chunk_text),
                metadata={
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'document_title': document.title,
                    'document_source': document.source
                }
            )
            chunks.append(chunk)
            current_index += len(chunk_text) + 1  # +1 for potential separator
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of Chunk objects from all documents
        """
        all_chunks = []
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks
