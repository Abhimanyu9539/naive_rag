"""
Base chunker interface using LangChain text splitters.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseChunker(ABC):
    """
    Base interface for document chunkers using LangChain text splitters.
    
    This provides a consistent interface while leveraging LangChain's
    native text splitting functionality.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the base chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = self._create_text_splitter()
    
    @abstractmethod
    def _create_text_splitter(self):
        """Create the appropriate text splitter."""
        pass
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a single document.
        
        Args:
            document: LangChain Document to chunk
            
        Returns:
            List of chunked Document objects
        """
        try:
            # Use LangChain's text splitter to preserve metadata
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk metadata
            document_id = document.metadata.get('source', 'unknown')
            chunks = self.add_chunk_metadata(chunks, document_id)
            
            return chunks
            
        except Exception as e:
            print(f"Error chunking document: {e}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of LangChain Documents to chunk
            
        Returns:
            List of chunked Document objects
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def add_chunk_metadata(self, chunks: List[Document], document_id: str) -> List[Document]:
        """
        Add chunk-specific metadata to documents.
        
        Args:
            chunks: List of chunked documents
            document_id: ID of the source document
            
        Returns:
            Updated documents with chunk metadata
        """
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': f"{document_id}_chunk_{i:04d}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source_document_id': document_id
            })
        
        return chunks
