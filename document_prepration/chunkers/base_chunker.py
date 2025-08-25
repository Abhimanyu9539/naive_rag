import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.text_splitter import TextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Base class for all document chunkers."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the base chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = self._create_text_splitter()
        
        logger.info(f"Initialized {self.__class__.__name__} with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    @abstractmethod
    def _create_text_splitter(self) -> TextSplitter:
        """Create and return the specific text splitter implementation."""
        pass
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Split text into chunks using LangChain's split_text method.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of LangChain Document objects
        """
        try:
            logger.debug(f"Chunking text of length {len(text)} characters")
            
            # Use LangChain's split_text method
            chunks = self.text_splitter.split_text(text)
            
            # Convert to Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            logger.info(f"Successfully created {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk multiple documents using LangChain's split_documents method.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            logger.debug(f"Chunking {len(documents)} documents")
            
            # Use LangChain's split_documents method directly
            chunked_docs = self.text_splitter.split_documents(documents)
            
            logger.info(f"Successfully created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise
