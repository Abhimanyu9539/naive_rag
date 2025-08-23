"""
Base loader interface using LangChain components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


class BaseDocumentLoader(ABC):
    """
    Base interface for document loaders using LangChain components.
    
    This provides a consistent interface while leveraging LangChain's
    native loader functionality.
    """
    
    def __init__(self):
        """Initialize the base loader."""
        self.supported_extensions = set()
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
        """
        pass
    
    def can_load(self, file_path: str) -> bool:
        """
        Check if this loader can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the loader can handle this file type
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions
    
    def add_metadata(self, documents: List[Document], source: str, **kwargs) -> List[Document]:
        """
        Add metadata to documents.
        
        Args:
            documents: List of documents to update
            source: Source file path
            **kwargs: Additional metadata
            
        Returns:
            Updated documents with metadata
        """
        for doc in documents:
            doc.metadata.update({
                'source': source,
                'loader_type': self.__class__.__name__,
                **kwargs
            })
        return documents
