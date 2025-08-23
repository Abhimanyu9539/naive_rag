"""
Base preprocessor class that defines the interface for text preprocessing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..utils import Document


class BasePreprocessor(ABC):
    """Abstract base class for text preprocessors."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        pass
    
    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocess a document.
        
        Args:
            document: Document object to preprocess
            
        Returns:
            Document with preprocessed content
        """
        preprocessed_content = self.preprocess(document.content)
        
        # Create a new document with preprocessed content
        return Document(
            content=preprocessed_content,
            source=document.source,
            title=document.title,
            author=document.author,
            created_date=document.created_date,
            modified_date=document.modified_date,
            metadata=document.metadata
        )
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess multiple documents.
        
        Args:
            documents: List of Document objects to preprocess
            
        Returns:
            List of Document objects with preprocessed content
        """
        return [self.preprocess_document(doc) for doc in documents]
