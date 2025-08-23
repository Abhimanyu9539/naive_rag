"""
Base preprocessor interface using LangChain components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document


class BasePreprocessor(ABC):
    """
    Base interface for document preprocessors using LangChain components.
    
    This provides a consistent interface for text cleaning and normalization
    while working with LangChain's native Document class.
    """
    
    def __init__(self):
        """Initialize the base preprocessor."""
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        pass
    
    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocess a single LangChain document.
        
        Args:
            document: LangChain Document to preprocess
            
        Returns:
            Preprocessed Document
        """
        # Create a new document with preprocessed content
        preprocessed_content = self.preprocess_text(document.page_content)
        
        # Create new document with preprocessed content and original metadata
        preprocessed_doc = Document(
            page_content=preprocessed_content,
            metadata=document.metadata.copy()
        )
        
        # Add preprocessing metadata
        preprocessed_doc.metadata['preprocessed'] = True
        preprocessed_doc.metadata['preprocessor'] = self.__class__.__name__
        
        return preprocessed_doc
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess multiple LangChain documents.
        
        Args:
            documents: List of LangChain Documents to preprocess
            
        Returns:
            List of preprocessed Documents
        """
        preprocessed_docs = []
        
        for document in documents:
            preprocessed_doc = self.preprocess_document(document)
            preprocessed_docs.append(preprocessed_doc)
        
        return preprocessed_docs
