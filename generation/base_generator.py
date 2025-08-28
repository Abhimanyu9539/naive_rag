"""
Base generator class that defines the interface for all generation strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for all generation strategies.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 **kwargs):
        """
        Initialize the base generator.
        
        Args:
            llm: Language model instance to use for generation
            **kwargs: Additional configuration parameters
        """
        self.llm = llm
        self.config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def generate(self, 
                query: str,
                retrieved_docs: List[Document],
                **kwargs) -> str:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: User's query
            retrieved_docs: List of retrieved documents
            **kwargs: Additional parameters specific to the generation strategy
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def generate_with_metadata(self, 
                             query: str,
                             retrieved_docs: List[Document],
                             **kwargs) -> Dict[str, Any]:
        """
        Generate a response with additional metadata.
        
        Args:
            query: User's query
            retrieved_docs: List of retrieved documents
            **kwargs: Additional parameters specific to the generation strategy
            
        Returns:
            Dictionary containing response and metadata
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the generator.
        
        Returns:
            Dictionary containing the generator configuration
        """
        return {
            'generator_type': self.__class__.__name__,
            'llm_type': self.llm.__class__.__name__,
            'config': self.config
        }
    
    def update_config(self, **kwargs):
        """
        Update the generator configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
        logger.info(f"Updated configuration for {self.__class__.__name__}")
    
    def validate_inputs(self, query: str, retrieved_docs: List[Document]) -> bool:
        """
        Validate if the inputs are suitable for generation.
        
        Args:
            query: Query text to validate
            retrieved_docs: Retrieved documents to validate
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query provided")
            return False
        
        if not retrieved_docs:
            logger.warning("No retrieved documents provided")
            return False
        
        if len(query.strip()) < 2:
            logger.warning("Query too short (less than 2 characters)")
            return False
        
        return True
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query before generation.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query text
        """
        # Basic preprocessing: strip whitespace
        processed_query = query.strip()
        
        logger.debug(f"Preprocessed query: '{query}' -> '{processed_query}'")
        return processed_query
    
    def preprocess_documents(self, 
                           documents: List[Document],
                           max_length: int = 4000) -> List[Document]:
        """
        Preprocess retrieved documents for generation.
        
        Args:
            documents: List of documents to preprocess
            max_length: Maximum total length of document content
            
        Returns:
            Preprocessed list of documents
        """
        if not documents:
            return []
        
        # Sort documents by relevance if they have metadata scores
        sorted_docs = sorted(
            documents,
            key=lambda doc: doc.metadata.get('score', 0) if doc.metadata else 0,
            reverse=True
        )
        
        # Truncate documents if total length exceeds max_length
        total_length = 0
        processed_docs = []
        
        for doc in sorted_docs:
            doc_length = len(doc.page_content)
            if total_length + doc_length <= max_length:
                processed_docs.append(doc)
                total_length += doc_length
            else:
                # Truncate this document if it would exceed the limit
                remaining_length = max_length - total_length
                if remaining_length > 100:  # Only include if we have meaningful content
                    truncated_content = doc.page_content[:remaining_length] + "..."
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata=doc.metadata
                    )
                    processed_docs.append(truncated_doc)
                break
        
        logger.info(f"Preprocessed {len(documents)} documents to {len(processed_docs)} documents")
        return processed_docs
    
    def create_context(self, 
                      query: str,
                      documents: List[Document]) -> str:
        """
        Create context string from query and documents.
        
        Args:
            query: User's query
            documents: Retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = [f"Query: {query}\n\n"]
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', f'Document {i}')
            context_parts.append(f"Document {i} (Source: {source}):\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the generation performance.
        
        Returns:
            Dictionary containing generation statistics
        """
        return {
            'generator_type': self.__class__.__name__,
            'llm_type': self.llm.__class__.__name__,
            'config': self.config
        }
