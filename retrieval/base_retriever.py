"""
Base retriever class that defines the interface for all retrieval strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.
    """
    
    def __init__(self, 
                 embeddings: Embeddings,
                 vector_store,
                 **kwargs):
        """
        Initialize the base retriever.
        
        Args:
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            **kwargs: Additional configuration parameters
        """
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.config = kwargs
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def retrieve(self, 
                query: str, 
                k: int = 4,
                **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters specific to the retrieval strategy
            
        Returns:
            List of relevant Document objects
        """
        pass
    
    @abstractmethod
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           **kwargs) -> List[tuple]:
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters specific to the retrieval strategy
            
        Returns:
            List of tuples containing (Document, score)
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        return {
            'retriever_type': self.__class__.__name__,
            'config': self.config
        }
    
    def update_config(self, **kwargs):
        """
        Update the retriever configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
        logger.info(f"Updated configuration for {self.__class__.__name__}")
    
    def validate_query(self, query: str) -> bool:
        """
        Validate if the query is suitable for retrieval.
        
        Args:
            query: Query text to validate
            
        Returns:
            True if query is valid, False otherwise
        """
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query provided")
            return False
        
        if len(query.strip()) < 2:
            logger.warning("Query too short (less than 2 characters)")
            return False
        
        return True
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query before retrieval.
        
        Args:
            query: Raw query text
            
        Returns:
            Preprocessed query text
        """
        # Basic preprocessing: strip whitespace and normalize
        processed_query = query.strip()
        
        # Convert to lowercase for consistency
        processed_query = processed_query.lower()
        
        logger.debug(f"Preprocessed query: '{query}' -> '{processed_query}'")
        return processed_query
    
    def postprocess_results(self, 
                          results: List[Document],
                          **kwargs) -> List[Document]:
        """
        Postprocess retrieval results.
        
        Args:
            results: List of retrieved documents
            **kwargs: Additional parameters for postprocessing
            
        Returns:
            Postprocessed list of documents
        """
        # Remove duplicates based on content
        seen_contents = set()
        unique_results = []
        
        for doc in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(doc)
        
        if len(unique_results) < len(results):
            logger.info(f"Removed {len(results) - len(unique_results)} duplicate documents")
        
        return unique_results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval performance.
        
        Returns:
            Dictionary containing retrieval statistics
        """
        return {
            'retriever_type': self.__class__.__name__,
            'config': self.config,
            'embeddings_type': self.embeddings.__class__.__name__,
            'vector_store_type': self.vector_store.__class__.__name__
        }
