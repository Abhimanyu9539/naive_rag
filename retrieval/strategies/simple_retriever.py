"""
Simple retriever that uses basic vector similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from ..base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class SimpleRetriever(BaseRetriever):
    """
    Simple retriever that performs basic vector similarity search.
    This is the most straightforward retrieval strategy.
    """
    
    def __init__(self, 
                 embeddings,
                 vector_store,
                 namespace: Optional[str] = None,
                 **kwargs):
        """
        Initialize the simple retriever.
        
        Args:
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            namespace: Optional namespace for the vector store
            **kwargs: Additional configuration parameters
        """
        super().__init__(embeddings, vector_store, **kwargs)
        self.namespace = namespace
        
        logger.info(f"Initialized SimpleRetriever with namespace: {namespace}")
    
    def retrieve(self, 
                query: str, 
                k: int = 4,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using simple vector similarity search.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters (namespace override, etc.)
            
        Returns:
            List of relevant Document objects
        """
        if not self.validate_query(query):
            return []
        
        try:
            # Preprocess query
            processed_query = self.preprocess_query(query)
            
            # Get namespace from kwargs or use default
            namespace = kwargs.get('namespace', self.namespace)
            
            # Perform similarity search
            results = self.vector_store.similarity_search(
                query=processed_query,
                embeddings=self.embeddings,
                k=k,
                namespace=namespace
            )
            
            # Postprocess results
            results = self.postprocess_results(results)
            
            logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in simple retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           **kwargs) -> List[tuple]:
        """
        Retrieve documents with similarity scores using simple vector search.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters (namespace override, etc.)
            
        Returns:
            List of tuples containing (Document, score)
        """
        if not self.validate_query(query):
            return []
        
        try:
            # Preprocess query
            processed_query = self.preprocess_query(query)
            
            # Get namespace from kwargs or use default
            namespace = kwargs.get('namespace', self.namespace)
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=processed_query,
                embeddings=self.embeddings,
                k=k,
                namespace=namespace
            )
            
            # Postprocess results (remove duplicates)
            seen_contents = set()
            unique_results = []
            
            for doc, score in results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append((doc, score))
            
            logger.info(f"Retrieved {len(unique_results)} documents with scores for query: {query[:50]}...")
            return unique_results
            
        except Exception as e:
            logger.error(f"Error in simple retrieval with scores: {str(e)}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the simple retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        config = super().get_config()
        config.update({
            'namespace': self.namespace
        })
        return config
