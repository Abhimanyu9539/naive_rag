"""
Hybrid retriever that combines vector similarity with keyword-based search.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from collections import Counter

from ..base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines vector similarity search with keyword-based search.
    This strategy can improve retrieval by leveraging both semantic and lexical matching.
    """
    
    def __init__(self, 
                 embeddings,
                 vector_store,
                 vector_weight: float = 0.7,
                 keyword_weight: float = 0.3,
                 namespace: Optional[str] = None,
                 **kwargs):
        """
        Initialize the hybrid retriever.
        
        Args:
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            vector_weight: Weight for vector similarity scores (0.0 to 1.0)
            keyword_weight: Weight for keyword matching scores (0.0 to 1.0)
            namespace: Optional namespace for the vector store
            **kwargs: Additional configuration parameters
        """
        super().__init__(embeddings, vector_store, **kwargs)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.namespace = namespace
        
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        self.vector_weight /= total_weight
        self.keyword_weight /= total_weight
        
        logger.info(f"Initialized HybridRetriever with vector_weight={self.vector_weight:.2f}, keyword_weight={self.keyword_weight:.2f}")
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using simple tokenization.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction: lowercase, split on whitespace, remove punctuation
        keywords = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Remove common stop words (basic list)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in keywords if word not in stop_words and len(word) > 2]
        return keywords
    
    def calculate_keyword_score(self, query_keywords: List[str], doc_content: str) -> float:
        """
        Calculate keyword matching score between query and document.
        
        Args:
            query_keywords: Keywords extracted from query
            doc_content: Document content
            
        Returns:
            Keyword matching score (0.0 to 1.0)
        """
        if not query_keywords:
            return 0.0
        
        doc_keywords = self.extract_keywords(doc_content)
        doc_keyword_freq = Counter(doc_keywords)
        
        # Calculate TF-IDF like score
        total_matches = 0
        for keyword in query_keywords:
            if keyword in doc_keyword_freq:
                total_matches += doc_keyword_freq[keyword]
        
        # Normalize by query length and document length
        if not doc_keywords:
            return 0.0
        
        score = total_matches / (len(query_keywords) * len(doc_keywords))
        return min(score, 1.0)  # Cap at 1.0
    
    def retrieve(self, 
                query: str, 
                k: int = 4,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using hybrid search (vector + keyword).
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of relevant Document objects
        """
        if not self.validate_query(query):
            return []
        
        try:
            # Get results with scores
            results_with_scores = self.retrieve_with_scores(query, k * 2, **kwargs)
            
            # Return only documents (without scores)
            results = [doc for doc, _ in results_with_scores[:k]]
            
            logger.info(f"Retrieved {len(results)} documents using hybrid search")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           **kwargs) -> List[tuple]:
        """
        Retrieve documents with hybrid scores (vector + keyword).
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of tuples containing (Document, hybrid_score)
        """
        if not self.validate_query(query):
            return []
        
        try:
            # Preprocess query
            processed_query = self.preprocess_query(query)
            
            # Extract keywords from query
            query_keywords = self.extract_keywords(processed_query)
            
            # Get namespace from kwargs or use default
            namespace = kwargs.get('namespace', self.namespace)
            
            # Get vector similarity results
            vector_results = self.vector_store.similarity_search_with_score(
                query=processed_query,
                embeddings=self.embeddings,
                k=k * 2,  # Get more results for better hybrid scoring
                namespace=namespace
            )
            
            # Calculate hybrid scores
            hybrid_results = []
            for doc, vector_score in vector_results:
                # Calculate keyword score
                keyword_score = self.calculate_keyword_score(query_keywords, doc.page_content)
                
                # Combine scores using weighted average
                hybrid_score = (self.vector_weight * vector_score + 
                              self.keyword_weight * keyword_score)
                
                hybrid_results.append((doc, hybrid_score))
            
            # Sort by hybrid score (higher is better)
            hybrid_results.sort(key=lambda x: x[1], reverse=True)
            
            # Remove duplicates
            seen_contents = set()
            unique_results = []
            
            for doc, score in hybrid_results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append((doc, score))
            
            # Return top k results
            final_results = unique_results[:k]
            
            logger.info(f"Retrieved {len(final_results)} documents with hybrid scores")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval with scores: {str(e)}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the hybrid retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        config = super().get_config()
        config.update({
            'vector_weight': self.vector_weight,
            'keyword_weight': self.keyword_weight,
            'namespace': self.namespace
        })
        return config
