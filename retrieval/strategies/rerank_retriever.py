"""
Rerank retriever that uses a second-stage reranking model to improve retrieval quality.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from ..base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class RerankRetriever(BaseRetriever):
    """
    Rerank retriever that uses a second-stage reranking model to improve retrieval quality.
    This strategy first retrieves a larger set of candidates, then reranks them using
    a more sophisticated scoring model.
    """
    
    def __init__(self, 
                 embeddings,
                 vector_store,
                 initial_k: int = 20,
                 final_k: int = 4,
                 rerank_strategy: str = "cross_encoder",
                 namespace: Optional[str] = None,
                 **kwargs):
        """
        Initialize the rerank retriever.
        
        Args:
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            initial_k: Number of documents to retrieve in first stage
            final_k: Number of documents to return after reranking
            rerank_strategy: Strategy for reranking ('cross_encoder', 'bm25', 'hybrid')
            namespace: Optional namespace for the vector store
            **kwargs: Additional configuration parameters
        """
        super().__init__(embeddings, vector_store, **kwargs)
        self.initial_k = initial_k
        self.final_k = final_k
        self.rerank_strategy = rerank_strategy
        self.namespace = namespace
        
        logger.info(f"Initialized RerankRetriever with initial_k={initial_k}, final_k={final_k}, strategy={rerank_strategy}")
    
    def calculate_bm25_score(self, query: str, doc_content: str) -> float:
        """
        Calculate BM25 score for a document.
        
        Args:
            query: Query text
            doc_content: Document content
            
        Returns:
            BM25 score
        """
        import re
        from collections import Counter
        
        # Tokenize query and document
        query_tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        doc_tokens = re.findall(r'\b[a-zA-Z]+\b', doc_content.lower())
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        query_tokens = [token for token in query_tokens if token not in stop_words]
        doc_tokens = [token for token in doc_tokens if token not in stop_words]
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Calculate term frequencies
        doc_freq = Counter(doc_tokens)
        doc_length = len(doc_tokens)
        
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        avg_doc_length = 100  # Simplified assumption
        
        # Calculate BM25 score
        score = 0.0
        for term in query_tokens:
            if term in doc_freq:
                tf = doc_freq[term]
                idf = 1.0  # Simplified IDF calculation
                
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def calculate_cross_encoder_score(self, query: str, doc_content: str) -> float:
        """
        Calculate cross-encoder score for a document.
        This is a simplified implementation - in practice, you'd use a proper cross-encoder model.
        
        Args:
            query: Query text
            doc_content: Document content
            
        Returns:
            Cross-encoder score
        """
        # Simplified cross-encoder scoring based on semantic similarity
        # In practice, you'd use models like BERT cross-encoders
        
        # Calculate semantic similarity using embeddings
        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embedding = self.embeddings.embed_query(doc_content[:1000])  # Limit length
            
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating cross-encoder score: {str(e)}")
            return 0.0
    
    def calculate_hybrid_score(self, query: str, doc_content: str, vector_score: float) -> float:
        """
        Calculate hybrid score combining multiple scoring methods.
        
        Args:
            query: Query text
            doc_content: Document content
            vector_score: Original vector similarity score
            
        Returns:
            Hybrid score
        """
        # Get different scores
        bm25_score = self.calculate_bm25_score(query, doc_content)
        cross_encoder_score = self.calculate_cross_encoder_score(query, doc_content)
        
        # Normalize scores to 0-1 range
        bm25_score = min(bm25_score / 10.0, 1.0)  # Rough normalization
        cross_encoder_score = max(0.0, min(1.0, cross_encoder_score))
        vector_score = max(0.0, min(1.0, vector_score))
        
        # Combine scores with weights
        weights = {
            'vector': 0.4,
            'bm25': 0.3,
            'cross_encoder': 0.3
        }
        
        hybrid_score = (
            weights['vector'] * vector_score +
            weights['bm25'] * bm25_score +
            weights['cross_encoder'] * cross_encoder_score
        )
        
        return hybrid_score
    
    def rerank_documents(self, 
                        query: str, 
                        candidates: List[tuple]) -> List[tuple]:
        """
        Rerank candidate documents using the specified strategy.
        
        Args:
            query: Original query
            candidates: List of (Document, vector_score) tuples
            
        Returns:
            List of reranked (Document, final_score) tuples
        """
        reranked_results = []
        
        for doc, vector_score in candidates:
            if self.rerank_strategy == "bm25":
                final_score = self.calculate_bm25_score(query, doc.page_content)
            elif self.rerank_strategy == "cross_encoder":
                final_score = self.calculate_cross_encoder_score(query, doc.page_content)
            elif self.rerank_strategy == "hybrid":
                final_score = self.calculate_hybrid_score(query, doc.page_content, vector_score)
            else:
                # Default to vector score
                final_score = vector_score
            
            reranked_results.append((doc, final_score))
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results
    
    def retrieve(self, 
                query: str, 
                k: int = 4,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using reranking.
        
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
            results_with_scores = self.retrieve_with_scores(query, k, **kwargs)
            
            # Return only documents (without scores)
            results = [doc for doc, _ in results_with_scores]
            
            logger.info(f"Retrieved {len(results)} documents using reranking")
            return results
            
        except Exception as e:
            logger.error(f"Error in rerank retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           **kwargs) -> List[tuple]:
        """
        Retrieve documents with scores using reranking.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            **kwargs: Additional parameters
            
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
            
            # First stage: retrieve initial candidates
            initial_candidates = self.vector_store.similarity_search_with_score(
                query=processed_query,
                embeddings=self.embeddings,
                k=self.initial_k,
                namespace=namespace
            )
            
            logger.debug(f"Retrieved {len(initial_candidates)} initial candidates")
            
            # Second stage: rerank candidates
            reranked_results = self.rerank_documents(processed_query, initial_candidates)
            
            # Return top k results
            final_results = reranked_results[:k]
            
            logger.info(f"Retrieved {len(final_results)} documents using reranking")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in rerank retrieval with scores: {str(e)}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the rerank retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        config = super().get_config()
        config.update({
            'initial_k': self.initial_k,
            'final_k': self.final_k,
            'rerank_strategy': self.rerank_strategy,
            'namespace': self.namespace
        })
        return config
