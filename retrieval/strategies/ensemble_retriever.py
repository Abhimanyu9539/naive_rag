"""
Ensemble retriever that combines multiple retrieval strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from collections import defaultdict

from ..base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines multiple retrieval strategies.
    This strategy can improve retrieval performance by leveraging the strengths
    of different retrieval methods.
    """
    
    def __init__(self, 
                 retrievers: List[Tuple[BaseRetriever, float]],
                 ensemble_strategy: str = "weighted",
                 **kwargs):
        """
        Initialize the ensemble retriever.
        
        Args:
            retrievers: List of (retriever, weight) tuples
            ensemble_strategy: Strategy for combining results ('weighted', 'voting', 'rank_fusion')
            **kwargs: Additional configuration parameters
        """
        # Use the first retriever's embeddings and vector_store for compatibility
        first_retriever = retrievers[0][0]
        super().__init__(first_retriever.embeddings, first_retriever.vector_store, **kwargs)
        
        self.retrievers = retrievers
        self.ensemble_strategy = ensemble_strategy
        
        # Normalize weights
        total_weight = sum(weight for _, weight in retrievers)
        self.normalized_retrievers = [
            (retriever, weight / total_weight) for retriever, weight in retrievers
        ]
        
        logger.info(f"Initialized EnsembleRetriever with {len(retrievers)} retrievers using {ensemble_strategy} strategy")
    
    def combine_weighted_results(self, 
                               all_results: List[List[tuple]], 
                               k: int) -> List[tuple]:
        """
        Combine results using weighted averaging.
        
        Args:
            all_results: List of results from each retriever
            k: Number of final results to return
            
        Returns:
            List of combined (Document, score) tuples
        """
        # Collect all documents with their weighted scores
        doc_scores = defaultdict(list)
        
        for i, results in enumerate(all_results):
            retriever, weight = self.normalized_retrievers[i]
            
            for doc, score in results:
                content_hash = hash(doc.page_content)
                doc_scores[content_hash].append((doc, score * weight))
        
        # Calculate final scores (sum of weighted scores)
        final_results = []
        for content_hash, doc_score_list in doc_scores.items():
            doc = doc_score_list[0][0]  # Take the first document instance
            final_score = sum(score for _, score in doc_score_list)
            final_results.append((doc, final_score))
        
        # Sort by final score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
    
    def combine_voting_results(self, 
                             all_results: List[List[tuple]], 
                             k: int) -> List[tuple]:
        """
        Combine results using voting mechanism.
        
        Args:
            all_results: List of results from each retriever
            k: Number of final results to return
            
        Returns:
            List of combined (Document, score) tuples
        """
        # Count votes for each document
        doc_votes = defaultdict(int)
        doc_scores = defaultdict(list)
        
        for i, results in enumerate(all_results):
            retriever, weight = self.normalized_retrievers[i]
            
            for rank, (doc, score) in enumerate(results):
                content_hash = hash(doc.page_content)
                doc_votes[content_hash] += 1
                doc_scores[content_hash].append((doc, score))
        
        # Calculate final scores based on votes and average scores
        final_results = []
        for content_hash, votes in doc_votes.items():
            doc = doc_scores[content_hash][0][0]
            avg_score = sum(score for _, score in doc_scores[content_hash]) / len(doc_scores[content_hash])
            
            # Combine votes and average score
            final_score = (votes * 0.7) + (avg_score * 0.3)
            final_results.append((doc, final_score))
        
        # Sort by final score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
    
    def combine_rank_fusion_results(self, 
                                  all_results: List[List[tuple]], 
                                  k: int) -> List[tuple]:
        """
        Combine results using rank fusion (Reciprocal Rank Fusion).
        
        Args:
            all_results: List of results from each retriever
            k: Number of final results to return
            
        Returns:
            List of combined (Document, score) tuples
        """
        # Calculate RRF scores
        doc_rrf_scores = defaultdict(float)
        
        for i, results in enumerate(all_results):
            retriever, weight = self.normalized_retrievers[i]
            
            for rank, (doc, _) in enumerate(results):
                content_hash = hash(doc.page_content)
                # RRF formula: 1 / (k + rank), where k is typically 60
                rrf_score = weight / (60 + rank + 1)
                doc_rrf_scores[content_hash] += rrf_score
        
        # Convert to final results
        final_results = []
        for content_hash, rrf_score in doc_rrf_scores.items():
            # Find the document instance
            for results in all_results:
                for doc, _ in results:
                    if hash(doc.page_content) == content_hash:
                        final_results.append((doc, rrf_score))
                        break
                else:
                    continue
                break
        
        # Sort by RRF score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
    
    def retrieve(self, 
                query: str, 
                k: int = 4,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using ensemble of retrievers.
        
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
            
            logger.info(f"Retrieved {len(results)} documents using ensemble search")
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           **kwargs) -> List[tuple]:
        """
        Retrieve documents with scores using ensemble of retrievers.
        
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
            # Get results from all retrievers
            all_results = []
            
            for retriever, weight in self.normalized_retrievers:
                try:
                    # Get results from this retriever
                    results = retriever.retrieve_with_scores(query, k * 2, **kwargs)
                    all_results.append(results)
                    
                    logger.debug(f"Retriever {retriever.__class__.__name__} returned {len(results)} results")
                    
                except Exception as e:
                    logger.warning(f"Error in retriever {retriever.__class__.__name__}: {str(e)}")
                    all_results.append([])
            
            # Combine results based on strategy
            if self.ensemble_strategy == "weighted":
                final_results = self.combine_weighted_results(all_results, k)
            elif self.ensemble_strategy == "voting":
                final_results = self.combine_voting_results(all_results, k)
            elif self.ensemble_strategy == "rank_fusion":
                final_results = self.combine_rank_fusion_results(all_results, k)
            else:
                logger.warning(f"Unknown ensemble strategy: {self.ensemble_strategy}, using weighted")
                final_results = self.combine_weighted_results(all_results, k)
            
            logger.info(f"Retrieved {len(final_results)} documents using ensemble search")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in ensemble retrieval with scores: {str(e)}")
            return []
    
    def get_retriever_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all retrievers in the ensemble.
        
        Returns:
            List of retriever information dictionaries
        """
        info = []
        for retriever, weight in self.normalized_retrievers:
            info.append({
                'retriever_type': retriever.__class__.__name__,
                'weight': weight,
                'config': retriever.get_config()
            })
        return info
    
    def update_weights(self, new_weights: List[float]):
        """
        Update the weights of retrievers in the ensemble.
        
        Args:
            new_weights: List of new weights (must match number of retrievers)
        """
        if len(new_weights) != len(self.retrievers):
            raise ValueError(f"Number of weights ({len(new_weights)}) must match number of retrievers ({len(self.retrievers)})")
        
        # Update retrievers with new weights
        self.retrievers = [(retriever, weight) for (retriever, _), weight in zip(self.retrievers, new_weights)]
        
        # Re-normalize weights
        total_weight = sum(weight for _, weight in self.retrievers)
        self.normalized_retrievers = [
            (retriever, weight / total_weight) for retriever, weight in self.retrievers
        ]
        
        logger.info(f"Updated ensemble weights: {new_weights}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the ensemble retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        config = super().get_config()
        config.update({
            'ensemble_strategy': self.ensemble_strategy,
            'num_retrievers': len(self.retrievers),
            'retriever_weights': [weight for _, weight in self.normalized_retrievers],
            'retriever_types': [retriever.__class__.__name__ for retriever, _ in self.retrievers]
        })
        return config
