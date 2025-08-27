"""
Time-aware retriever that considers temporal aspects in retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain.schema import Document

from ..base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class TimeAwareRetriever(BaseRetriever):
    """
    Time-aware retriever that considers temporal aspects in retrieval.
    This strategy is useful for documents with timestamps or when recency matters.
    """
    
    def __init__(self, 
                 embeddings,
                 vector_store,
                 time_decay_factor: float = 0.1,
                 recency_weight: float = 0.3,
                 namespace: Optional[str] = None,
                 **kwargs):
        """
        Initialize the time-aware retriever.
        
        Args:
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            time_decay_factor: Factor for time decay (0.0 to 1.0)
            recency_weight: Weight for recency in scoring (0.0 to 1.0)
            namespace: Optional namespace for the vector store
            **kwargs: Additional configuration parameters
        """
        super().__init__(embeddings, vector_store, **kwargs)
        self.time_decay_factor = time_decay_factor
        self.recency_weight = recency_weight
        self.namespace = namespace
        
        logger.info(f"Initialized TimeAwareRetriever with time_decay_factor={time_decay_factor}, recency_weight={recency_weight}")
    
    def extract_timestamp(self, doc: Document) -> Optional[datetime]:
        """
        Extract timestamp from document metadata.
        
        Args:
            doc: Document to extract timestamp from
            
        Returns:
            Timestamp if available, None otherwise
        """
        if not doc.metadata:
            return None
        
        # Try different timestamp fields
        timestamp_fields = ['timestamp', 'date', 'created_at', 'updated_at', 'published_at']
        
        for field in timestamp_fields:
            if field in doc.metadata:
                timestamp = doc.metadata[field]
                try:
                    if isinstance(timestamp, str):
                        # Try different date formats
                        formats = [
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%d',
                            '%Y-%m-%dT%H:%M:%S',
                            '%Y-%m-%dT%H:%M:%SZ',
                            '%d/%m/%Y',
                            '%m/%d/%Y'
                        ]
                        
                        for fmt in formats:
                            try:
                                return datetime.strptime(timestamp, fmt)
                            except ValueError:
                                continue
                    elif isinstance(timestamp, (int, float)):
                        # Unix timestamp
                        return datetime.fromtimestamp(timestamp)
                    elif isinstance(timestamp, datetime):
                        return timestamp
                        
                except Exception as e:
                    logger.debug(f"Error parsing timestamp from {field}: {str(e)}")
                    continue
        
        return None
    
    def calculate_time_score(self, doc: Document, reference_time: Optional[datetime] = None) -> float:
        """
        Calculate time-based score for a document.
        
        Args:
            doc: Document to score
            reference_time: Reference time (defaults to current time)
            
        Returns:
            Time-based score (0.0 to 1.0)
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        doc_timestamp = self.extract_timestamp(doc)
        
        if doc_timestamp is None:
            # If no timestamp, give neutral score
            return 0.5
        
        # Calculate time difference in days
        time_diff = abs((reference_time - doc_timestamp).days)
        
        # Apply exponential decay
        time_score = 1.0 / (1.0 + self.time_decay_factor * time_diff)
        
        return time_score
    
    def retrieve(self, 
                query: str, 
                k: int = 4,
                reference_time: Optional[datetime] = None,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using time-aware search.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            reference_time: Reference time for temporal scoring
            **kwargs: Additional parameters
            
        Returns:
            List of relevant Document objects
        """
        if not self.validate_query(query):
            return []
        
        try:
            # Get results with scores
            results_with_scores = self.retrieve_with_scores(query, k, reference_time, **kwargs)
            
            # Return only documents (without scores)
            results = [doc for doc, _ in results_with_scores]
            
            logger.info(f"Retrieved {len(results)} documents using time-aware search")
            return results
            
        except Exception as e:
            logger.error(f"Error in time-aware retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           reference_time: Optional[datetime] = None,
                           **kwargs) -> List[tuple]:
        """
        Retrieve documents with scores using time-aware search.
        
        Args:
            query: Query text to search for
            k: Number of documents to retrieve
            reference_time: Reference time for temporal scoring
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
            
            # Get vector similarity results
            vector_results = self.vector_store.similarity_search_with_score(
                query=processed_query,
                embeddings=self.embeddings,
                k=k * 2,  # Get more results for better time-aware scoring
                namespace=namespace
            )
            
            # Calculate time-aware scores
            time_aware_results = []
            for doc, vector_score in vector_results:
                # Calculate time score
                time_score = self.calculate_time_score(doc, reference_time)
                
                # Combine vector similarity and time score
                final_score = ((1 - self.recency_weight) * vector_score + 
                              self.recency_weight * time_score)
                
                time_aware_results.append((doc, final_score))
            
            # Sort by final score
            time_aware_results.sort(key=lambda x: x[1], reverse=True)
            
            # Remove duplicates
            seen_contents = set()
            unique_results = []
            
            for doc, score in time_aware_results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append((doc, score))
            
            # Return top k results
            final_results = unique_results[:k]
            
            logger.info(f"Retrieved {len(final_results)} documents using time-aware search")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in time-aware retrieval with scores: {str(e)}")
            return []
    
    def get_recent_documents(self, 
                           days: int = 30,
                           reference_time: Optional[datetime] = None,
                           **kwargs) -> List[Document]:
        """
        Get documents from the last N days.
        
        Args:
            days: Number of days to look back
            reference_time: Reference time (defaults to current time)
            **kwargs: Additional parameters
            
        Returns:
            List of recent documents
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        try:
            # Get all documents (this might be expensive for large collections)
            # In practice, you'd want to implement this more efficiently
            all_results = self.vector_store.similarity_search_with_score(
                query="",  # Empty query to get all documents
                embeddings=self.embeddings,
                k=1000,  # Large number to get many documents
                namespace=kwargs.get('namespace', self.namespace)
            )
            
            # Filter by time
            recent_docs = []
            cutoff_time = reference_time - timedelta(days=days)
            
            for doc, _ in all_results:
                doc_timestamp = self.extract_timestamp(doc)
                if doc_timestamp and doc_timestamp >= cutoff_time:
                    recent_docs.append(doc)
            
            logger.info(f"Found {len(recent_docs)} documents from the last {days} days")
            return recent_docs
            
        except Exception as e:
            logger.error(f"Error getting recent documents: {str(e)}")
            return []
    
    def get_documents_by_date_range(self, 
                                  start_date: datetime,
                                  end_date: datetime,
                                  **kwargs) -> List[Document]:
        """
        Get documents within a specific date range.
        
        Args:
            start_date: Start date for the range
            end_date: End date for the range
            **kwargs: Additional parameters
            
        Returns:
            List of documents in the date range
        """
        try:
            # Get all documents
            all_results = self.vector_store.similarity_search_with_score(
                query="",
                embeddings=self.embeddings,
                k=1000,
                namespace=kwargs.get('namespace', self.namespace)
            )
            
            # Filter by date range
            range_docs = []
            
            for doc, _ in all_results:
                doc_timestamp = self.extract_timestamp(doc)
                if doc_timestamp and start_date <= doc_timestamp <= end_date:
                    range_docs.append(doc)
            
            logger.info(f"Found {len(range_docs)} documents in date range {start_date} to {end_date}")
            return range_docs
            
        except Exception as e:
            logger.error(f"Error getting documents by date range: {str(e)}")
            return []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the time-aware retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        config = super().get_config()
        config.update({
            'time_decay_factor': self.time_decay_factor,
            'recency_weight': self.recency_weight,
            'namespace': self.namespace
        })
        return config
