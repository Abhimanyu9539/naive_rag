"""
Contextual retriever that uses conversation history to improve retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from ..base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class ContextualRetriever(BaseRetriever):
    """
    Contextual retriever that uses conversation history and context to improve retrieval.
    This strategy is useful for multi-turn conversations where previous context matters.
    """
    
    def __init__(self, 
                 embeddings,
                 vector_store,
                 context_window: int = 3,
                 context_weight: float = 0.3,
                 namespace: Optional[str] = None,
                 **kwargs):
        """
        Initialize the contextual retriever.
        
        Args:
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            context_window: Number of previous messages to consider as context
            context_weight: Weight for context in query expansion (0.0 to 1.0)
            namespace: Optional namespace for the vector store
            **kwargs: Additional configuration parameters
        """
        super().__init__(embeddings, vector_store, **kwargs)
        self.context_window = context_window
        self.context_weight = context_weight
        self.namespace = namespace
        self.conversation_history = []
        
        logger.info(f"Initialized ContextualRetriever with context_window={context_window}, context_weight={context_weight}")
    
    def add_to_history(self, message: str, is_user: bool = True):
        """
        Add a message to the conversation history.
        
        Args:
            message: Message text
            is_user: Whether this is a user message (True) or system response (False)
        """
        self.conversation_history.append({
            'message': message,
            'is_user': is_user,
            'timestamp': self._get_timestamp()
        })
        
        # Keep only the last context_window messages
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-self.context_window * 2:]
        
        logger.debug(f"Added message to history. Total messages: {len(self.conversation_history)}")
    
    def _get_timestamp(self) -> float:
        """
        Get current timestamp.
        
        Returns:
            Current timestamp
        """
        import time
        return time.time()
    
    def get_relevant_context(self, current_query: str) -> str:
        """
        Extract relevant context from conversation history.
        
        Args:
            current_query: Current query
            
        Returns:
            Relevant context string
        """
        if not self.conversation_history:
            return ""
        
        # Get recent user messages
        recent_user_messages = []
        for msg in self.conversation_history[-self.context_window * 2:]:
            if msg['is_user']:
                recent_user_messages.append(msg['message'])
        
        if not recent_user_messages:
            return ""
        
        # Combine recent messages
        context = " ".join(recent_user_messages[-self.context_window:])
        
        # Extract key terms from current query to find relevant context
        current_keywords = self.extract_keywords(current_query)
        
        # Find context messages that share keywords with current query
        relevant_context_parts = []
        for msg in recent_user_messages:
            msg_keywords = self.extract_keywords(msg)
            if any(keyword in msg_keywords for keyword in current_keywords):
                relevant_context_parts.append(msg)
        
        if relevant_context_parts:
            context = " ".join(relevant_context_parts)
        
        return context
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        import re
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def expand_query_with_context(self, query: str) -> str:
        """
        Expand the query with relevant context.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query with context
        """
        context = self.get_relevant_context(query)
        
        if not context:
            return query
        
        # Combine query with context
        expanded_query = f"{query} [context: {context}]"
        
        logger.debug(f"Expanded query: '{query}' -> '{expanded_query}'")
        return expanded_query
    
    def retrieve(self, 
                query: str, 
                k: int = 4,
                **kwargs) -> List[Document]:
        """
        Retrieve documents using contextual information.
        
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
            # Add current query to history
            self.add_to_history(query, is_user=True)
            
            # Get results with scores
            results_with_scores = self.retrieve_with_scores(query, k, **kwargs)
            
            # Return only documents (without scores)
            results = [doc for doc, _ in results_with_scores]
            
            logger.info(f"Retrieved {len(results)} documents using contextual search")
            return results
            
        except Exception as e:
            logger.error(f"Error in contextual retrieval: {str(e)}")
            return []
    
    def retrieve_with_scores(self, 
                           query: str, 
                           k: int = 4,
                           **kwargs) -> List[tuple]:
        """
        Retrieve documents with scores using contextual information.
        
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
            # Expand query with context
            expanded_query = self.expand_query_with_context(query)
            
            # Get namespace from kwargs or use default
            namespace = kwargs.get('namespace', self.namespace)
            
            # Perform search with expanded query
            results = self.vector_store.similarity_search_with_score(
                query=expanded_query,
                embeddings=self.embeddings,
                k=k * 2,  # Get more results for better selection
                namespace=namespace
            )
            
            # If we have context, also search with original query and combine
            if self.get_relevant_context(query):
                original_results = self.vector_store.similarity_search_with_score(
                    query=query,
                    embeddings=self.embeddings,
                    k=k,
                    namespace=namespace
                )
                
                # Combine and re-rank results
                all_results = []
                
                # Add expanded query results with context weight
                for doc, score in results:
                    adjusted_score = score * (1 - self.context_weight)
                    all_results.append((doc, adjusted_score))
                
                # Add original query results with original weight
                for doc, score in original_results:
                    adjusted_score = score * self.context_weight
                    all_results.append((doc, adjusted_score))
                
                # Remove duplicates and sort
                seen_contents = set()
                unique_results = []
                
                for doc, score in all_results:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        unique_results.append((doc, score))
                
                # Sort by score and return top k
                unique_results.sort(key=lambda x: x[1], reverse=True)
                results = unique_results[:k]
            else:
                # No context, just return original results
                results = results[:k]
            
            logger.info(f"Retrieved {len(results)} documents using contextual search")
            return results
            
        except Exception as e:
            logger.error(f"Error in contextual retrieval with scores: {str(e)}")
            return []
    
    def clear_history(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []
        logger.info("Cleared conversation history")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_history.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the contextual retriever.
        
        Returns:
            Dictionary containing the retriever configuration
        """
        config = super().get_config()
        config.update({
            'context_window': self.context_window,
            'context_weight': self.context_weight,
            'namespace': self.namespace,
            'history_length': len(self.conversation_history)
        })
        return config
