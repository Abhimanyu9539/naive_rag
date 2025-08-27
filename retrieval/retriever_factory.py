"""
Factory class for creating different types of retrievers.
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.embeddings import Embeddings

from .base_retriever import BaseRetriever
from .strategies.simple_retriever import SimpleRetriever
from .strategies.hybrid_retriever import HybridRetriever
from .strategies.multi_query_retriever import MultiQueryRetriever
from .strategies.contextual_retriever import ContextualRetriever
from .strategies.rerank_retriever import RerankRetriever
from .strategies.time_aware_retriever import TimeAwareRetriever
from .strategies.ensemble_retriever import EnsembleRetriever

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """
    Factory class for creating different types of retrievers.
    """
    
    # Registry of available retriever types
    RETRIEVER_REGISTRY = {
        'simple': SimpleRetriever,
        'hybrid': HybridRetriever,
        'multi_query': MultiQueryRetriever,
        'contextual': ContextualRetriever,
        'rerank': RerankRetriever,
        'time_aware': TimeAwareRetriever,
        'ensemble': EnsembleRetriever
    }
    
    @classmethod
    def create_retriever(cls,
                        retriever_type: str,
                        embeddings: Embeddings,
                        vector_store,
                        **kwargs) -> BaseRetriever:
        """
        Create a retriever instance based on the specified type.
        
        Args:
            retriever_type: Type of retriever to create
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseRetriever instance
            
        Raises:
            ValueError: If retriever_type is not supported
        """
        if retriever_type not in cls.RETRIEVER_REGISTRY:
            available_types = list(cls.RETRIEVER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported retriever type: {retriever_type}. "
                f"Available types: {available_types}"
            )
        
        retriever_class = cls.RETRIEVER_REGISTRY[retriever_type]
        
        try:
            retriever = retriever_class(embeddings, vector_store, **kwargs)
            logger.info(f"Created {retriever_type} retriever successfully")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating {retriever_type} retriever: {str(e)}")
            raise
    
    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available retriever types.
        
        Returns:
            List of available retriever type names
        """
        return list(cls.RETRIEVER_REGISTRY.keys())
    
    @classmethod
    def register_retriever(cls, 
                          name: str, 
                          retriever_class: type):
        """
        Register a new retriever type.
        
        Args:
            name: Name for the retriever type
            retriever_class: Class implementing BaseRetriever
        """
        if not issubclass(retriever_class, BaseRetriever):
            raise ValueError(f"Retriever class must inherit from BaseRetriever")
        
        cls.RETRIEVER_REGISTRY[name] = retriever_class
        logger.info(f"Registered new retriever type: {name}")
    
    @classmethod
    def get_retriever_info(cls, retriever_type: str) -> Dict[str, Any]:
        """
        Get information about a specific retriever type.
        
        Args:
            retriever_type: Type of retriever
            
        Returns:
            Dictionary containing retriever information
        """
        if retriever_type not in cls.RETRIEVER_REGISTRY:
            return {}
        
        retriever_class = cls.RETRIEVER_REGISTRY[retriever_type]
        
        return {
            'name': retriever_type,
            'class': retriever_class.__name__,
            'module': retriever_class.__module__,
            'docstring': retriever_class.__doc__ or "No documentation available"
        }
    
    @classmethod
    def create_ensemble_retriever(cls,
                                 retriever_configs: list,
                                 embeddings: Embeddings,
                                 vector_store,
                                 ensemble_strategy: str = "weighted") -> 'EnsembleRetriever':
        """
        Create an ensemble retriever combining multiple retrieval strategies.
        
        Args:
            retriever_configs: List of dictionaries with retriever configurations
            embeddings: Embeddings instance to use
            vector_store: Vector store instance
            ensemble_strategy: Strategy for combining results ('weighted', 'voting', 'rank_fusion')
            
        Returns:
            EnsembleRetriever instance
        """
        retrievers = []
        
        for config in retriever_configs:
            retriever_type = config.pop('type')
            weight = config.pop('weight', 1.0)
            
            retriever = cls.create_retriever(
                retriever_type=retriever_type,
                embeddings=embeddings,
                vector_store=vector_store,
                **config
            )
            
            retrievers.append((retriever, weight))
        
        from .strategies.ensemble_retriever import EnsembleRetriever
        return EnsembleRetriever(retrievers, ensemble_strategy=ensemble_strategy)
