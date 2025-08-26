"""
This module contains the EmbedderFactory class for creating different types of embedders.
"""

import logging
from typing import Dict, Type, Optional
from .base_embedder import BaseEmbedder
from .openai_embedder import OpenAIEmbedder

logger = logging.getLogger(__name__)


class EmbedderFactory:
    """
    Factory class for creating different types of embedders.
    """
    
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        'openai': OpenAIEmbedder,
    }
    
    @classmethod
    def register_embedder(cls, name: str, embedder_class: Type[BaseEmbedder]) -> None:
        """
        Register a new embedder type.
        
        Args:
            name: Name of the embedder type
            embedder_class: Class of the embedder to register
        """
        cls._embedders[name] = embedder_class
        logger.info(f"Registered embedder: {name}")
    
    @classmethod
    def get_embedder(cls, embedder_type: str, **kwargs) -> BaseEmbedder:
        """
        Create and return an embedder instance.
        
        Args:
            embedder_type: Type of embedder to create
            **kwargs: Arguments to pass to the embedder constructor
            
        Returns:
            BaseEmbedder instance
            
        Raises:
            ValueError: If embedder type is not supported
        """
        if embedder_type not in cls._embedders:
            available_embedders = list(cls._embedders.keys())
            raise ValueError(f"Unsupported embedder type: {embedder_type}. Available types: {available_embedders}")
        
        embedder_class = cls._embedders[embedder_type]
        logger.info(f"Creating embedder of type: {embedder_type}")
        
        return embedder_class(**kwargs)
    
    @classmethod
    def list_available_embedders(cls) -> list:
        """
        Get a list of available embedder types.
        
        Returns:
            List of available embedder type names
        """
        return list(cls._embedders.keys())
    
    @classmethod
    def get_embedder_info(cls, embedder_type: str) -> Optional[Dict]:
        """
        Get information about a specific embedder type.
        
        Args:
            embedder_type: Type of embedder to get info for
            
        Returns:
            Dictionary containing embedder information or None if not found
        """
        if embedder_type not in cls._embedders:
            return None
        
        embedder_class = cls._embedders[embedder_type]
        
        info = {
            'name': embedder_type,
            'class': embedder_class.__name__,
            'module': embedder_class.__module__,
            'doc': embedder_class.__doc__
        }
        
        return info
