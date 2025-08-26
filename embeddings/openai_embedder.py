"""
This module contains the OpenAIEmbedder class, which uses OpenAI's embedding models.
"""

import os
import logging
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedder using LangChain's OpenAIEmbeddings.
    """
    
    def __init__(self, 
                 model_name: str = "text-embedding-ada-002",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model_name: OpenAI embedding model name (default: text-embedding-ada-002)
            api_key: OpenAI API key (if not provided, will use environment variable)
            **kwargs: Additional arguments for OpenAIEmbeddings
        """
        # Set API key from parameter or environment
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            logger.warning("No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
        
        super().__init__(model_name=model_name, **kwargs)
        
        logger.info(f"OpenAI embedder initialized with model: {model_name}")
    
    def _create_embedder(self, **kwargs) -> OpenAIEmbeddings:
        """
        Create and return OpenAIEmbeddings instance.
        
        Args:
            **kwargs: Additional arguments for OpenAIEmbeddings
            
        Returns:
            OpenAIEmbeddings instance
        """
        try:
            embedder = OpenAIEmbeddings(
                model=self.model_name,
                **kwargs
            )
            logger.debug(f"Created OpenAIEmbeddings with model: {self.model_name}")
            return embedder
            
        except Exception as e:
            logger.error(f"Error creating OpenAI embedder: {str(e)}")
            raise
    
    def embed_text_with_metadata(self, text: str, metadata: Optional[dict] = None) -> dict:
        """
        Embed a single text with metadata.
        
        Args:
            text: The text to embed
            metadata: Optional metadata to include
            
        Returns:
            Dictionary containing embedding and metadata
        """
        try:
            embedding = self.embed_text(text)
            
            result = {
                'text': text,
                'embedding': embedding,
                'model': self.model_name,
                'dimensions': len(embedding)
            }
            
            if metadata:
                result['metadata'] = metadata
            
            return result
            
        except Exception as e:
            logger.error(f"Error embedding text with metadata: {str(e)}")
            raise
    
    def batch_embed_with_metadata(self, texts: list, metadata_list: Optional[list] = None) -> list:
        """
        Embed multiple texts with corresponding metadata.
        
        Args:
            texts: List of texts to embed
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        try:
            embeddings = self.embed_texts(texts)
            
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result = {
                    'id': f"batch_{i}",
                    'text': text,
                    'embedding': embedding,
                    'model': self.model_name,
                    'dimensions': len(embedding)
                }
                
                if metadata_list and i < len(metadata_list):
                    result['metadata'] = metadata_list[i]
                
                results.append(result)
            
            logger.info(f"Successfully batch embedded {len(results)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Error batch embedding with metadata: {str(e)}")
            raise
