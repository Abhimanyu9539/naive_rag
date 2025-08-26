"""
This module contains the DocumentProcessor class for processing documents through embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from .base_embedder import BaseEmbedder
from .embedder_factory import EmbedderFactory

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processor for embedding documents and preparing them for storage.
    """
    
    def __init__(self, embedder: BaseEmbedder):
        """
        Initialize the document processor.
        
        Args:
            embedder: Embedder instance to use
        """
        self.embedder = embedder
        
        logger.info(f"Initialized DocumentProcessor with embedder: {embedder.__class__.__name__}")
    
    @classmethod
    def create_with_factory(cls, embedder_type: str = "openai", **kwargs):
        """
        Create a DocumentProcessor using the factory pattern.
        
        Args:
            embedder_type: Type of embedder to create
            **kwargs: Additional arguments for embedder
            
        Returns:
            DocumentProcessor instance
        """
        # Create embedder
        embedder = EmbedderFactory.get_embedder(embedder_type, **kwargs)
        
        # Create processor
        processor = cls(embedder)
        
        logger.info(f"Created DocumentProcessor with embedder_type={embedder_type}")
        return processor
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Number of dimensions in the embedding vectors
        """
        try:
            dimensions = self.embedder.get_embedding_dimensions()
            logger.info(f"Embedding dimensions: {dimensions}")
            return dimensions
        except Exception as e:
            logger.error(f"Error getting embedding dimensions: {str(e)}")
            raise
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents and return with metadata.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        try:
            logger.info(f"Embedding {len(documents)} documents")
            
            embedded_docs = self.embedder.embed_documents(documents)
            
            logger.info(f"Successfully embedded {len(embedded_docs)} documents")
            return embedded_docs
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding lists
        """
        try:
            logger.info(f"Embedding {len(texts)} texts")
            
            embeddings = self.embedder.embed_texts(texts)
            
            logger.info(f"Successfully embedded {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        try:
            logger.debug(f"Embedding text of length {len(text)} characters")
            
            embedding = self.embedder.embed_text(text)
            
            logger.debug(f"Successfully embedded text with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise
    
    def prepare_documents_for_storage(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Prepare documents for storage by embedding them and formatting the data.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of dictionaries ready for vector store storage
        """
        try:
            logger.info(f"Preparing {len(documents)} documents for storage")
            
            # Embed the documents
            embedded_docs = self.embed_documents(documents)
            
            # Format for storage
            storage_ready_docs = []
            for i, embedded_doc in enumerate(embedded_docs):
                storage_doc = {
                    'id': embedded_doc.get('id', f"doc_{i}"),
                    'text': embedded_doc['text'],
                    'embedding': embedded_doc['embedding'],
                    'metadata': embedded_doc['metadata']
                }
                storage_ready_docs.append(storage_doc)
            
            logger.info(f"Successfully prepared {len(storage_ready_docs)} documents for storage")
            return storage_ready_docs
            
        except Exception as e:
            logger.error(f"Error preparing documents for storage: {str(e)}")
            raise
