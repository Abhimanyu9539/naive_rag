import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Base class for all document embedders."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", **kwargs):
        """
        Initialize the base embedder.
        
        Args:
            model_name: Name of the embedding model to use
            **kwargs: Additional arguments for the embedder
        """
        self.model_name = model_name
        self.embedder = self._create_embedder(**kwargs)
        
        logger.info(f"Initialized {self.__class__.__name__} with model={model_name}")
    
    @abstractmethod
    def _create_embedder(self, **kwargs) -> Embeddings:
        """Create and return the specific embedding implementation."""
        pass
    
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
            
            embedding = self.embedder.embed_query(text)
            
            logger.debug(f"Successfully embedded text with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
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
            logger.debug(f"Embedding {len(texts)} texts")
            
            embeddings = self.embedder.embed_documents(texts)
            
            logger.info(f"Successfully embedded {len(texts)} texts with {len(embeddings[0]) if embeddings else 0} dimensions each")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
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
            logger.debug(f"Embedding {len(documents)} documents")
            
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Get embeddings
            embeddings = self.embed_texts(texts)
            
            # Combine with metadata
            embedded_docs = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                embedded_doc = {
                    'id': f"doc_{i}",
                    'text': doc.page_content,
                    'embedding': embedding,
                    'metadata': doc.metadata.copy() if doc.metadata else {}
                }
                embedded_docs.append(embedded_doc)
            
            logger.info(f"Successfully embedded {len(embedded_docs)} documents")
            return embedded_docs
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Number of dimensions in the embedding vectors
        """
        try:
            # Create a test embedding to determine dimensions
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimensions: {str(e)}")
            raise
