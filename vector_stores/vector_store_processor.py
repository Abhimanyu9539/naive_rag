"""
This module contains the VectorStoreProcessor class for managing vector store operations.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from .pinecone_store import PineconeStore

logger = logging.getLogger(__name__)


class VectorStoreProcessor:
    """
    Processor for managing vector store operations.
    """
    
    def __init__(self, vector_store: PineconeStore):
        """
        Initialize the vector store processor.
        
        Args:
            vector_store: Vector store instance to use
        """
        self.vector_store = vector_store
        
        logger.info(f"Initialized VectorStoreProcessor with vector store: {vector_store.__class__.__name__}")
    
    def setup_index(self, dimension: int, metric: str = "cosine") -> bool:
        """
        Set up the vector store index with the correct dimensions.
        
        Args:
            dimension: Dimension of embeddings
            metric: Distance metric to use
            
        Returns:
            True if index was set up successfully
        """
        try:
            # Create index
            success = self.vector_store.create_index(dimension, metric)
            
            if success:
                logger.info(f"Successfully set up vector store index with dimension {dimension}")
            else:
                logger.info("Vector store index already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up index: {str(e)}")
            raise
    
    def store_documents(self, 
                       documents: List[Document], 
                       embeddings: Embeddings,
                       namespace: Optional[str] = None) -> bool:
        """
        Store documents in the vector database.
        
        Args:
            documents: List of LangChain Document objects
            embeddings: Embeddings instance to use
            namespace: Optional namespace for storing embeddings
            
        Returns:
            True if documents were stored successfully
        """
        try:
            logger.info(f"Storing {len(documents)} documents in vector store")
            
            # Add documents to vector store
            success = self.vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                namespace=namespace
            )
            
            if success:
                logger.info(f"Successfully stored {len(documents)} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise
    
    def store_texts(self, 
                   texts: List[str], 
                   embeddings: Embeddings,
                   metadatas: Optional[List[Dict[str, Any]]] = None,
                   namespace: Optional[str] = None) -> bool:
        """
        Store texts in the vector database.
        
        Args:
            texts: List of text strings
            embeddings: Embeddings instance to use
            metadatas: Optional list of metadata dictionaries
            namespace: Optional namespace for storing embeddings
            
        Returns:
            True if texts were stored successfully
        """
        try:
            logger.info(f"Storing {len(texts)} texts in vector store")
            
            # Add texts to vector store
            success = self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                namespace=namespace
            )
            
            if success:
                logger.info(f"Successfully stored {len(texts)} texts")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing texts: {str(e)}")
            raise
    
    def search_similar(self, 
                      query: str, 
                      embeddings: Embeddings,
                      k: int = 4,
                      namespace: Optional[str] = None) -> List[Document]:
        """
        Search for similar documents in the vector database.
        
        Args:
            query: Query text to search for
            embeddings: Embeddings instance to use
            k: Number of similar documents to return
            namespace: Optional namespace to search in
            
        Returns:
            List of similar Document objects
        """
        try:
            logger.info(f"Searching for documents similar to: {query[:50]}...")
            
            results = self.vector_store.similarity_search(
                query=query,
                embeddings=embeddings,
                k=k,
                namespace=namespace
            )
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar documents: {str(e)}")
            raise
    
    def search_similar_with_scores(self, 
                                 query: str, 
                                 embeddings: Embeddings,
                                 k: int = 4,
                                 namespace: Optional[str] = None) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text to search for
            embeddings: Embeddings instance to use
            k: Number of similar documents to return
            namespace: Optional namespace to search in
            
        Returns:
            List of tuples containing (Document, score)
        """
        try:
            logger.info(f"Searching for documents similar to: {query[:50]}... (with scores)")
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                embeddings=embeddings,
                k=k,
                namespace=namespace
            )
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar documents with scores: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database index.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = self.vector_store.get_index_stats()
            logger.info("Retrieved index statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise
    
    def delete_index(self) -> bool:
        """
        Delete the vector database index.
        
        Returns:
            True if index was deleted successfully
        """
        try:
            success = self.vector_store.delete_index()
            
            if success:
                logger.info("Successfully deleted vector database index")
            else:
                logger.warning("Index does not exist or could not be deleted")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            raise
