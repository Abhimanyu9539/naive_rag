"""
This module contains the PineconeStore class for storing and retrieving embeddings in Pinecone.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
import pinecone

logger = logging.getLogger(__name__)


class PineconeStore:
    """
    Pinecone vector store for storing and retrieving embeddings.
    """
    
    def __init__(self, 
                 index_name: str,
                 api_key: Optional[str] = None,
                 environment: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Pinecone store.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key (if not provided, will use environment variable)
            environment: Pinecone environment (if not provided, will use environment variable)
            **kwargs: Additional arguments for Pinecone initialization
        """
        # Set API key and environment from parameters or environment
        if api_key:
            os.environ["PINECONE_API_KEY"] = api_key
        if environment:
            os.environ["PINECONE_ENVIRONMENT"] = environment
            
        if not os.getenv("PINECONE_API_KEY"):
            logger.warning("No Pinecone API key provided. Please set PINECONE_API_KEY environment variable.")
        if not os.getenv("PINECONE_ENVIRONMENT"):
            logger.warning("No Pinecone environment provided. Please set PINECONE_ENVIRONMENT environment variable.")
        
        self.index_name = index_name
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        
        # Initialize Pinecone
        try:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=self.environment
            )
            logger.info(f"Pinecone initialized with index: {index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """
        Create a new Pinecone index.
        
        Args:
            dimension: Dimension of the embedding vectors
            metric: Distance metric to use (cosine, euclidean, dotproduct)
            
        Returns:
            True if index was created successfully
        """
        try:
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric
                )
                logger.info(f"Created Pinecone index: {self.index_name} with dimension {dimension}")
                return True
            else:
                logger.info(f"Index {self.index_name} already exists")
                return False
                
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {str(e)}")
            raise
    
    def get_vector_store(self, embeddings: Embeddings) -> PineconeVectorStore:
        """
        Get a PineconeVectorStore instance.
        
        Args:
            embeddings: Embeddings instance to use
            
        Returns:
            PineconeVectorStore instance
        """
        try:
            vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=embeddings
            )
            logger.debug(f"Created PineconeVectorStore for index: {self.index_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating PineconeVectorStore: {str(e)}")
            raise
    
    def add_documents(self, 
                     documents: List[Document], 
                     embeddings: Embeddings,
                     namespace: Optional[str] = None) -> bool:
        """
        Add documents to the Pinecone index.
        
        Args:
            documents: List of LangChain Document objects
            embeddings: Embeddings instance to use
            namespace: Optional namespace for the documents
            
        Returns:
            True if documents were added successfully
        """
        try:
            vector_store = self.get_vector_store(embeddings)
            
            # Add documents to the vector store
            vector_store.add_documents(documents, namespace=namespace)
            
            logger.info(f"Successfully added {len(documents)} documents to Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            raise
    
    def add_texts(self, 
                  texts: List[str], 
                  embeddings: Embeddings,
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  namespace: Optional[str] = None) -> bool:
        """
        Add texts to the Pinecone index.
        
        Args:
            texts: List of text strings
            embeddings: Embeddings instance to use
            metadatas: Optional list of metadata dictionaries
            namespace: Optional namespace for the texts
            
        Returns:
            True if texts were added successfully
        """
        try:
            vector_store = self.get_vector_store(embeddings)
            
            # Add texts to the vector store
            vector_store.add_texts(texts, metadatas=metadatas, namespace=namespace)
            
            logger.info(f"Successfully added {len(texts)} texts to Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding texts to Pinecone: {str(e)}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         embeddings: Embeddings,
                         k: int = 4,
                         namespace: Optional[str] = None) -> List[Document]:
        """
        Perform similarity search in the Pinecone index.
        
        Args:
            query: Query text to search for
            embeddings: Embeddings instance to use
            k: Number of similar documents to return
            namespace: Optional namespace to search in
            
        Returns:
            List of similar Document objects
        """
        try:
            vector_store = self.get_vector_store(embeddings)
            
            # Perform similarity search
            results = vector_store.similarity_search(query, k=k, namespace=namespace)
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   embeddings: Embeddings,
                                   k: int = 4,
                                   namespace: Optional[str] = None) -> List[tuple]:
        """
        Perform similarity search with scores in the Pinecone index.
        
        Args:
            query: Query text to search for
            embeddings: Embeddings instance to use
            k: Number of similar documents to return
            namespace: Optional namespace to search in
            
        Returns:
            List of tuples containing (Document, score)
        """
        try:
            vector_store = self.get_vector_store(embeddings)
            
            # Perform similarity search with scores
            results = vector_store.similarity_search_with_score(query, k=k, namespace=namespace)
            
            logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            raise
    
    def delete_index(self) -> bool:
        """
        Delete the Pinecone index.
        
        Returns:
            True if index was deleted successfully
        """
        try:
            if self.index_name in pinecone.list_indexes():
                pinecone.delete_index(self.index_name)
                logger.info(f"Deleted Pinecone index: {self.index_name}")
                return True
            else:
                logger.warning(f"Index {self.index_name} does not exist")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting Pinecone index: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            if self.index_name in pinecone.list_indexes():
                index = pinecone.Index(self.index_name)
                stats = index.describe_index_stats()
                logger.info(f"Retrieved stats for index: {self.index_name}")
                return stats
            else:
                logger.warning(f"Index {self.index_name} does not exist")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise
