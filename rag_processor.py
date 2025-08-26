"""
This module contains the RAGProcessor class for orchestrating the complete RAG pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from embeddings.document_processor import DocumentProcessor
from vector_stores.vector_store_processor import VectorStoreProcessor
from vector_stores.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)


class RAGProcessor:
    """
    High-level processor for orchestrating the complete RAG pipeline.
    """
    
    def __init__(self, 
                 document_processor: DocumentProcessor,
                 vector_store_processor: VectorStoreProcessor,
                 namespace: Optional[str] = None):
        """
        Initialize the RAG processor.
        
        Args:
            document_processor: Document processor for embedding operations
            vector_store_processor: Vector store processor for storage operations
            namespace: Optional namespace for storing embeddings
        """
        self.document_processor = document_processor
        self.vector_store_processor = vector_store_processor
        self.namespace = namespace
        
        logger.info(f"Initialized RAGProcessor with namespace: {namespace}")
    
    @classmethod
    def create_with_factory(cls, 
                           embedder_type: str = "openai",
                           index_name: str = "default-index",
                           namespace: Optional[str] = None,
                           **kwargs):
        """
        Create a RAGProcessor using the factory pattern.
        
        Args:
            embedder_type: Type of embedder to create
            index_name: Name of the Pinecone index
            namespace: Optional namespace for storing embeddings
            **kwargs: Additional arguments for embedder and vector store
            
        Returns:
            RAGProcessor instance
        """
        # Create document processor
        document_processor = DocumentProcessor.create_with_factory(embedder_type, **kwargs)
        
        # Create vector store
        vector_store = PineconeStore(index_name=index_name, **kwargs)
        
        # Create vector store processor
        vector_store_processor = VectorStoreProcessor(vector_store)
        
        # Create RAG processor
        processor = cls(document_processor, vector_store_processor, namespace)
        
        logger.info(f"Created RAGProcessor with embedder_type={embedder_type}, index_name={index_name}")
        return processor
    
    def setup_index(self, metric: str = "cosine") -> bool:
        """
        Set up the vector store index with the correct dimensions.
        
        Args:
            metric: Distance metric to use
            
        Returns:
            True if index was set up successfully
        """
        try:
            # Get embedding dimensions from document processor
            dimension = self.document_processor.get_embedding_dimensions()
            
            # Set up index using vector store processor
            success = self.vector_store_processor.setup_index(dimension, metric)
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting up index: {str(e)}")
            raise
    
    def process_documents(self, documents: List[Document]) -> bool:
        """
        Process documents by embedding them and storing in the vector database.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            True if documents were processed successfully
        """
        try:
            logger.info(f"Processing {len(documents)} documents through RAG pipeline")
            
            # Store documents using vector store processor
            success = self.vector_store_processor.store_documents(
                documents=documents,
                embeddings=self.document_processor.embedder.embedder,
                namespace=self.namespace
            )
            
            if success:
                logger.info(f"Successfully processed and stored {len(documents)} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def process_texts(self, 
                     texts: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Process texts by embedding them and storing in the vector database.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            True if texts were processed successfully
        """
        try:
            logger.info(f"Processing {len(texts)} texts through RAG pipeline")
            
            # Store texts using vector store processor
            success = self.vector_store_processor.store_texts(
                texts=texts,
                embeddings=self.document_processor.embedder.embedder,
                metadatas=metadatas,
                namespace=self.namespace
            )
            
            if success:
                logger.info(f"Successfully processed and stored {len(texts)} texts")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing texts: {str(e)}")
            raise
    
    def search_similar(self, 
                      query: str, 
                      k: int = 4) -> List[Document]:
        """
        Search for similar documents in the vector database.
        
        Args:
            query: Query text to search for
            k: Number of similar documents to return
            
        Returns:
            List of similar Document objects
        """
        try:
            logger.info(f"Searching for documents similar to: {query[:50]}...")
            
            results = self.vector_store_processor.search_similar(
                query=query,
                embeddings=self.document_processor.embedder.embedder,
                k=k,
                namespace=self.namespace
            )
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar documents: {str(e)}")
            raise
    
    def search_similar_with_scores(self, 
                                 query: str, 
                                 k: int = 4) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Query text to search for
            k: Number of similar documents to return
            
        Returns:
            List of tuples containing (Document, score)
        """
        try:
            logger.info(f"Searching for documents similar to: {query[:50]}... (with scores)")
            
            results = self.vector_store_processor.search_similar_with_scores(
                query=query,
                embeddings=self.document_processor.embedder.embedder,
                k=k,
                namespace=self.namespace
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
            stats = self.vector_store_processor.get_index_stats()
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
            success = self.vector_store_processor.delete_index()
            
            if success:
                logger.info("Successfully deleted vector database index")
            else:
                logger.warning("Index does not exist or could not be deleted")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query text using the document processor.
        
        Args:
            query: Query text to embed
            
        Returns:
            List of embedding values
        """
        try:
            embedding = self.document_processor.embed_text(query)
            logger.debug(f"Embedded query with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
