"""
This module contains the RAGProcessor class for orchestrating the complete RAG pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from embeddings.document_processor import DocumentProcessor
from vector_stores.vector_store_processor import VectorStoreProcessor
from vector_stores.pinecone_store import PineconeStore
from generation import GeneratorFactory
from generation.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class RAGProcessor:
    """
    High-level processor for orchestrating the complete RAG pipeline.
    """
    
    def __init__(self, 
                 document_processor: DocumentProcessor,
                 vector_store_processor: VectorStoreProcessor,
                 generator: Optional[BaseGenerator] = None,
                 namespace: Optional[str] = None):
        """
        Initialize the RAG processor.
        
        Args:
            document_processor: Document processor for embedding operations
            vector_store_processor: Vector store processor for storage operations
            generator: Optional generator for response generation
            namespace: Optional namespace for storing embeddings
        """
        self.document_processor = document_processor
        self.vector_store_processor = vector_store_processor
        self.generator = generator
        self.namespace = namespace
        
        logger.info(f"Initialized RAGProcessor with namespace: {namespace}")
    
    @classmethod
    def create_with_factory(cls, 
                           embedder_type: str = "openai",
                           index_name: str = "default-index",
                           generator_type: Optional[str] = None,
                           llm: Optional[BaseLanguageModel] = None,
                           namespace: Optional[str] = None,
                           **kwargs):
        """
        Create a RAGProcessor using the factory pattern.
        
        Args:
            embedder_type: Type of embedder to create
            index_name: Name of the Pinecone index
            generator_type: Type of generator to create (optional)
            llm: Language model for generation (required if generator_type is specified)
            namespace: Optional namespace for storing embeddings
            **kwargs: Additional arguments for embedder, vector store, and generator
            
        Returns:
            RAGProcessor instance
        """
        # Create document processor
        document_processor = DocumentProcessor.create_with_factory(embedder_type, **kwargs)
        
        # Create vector store
        vector_store = PineconeStore(index_name=index_name, **kwargs)
        
        # Create vector store processor
        vector_store_processor = VectorStoreProcessor(vector_store)
        
        # Create generator if specified
        generator = None
        if generator_type and llm:
            generator = GeneratorFactory.create_generator(
                generator_type=generator_type,
                llm=llm,
                **kwargs
            )
        
        # Create RAG processor
        processor = cls(document_processor, vector_store_processor, generator, namespace)
        
        logger.info(f"Created RAGProcessor with embedder_type={embedder_type}, index_name={index_name}, generator_type={generator_type}")
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
    
    def query(self, 
             query: str, 
             k: int = 4,
             **kwargs) -> str:
        """
        Complete RAG pipeline: retrieve documents and generate response.
        
        Args:
            query: User's query
            k: Number of documents to retrieve
            **kwargs: Additional parameters for generation
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If no generator is configured
        """
        if not self.generator:
            raise ValueError("No generator configured. Please set up a generator first.")
        
        try:
            logger.info(f"Processing RAG query: {query[:50]}...")
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.search_similar(query, k)
            
            if not retrieved_docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Step 2: Generate response
            response = self.generator.generate(query, retrieved_docs, **kwargs)
            
            logger.info(f"Generated response for query: {query[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            raise
    
    def query_with_metadata(self, 
                          query: str, 
                          k: int = 4,
                          **kwargs) -> Dict[str, Any]:
        """
        Complete RAG pipeline with metadata: retrieve documents and generate response with details.
        
        Args:
            query: User's query
            k: Number of documents to retrieve
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary containing response and metadata
            
        Raises:
            ValueError: If no generator is configured
        """
        if not self.generator:
            raise ValueError("No generator configured. Please set up a generator first.")
        
        try:
            logger.info(f"Processing RAG query with metadata: {query[:50]}...")
            
            # Step 1: Retrieve relevant documents with scores
            retrieved_docs_with_scores = self.search_similar_with_scores(query, k)
            retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
            
            if not retrieved_docs:
                return {
                    'response': "I couldn't find any relevant information to answer your question.",
                    'query': query,
                    'retrieved_documents': [],
                    'generation_metadata': None
                }
            
            # Step 2: Generate response with metadata
            generation_result = self.generator.generate_with_metadata(query, retrieved_docs, **kwargs)
            
            # Step 3: Combine retrieval and generation metadata
            result = {
                'response': generation_result['response'],
                'query': query,
                'retrieved_documents': retrieved_docs_with_scores,
                'generation_metadata': generation_result,
                'retrieval_stats': {
                    'num_documents_retrieved': len(retrieved_docs),
                    'average_similarity_score': sum(score for _, score in retrieved_docs_with_scores) / len(retrieved_docs_with_scores) if retrieved_docs_with_scores else 0
                }
            }
            
            logger.info(f"Generated response with metadata for query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG query with metadata: {str(e)}")
            raise
    
    def set_generator(self, generator: BaseGenerator):
        """
        Set or update the generator for the RAG processor.
        
        Args:
            generator: Generator instance to use
        """
        self.generator = generator
        logger.info(f"Updated generator to: {generator.__class__.__name__}")
    
    def get_generator_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the current generator configuration.
        
        Returns:
            Generator configuration dictionary or None if no generator is set
        """
        if self.generator:
            return self.generator.get_config()
        return None
