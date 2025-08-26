"""
Example demonstrating the separated embedding and vector store components.

This script shows how to use the new modular structure where:
- Embedding operations are handled by DocumentProcessor in the embeddings folder
- Vector store operations are handled by VectorStoreProcessor in the vector_stores folder
- High-level orchestration is handled by RAGProcessor
"""

import os
import logging
from dotenv import load_dotenv

# Import document preparation components
from document_prepration.loaders.txt_loader import TXTLoader
from document_prepration.chunkers.recursive_character_chunker import RecursiveCharacterChunker

# Import separated components
from embeddings import DocumentProcessor
from vector_stores import VectorStoreProcessor, PineconeStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_separated_components():
    """Demonstrate using embedding and vector store components separately."""
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return
    
    try:
        # Step 1: Create sample data
        logger.info("Step 1: Creating sample data...")
        sample_text = """
        Artificial Intelligence (AI) is a broad field of computer science that aims to create systems capable of performing tasks that typically require human intelligence.
        Machine learning is a subset of AI that focuses on algorithms that can learn from and make predictions on data.
        Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns.
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language.
        Computer vision is another important area of AI that enables computers to interpret and understand visual information.
        """
        
        os.makedirs("data/raw", exist_ok=True)
        file_path = "data/raw/ai_overview.txt"
        
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(sample_text.strip())
            logger.info(f"Created sample file: {file_path}")
        
        # Step 2: Load and chunk documents
        logger.info("Step 2: Loading and chunking documents...")
        loader = TXTLoader(file_path)
        documents = loader.load_with_metadata({
            'source_file': 'ai_overview.txt',
            'file_type': 'text',
            'pipeline': 'separated_components_example'
        })
        
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
        chunked_documents = chunker.chunk_documents(documents)
        
        logger.info(f"Created {len(chunked_documents)} chunks")
        
        # Step 3: Create separate processors
        logger.info("Step 3: Creating separate processors...")
        
        # Create document processor for embedding operations
        document_processor = DocumentProcessor.create_with_factory("openai")
        
        # Create vector store and processor for storage operations
        vector_store = PineconeStore(index_name="separated-components-index")
        vector_store_processor = VectorStoreProcessor(vector_store)
        
        # Step 4: Demonstrate embedding operations
        logger.info("Step 4: Demonstrating embedding operations...")
        
        # Get embedding dimensions
        dimensions = document_processor.get_embedding_dimensions()
        logger.info(f"Embedding dimensions: {dimensions}")
        
        # Embed a single text
        test_text = "What is artificial intelligence?"
        embedding = document_processor.embed_text(test_text)
        logger.info(f"Embedded text with {len(embedding)} dimensions")
        
        # Embed multiple texts
        texts = [doc.page_content for doc in chunked_documents[:3]]
        embeddings = document_processor.embed_texts(texts)
        logger.info(f"Embedded {len(embeddings)} texts")
        
        # Prepare documents for storage
        storage_ready_docs = document_processor.prepare_documents_for_storage(chunked_documents[:3])
        logger.info(f"Prepared {len(storage_ready_docs)} documents for storage")
        
        # Step 5: Demonstrate vector store operations
        logger.info("Step 5: Demonstrating vector store operations...")
        
        # Set up index
        success = vector_store_processor.setup_index(dimensions)
        logger.info(f"Index setup: {'Success' if success else 'Already exists'}")
        
        # Store documents
        success = vector_store_processor.store_documents(
            documents=chunked_documents,
            embeddings=document_processor.embedder.embedder,
            namespace="separated-namespace"
        )
        logger.info(f"Document storage: {'Success' if success else 'Failed'}")
        
        # Step 6: Demonstrate search operations
        logger.info("Step 6: Demonstrating search operations...")
        
        # Search for similar documents
        query = "What is machine learning?"
        similar_docs = vector_store_processor.search_similar(
            query=query,
            embeddings=document_processor.embedder.embedder,
            k=3,
            namespace="separated-namespace"
        )
        
        logger.info(f"Found {len(similar_docs)} similar documents for query: '{query}'")
        for i, doc in enumerate(similar_docs, 1):
            logger.info(f"  Result {i}: {doc.page_content[:80]}...")
        
        # Search with scores
        similar_docs_with_scores = vector_store_processor.search_similar_with_scores(
            query=query,
            embeddings=document_processor.embedder.embedder,
            k=3,
            namespace="separated-namespace"
        )
        
        logger.info(f"Found {len(similar_docs_with_scores)} similar documents with scores:")
        for i, (doc, score) in enumerate(similar_docs_with_scores, 1):
            logger.info(f"  Result {i} (Score: {score:.4f}): {doc.page_content[:60]}...")
        
        # Step 7: Get index statistics
        logger.info("Step 7: Getting index statistics...")
        stats = vector_store_processor.get_index_stats()
        logger.info(f"Index statistics: {stats}")
        
        logger.info("\nSeparated components example completed successfully!")
        logger.info("This demonstrates the new modular structure where embedding and vector store operations are properly separated.")
        
    except Exception as e:
        logger.error(f"Error in separated components example: {str(e)}")
        raise


if __name__ == "__main__":
    demonstrate_separated_components()
