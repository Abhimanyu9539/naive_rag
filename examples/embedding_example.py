"""
Example script demonstrating the embedding system.

This script shows how to:
1. Load documents using the document preparation system
2. Chunk documents using the chunking system  
3. Embed chunks using OpenAI embeddings
4. Store embeddings in Pinecone
5. Perform similarity search
"""

import os
import logging
from dotenv import load_dotenv

# Import document preparation components
from document_prepration.loaders.txt_loader import TXTLoader
from document_prepration.chunkers.recursive_character_chunker import RecursiveCharacterChunker

# Import embedding components
from embeddings import EmbedderFactory
from rag_processor import RAGProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating the embedding pipeline."""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        logger.error("PINECONE_API_KEY environment variable is required")
        return
    
    if not os.getenv("PINECONE_ENVIRONMENT"):
        logger.error("PINECONE_ENVIRONMENT environment variable is required")
        return
    
    try:
        # Step 1: Load documents
        logger.info("Step 1: Loading documents...")
        file_path = "data/raw/attention.txt"  # Adjust path as needed
        
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found. Creating sample text...")
            # Create sample text if file doesn't exist
            sample_text = """
            Attention is a fundamental concept in machine learning and artificial intelligence.
            It allows models to focus on specific parts of input data when making predictions.
            The attention mechanism was first introduced in the context of neural machine translation.
            Since then, it has become a core component of transformer architectures.
            Attention mechanisms enable models to process sequential data more effectively.
            They help capture long-range dependencies in text and other sequential data.
            """
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(sample_text)
        
        loader = TXTLoader(file_path)
        documents = loader.load_with_metadata({
            'source': 'example',
            'file_type': 'text',
            'processed_by': 'embedding_example'
        })
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Step 2: Chunk documents
        logger.info("Step 2: Chunking documents...")
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
        chunked_documents = chunker.chunk_documents(documents)
        
        logger.info(f"Created {len(chunked_documents)} chunks")
        
        # Step 3: Create RAG processor
        logger.info("Step 3: Setting up RAG processor...")
        processor = RAGProcessor.create_with_factory(
            embedder_type="openai",
            index_name="example-index",
            namespace="example-namespace"
        )
        
        # Step 4: Set up Pinecone index
        logger.info("Step 4: Setting up Pinecone index...")
        processor.setup_index()
        
        # Step 5: Process and store embeddings
        logger.info("Step 5: Processing and storing embeddings...")
        success = processor.process_documents(chunked_documents)
        
        if success:
            logger.info("Successfully processed and stored embeddings")
        else:
            logger.error("Failed to process embeddings")
            return
        
        # Step 6: Get index statistics
        logger.info("Step 6: Getting index statistics...")
        stats = processor.get_index_stats()
        logger.info(f"Index statistics: {stats}")
        
        # Step 7: Perform similarity search
        logger.info("Step 7: Performing similarity search...")
        query = "What is attention in machine learning?"
        similar_docs = processor.search_similar(query, k=3)
        
        logger.info(f"Found {len(similar_docs)} similar documents for query: '{query}'")
        
        for i, doc in enumerate(similar_docs, 1):
            logger.info(f"Document {i}:")
            logger.info(f"  Content: {doc.page_content[:100]}...")
            logger.info(f"  Metadata: {doc.metadata}")
            logger.info("")
        
        # Step 8: Perform similarity search with scores
        logger.info("Step 8: Performing similarity search with scores...")
        similar_docs_with_scores = processor.search_similar_with_scores(query, k=3)
        
        logger.info(f"Found {len(similar_docs_with_scores)} similar documents with scores:")
        
        for i, (doc, score) in enumerate(similar_docs_with_scores, 1):
            logger.info(f"Document {i} (Score: {score:.4f}):")
            logger.info(f"  Content: {doc.page_content[:100]}...")
            logger.info("")
        
        logger.info("Embedding example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in embedding example: {str(e)}")
        raise


if __name__ == "__main__":
    main()
