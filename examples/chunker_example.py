"""
Example script demonstrating the use of different document chunkers.
"""

import logging
import sys
import os

# Add the parent directory to the path to import the document_prepration module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_prepration.chunkers import ChunkerFactory
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate different chunking strategies."""
    
    # Sample text to chunk
    sample_text = """
    This is a sample document for testing different chunking strategies.
    
    It contains multiple paragraphs with different content.
    
    The first paragraph introduces the topic of chunking documents.
    Chunking is an important step in document processing for RAG systems.
    
    The second paragraph discusses various chunking strategies.
    We can chunk by characters, tokens, sentences, or other criteria.
    
    The third paragraph explains why chunking is necessary.
    Large documents need to be broken down into smaller pieces for processing.
    This helps with memory management and improves retrieval performance.
    
    Finally, this paragraph concludes the sample document.
    Thank you for reading this example text.
    """
    
    # Sample metadata
    metadata = {
        'source': 'example.txt',
        'author': 'Example Author',
        'date': '2024-01-01'
    }
    
    # Create a LangChain Document
    document = Document(
        page_content=sample_text,
        metadata=metadata
    )
    
    # Get available chunker types
    available_chunkers = ChunkerFactory.get_available_chunkers()
    logger.info(f"Available chunker types: {available_chunkers}")
    
    # Test each chunker type
    for chunker_type in available_chunkers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {chunker_type} chunker")
        logger.info(f"{'='*50}")
        
        try:
            # Create chunker
            if chunker_type == 'sentence':
                # Sentence chunker requires sentence-transformers
                chunker = ChunkerFactory.create_chunker(
                    chunker_type, 
                    chunk_size=500, 
                    chunk_overlap=50
                )
            else:
                chunker = ChunkerFactory.create_chunker(
                    chunker_type, 
                    chunk_size=200, 
                    chunk_overlap=50
                )
            
            # Test chunking a single document
            logger.info("Testing chunk_documents method:")
            chunked_docs = chunker.chunk_documents([document])
            logger.info(f"Created {len(chunked_docs)} chunks:")
            for i, doc in enumerate(chunked_docs):
                logger.info(f"\nChunk {i+1}:")
                logger.info(f"Content: {doc.page_content[:100]}...")
                logger.info(f"Metadata: {doc.metadata}")
            
            # Test chunking text directly
            logger.info("\nTesting chunk_text method:")
            chunked_texts = chunker.chunk_text(sample_text, metadata)
            logger.info(f"Created {len(chunked_texts)} text chunks:")
            for i, doc in enumerate(chunked_texts):
                logger.info(f"\nText Chunk {i+1}:")
                logger.info(f"Content: {doc.page_content[:100]}...")
                logger.info(f"Metadata: {doc.metadata}")
                
        except Exception as e:
            logger.error(f"Error with {chunker_type} chunker: {str(e)}")
            continue


if __name__ == "__main__":
    main()
