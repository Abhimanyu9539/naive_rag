"""
Example showing integration between document loaders and chunkers.
"""

import logging
import sys
import os

# Add the parent directory to the path to import the document_prepration module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_prepration.chunkers import ChunkerFactory
from document_prepration.loaders import PDFLoader, TXTLoader
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate integration between loaders and chunkers."""
    
    # Create a chunker
    chunker = ChunkerFactory.create_chunker(
        'recursive_character',
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Example 1: Load and chunk a single pdf file
    logger.info("Example 1: Loading and chunking a pdf file")
    logger.info("=" * 60)
    
    try:
        # Load a text file
        pdf_loader = PDFLoader(r"E:\Gen AI\RAG_Projects\naive_rag\data\raw\attention.pdf")
        documents = pdf_loader.load()
        
        if documents:
            logger.info(f"Loaded {len(documents)} document(s) from pdf file")
            
            # Chunk the documents
            chunked_docs = chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunked_docs)} chunks")
            
            # Display first few chunks
            for i, doc in enumerate(chunked_docs[:3]):
                logger.info(f"Chunk {i+1}:")
                logger.info(f"Content: {doc.page_content[:150]}...")
                logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        else:
            logger.warning("No documents loaded from pdf file")
            
    except Exception as e:
        logger.error(f"Error processing pdf file: {e}")
    
if __name__ == "__main__":
    main()
