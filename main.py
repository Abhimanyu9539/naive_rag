"""
Simple example demonstrating PDF loading and recursive chunking.
"""

import logging
from document_prepration.loaders.pdf_loader import PDFLoader
from document_prepration.chunkers.chunker_factory import ChunkerFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate PDF loading and chunking."""
    
    # PDF file path - you can change this to your PDF file
    pdf_path = r"E:\Gen AI\RAG_Projects\naive_rag\data\raw\attention.pdf"
    
    try:
        # Step 1: Load the PDF
        logger.info("Loading PDF file...")
        pdf_loader = PDFLoader(pdf_path)
        documents = pdf_loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Step 2: Initialize the recursive chunker using factory
        logger.info("Initializing recursive chunker using factory...")
        chunker = ChunkerFactory.create_chunker(
            chunker_type='recursive_character',
            chunk_size=1000,      # 1000 characters per chunk
            chunk_overlap=200     # 200 characters overlap between chunks
        )
        
        # Step 3: Chunk the documents
        logger.info("Chunking documents...")
        chunked_docs = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        # Step 4: Display some information about the chunks
        logger.info("\n=== Chunk Information ===")
        for i, doc in enumerate(chunked_docs[:5]):  # Show first 5 chunks
            logger.info(f"Chunk {i+1}: {len(doc.page_content)} characters")
            logger.info(f"Content preview: {doc.page_content[:100]}...")
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("-" * 50)
        
        if len(chunked_docs) > 5:
            logger.info(f"... and {len(chunked_docs) - 5} more chunks")
        
        logger.info("PDF loading and chunking completed successfully!")
        
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        logger.info("Please place a PDF file in the data/ directory or update the pdf_path variable")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
