"""
Integration example showing how to combine loader, chunker, and embedder systems.

This script demonstrates a complete pipeline from document loading to embedding storage.
"""

import os
import logging
from dotenv import load_dotenv

# Import document preparation components
from document_prepration.loaders.txt_loader import TXTLoader
from document_prepration.chunkers.chunker_factory import ChunkerFactory

# Import embedding components
from rag_processor import RAGProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample data for demonstration."""
    sample_texts = [
        {
            'filename': 'machine_learning.txt',
            'content': """
            Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.
            It uses algorithms and statistical models to analyze and draw inferences from patterns in data.
            There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
            Supervised learning uses labeled training data to learn the mapping between inputs and outputs.
            Unsupervised learning finds hidden patterns in unlabeled data.
            Reinforcement learning learns through interaction with an environment to maximize rewards.
            """
        },
        {
            'filename': 'deep_learning.txt',
            'content': """
            Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers.
            These neural networks are inspired by the human brain and can learn complex patterns in data.
            Deep learning has achieved remarkable success in computer vision, natural language processing, and speech recognition.
            Convolutional Neural Networks (CNNs) are commonly used for image processing tasks.
            Recurrent Neural Networks (RNNs) and Transformers are used for sequential data like text and speech.
            The success of deep learning is largely due to the availability of large datasets and powerful computing resources.
            """
        },
        {
            'filename': 'natural_language_processing.txt',
            'content': """
            Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language.
            NLP enables computers to understand, interpret, and generate human language in a meaningful way.
            Key tasks in NLP include text classification, sentiment analysis, machine translation, and question answering.
            Modern NLP systems often use transformer architectures like BERT, GPT, and T5.
            These models are pre-trained on large text corpora and can be fine-tuned for specific tasks.
            NLP has applications in chatbots, virtual assistants, content recommendation, and automated customer service.
            """
        }
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    # Write sample files
    for sample in sample_texts:
        file_path = f"data/raw/{sample['filename']}"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(sample['content'].strip())
            logger.info(f"Created sample file: {file_path}")


def process_documents():
    """Process documents through the complete pipeline."""
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return
    
    try:
        # Step 1: Create sample data
        logger.info("Step 1: Creating sample data...")
        create_sample_data()
        
        # Step 2: Load documents
        logger.info("Step 2: Loading documents...")
        documents = []
        
        # Load all text files in the data/raw directory
        for filename in os.listdir("data/raw"):
            if filename.endswith('.txt'):
                file_path = os.path.join("data/raw", filename)
                loader = TXTLoader(file_path)
                docs = loader.load_with_metadata({
                    'source_file': filename,
                    'file_type': 'text',
                    'pipeline': 'integration_example'
                })
                documents.extend(docs)
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Step 3: Chunk documents
        logger.info("Step 3: Chunking documents...")
        chunker = ChunkerFactory.get_chunker("recursive_character", chunk_size=300, chunk_overlap=30)
        chunked_documents = chunker.chunk_documents(documents)
        
        logger.info(f"Created {len(chunked_documents)} chunks")
        
        # Step 4: Set up RAG processor
        logger.info("Step 4: Setting up RAG processor...")
        processor = RAGProcessor.create_with_factory(
            embedder_type="openai",
            index_name="integration-index",
            namespace="integration-namespace"
        )
        
        # Step 5: Set up Pinecone index
        logger.info("Step 5: Setting up Pinecone index...")
        processor.setup_index()
        
        # Step 6: Process and store embeddings
        logger.info("Step 6: Processing and storing embeddings...")
        success = processor.process_documents(chunked_documents)
        
        if not success:
            logger.error("Failed to process embeddings")
            return
        
        logger.info("Successfully processed and stored embeddings")
        
        # Step 7: Demonstrate search functionality
        logger.info("Step 7: Demonstrating search functionality...")
        
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "What are the applications of NLP?",
            "Explain deep learning architectures"
        ]
        
        for query in test_queries:
            logger.info(f"\nSearching for: '{query}'")
            similar_docs = processor.search_similar(query, k=2)
            
            for i, doc in enumerate(similar_docs, 1):
                logger.info(f"  Result {i}: {doc.page_content[:80]}...")
                logger.info(f"    Source: {doc.metadata.get('source_file', 'Unknown')}")
        
        # Step 8: Get index statistics
        logger.info("\nStep 8: Getting index statistics...")
        stats = processor.get_index_stats()
        logger.info(f"Index statistics: {stats}")
        
        logger.info("\nIntegration example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in integration example: {str(e)}")
        raise


if __name__ == "__main__":
    process_documents()
