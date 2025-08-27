"""
Example demonstrating different retrieval strategies.
"""

import os
import sys
import logging
from typing import List
from langchain.schema import Document
import dotenv

dotenv.load_dotenv()

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the retrieval components
from retrieval import RetrieverFactory
from embeddings.embedder_factory import EmbedderFactory
from vector_stores.pinecone_store import PineconeStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents() -> List[Document]:
    """
    Create sample documents for testing retrieval.
    
    Returns:
        List of sample documents
    """
    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            metadata={"source": "ai_guide.txt", "topic": "machine_learning", "timestamp": "2024-01-15"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"source": "ai_guide.txt", "topic": "deep_learning", "timestamp": "2024-01-16"}
        ),
        Document(
            page_content="Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
            metadata={"source": "nlp_guide.txt", "topic": "nlp", "timestamp": "2024-01-17"}
        ),
        Document(
            page_content="Computer vision is a field of AI that trains computers to interpret and understand visual information from the world.",
            metadata={"source": "cv_guide.txt", "topic": "computer_vision", "timestamp": "2024-01-18"}
        ),
        Document(
            page_content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment.",
            metadata={"source": "rl_guide.txt", "topic": "reinforcement_learning", "timestamp": "2024-01-19"}
        ),
        Document(
            page_content="Supervised learning is a machine learning approach where the model learns from labeled training data.",
            metadata={"source": "ml_guide.txt", "topic": "supervised_learning", "timestamp": "2024-01-20"}
        ),
        Document(
            page_content="Unsupervised learning finds hidden patterns in data without labeled examples.",
            metadata={"source": "ml_guide.txt", "topic": "unsupervised_learning", "timestamp": "2024-01-21"}
        ),
        Document(
            page_content="Transfer learning allows models to leverage knowledge from one task to improve performance on another related task.",
            metadata={"source": "transfer_learning.txt", "topic": "transfer_learning", "timestamp": "2024-01-22"}
        )
    ]
    
    return documents


def setup_vector_store() -> tuple:
    """
    Set up the vector store and embeddings.
    
    Returns:
        Tuple of (embeddings, vector_store)
    """
    # Create embeddings
    embeddings = EmbedderFactory.get_embedder("openai")
    
    # Create vector store
    vector_store = PineconeStore(
        index_name="retrieval-test-index",
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    
    # Create index if it doesn't exist
    dimension = embeddings.get_embedding_dimensions()
    vector_store.create_index(dimension=dimension)
    
    return embeddings, vector_store


def test_simple_retriever(embeddings, vector_store):
    """
    Test the simple retriever.
    
    Args:
        embeddings: Embeddings instance
        vector_store: Vector store instance
    """
    print("\n=== Testing Simple Retriever ===")
    
    # Create simple retriever
    retriever = RetrieverFactory.create_retriever(
        retriever_type="simple",
        embeddings=embeddings,
        vector_store=vector_store,
        namespace="test"
    )
    
    # Test retrieval
    query = "What is machine learning?"
    results = retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print()


def test_hybrid_retriever(embeddings, vector_store):
    """
    Test the hybrid retriever.
    
    Args:
        embeddings: Embeddings instance
        vector_store: Vector store instance
    """
    print("\n=== Testing Hybrid Retriever ===")
    
    # Create hybrid retriever
    retriever = RetrieverFactory.create_retriever(
        retriever_type="hybrid",
        embeddings=embeddings,
        vector_store=vector_store,
        vector_weight=0.7,
        keyword_weight=0.3,
        namespace="test"
    )
    
    # Test retrieval
    query = "How does deep learning work?"
    results = retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print()


def test_multi_query_retriever(embeddings, vector_store):
    """
    Test the multi-query retriever.
    
    Args:
        embeddings: Embeddings instance
        vector_store: Vector store instance
    """
    print("\n=== Testing Multi-Query Retriever ===")
    
    # Create multi-query retriever
    retriever = RetrieverFactory.create_retriever(
        retriever_type="multi_query",
        embeddings=embeddings,
        vector_store=vector_store,
        num_queries=3,
        query_generation_strategy="synonym",
        namespace="test"
    )
    
    # Test retrieval
    query = "Explain natural language processing"
    results = retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print()


def test_contextual_retriever(embeddings, vector_store):
    """
    Test the contextual retriever.
    
    Args:
        embeddings: Embeddings instance
        vector_store: Vector store instance
    """
    print("\n=== Testing Contextual Retriever ===")
    
    # Create contextual retriever
    retriever = RetrieverFactory.create_retriever(
        retriever_type="contextual",
        embeddings=embeddings,
        vector_store=vector_store,
        context_window=2,
        context_weight=0.3,
        namespace="test"
    )
    
    # Simulate conversation
    retriever.add_to_history("Tell me about AI", is_user=True)
    retriever.add_to_history("AI is a broad field that includes machine learning", is_user=False)
    retriever.add_to_history("What about deep learning?", is_user=True)
    
    # Test retrieval with context
    query = "How does it work?"
    results = retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Context: {retriever.get_history()}")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print()


def test_rerank_retriever(embeddings, vector_store):
    """
    Test the rerank retriever.
    
    Args:
        embeddings: Embeddings instance
        vector_store: Vector store instance
    """
    print("\n=== Testing Rerank Retriever ===")
    
    # Create rerank retriever
    retriever = RetrieverFactory.create_retriever(
        retriever_type="rerank",
        embeddings=embeddings,
        vector_store=vector_store,
        initial_k=10,
        final_k=3,
        rerank_strategy="hybrid",
        namespace="test"
    )
    
    # Test retrieval
    query = "What is supervised learning?"
    results = retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print()


def test_time_aware_retriever(embeddings, vector_store):
    """
    Test the time-aware retriever.
    
    Args:
        embeddings: Embeddings instance
        vector_store: Vector store instance
    """
    print("\n=== Testing Time-Aware Retriever ===")
    
    # Create time-aware retriever
    retriever = RetrieverFactory.create_retriever(
        retriever_type="time_aware",
        embeddings=embeddings,
        vector_store=vector_store,
        time_decay_factor=0.1,
        recency_weight=0.3,
        namespace="test"
    )
    
    # Test retrieval
    query = "What is transfer learning?"
    results = retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"   Date: {doc.metadata.get('timestamp', 'Unknown')}")
        print()


def test_ensemble_retriever(embeddings, vector_store):
    """
    Test the ensemble retriever.
    
    Args:
        embeddings: Embeddings instance
        vector_store: Vector store instance
    """
    print("\n=== Testing Ensemble Retriever ===")
    
    # Create individual retrievers
    simple_retriever = RetrieverFactory.create_retriever(
        retriever_type="simple",
        embeddings=embeddings,
        vector_store=vector_store,
        namespace="test"
    )
    
    hybrid_retriever = RetrieverFactory.create_retriever(
        retriever_type="hybrid",
        embeddings=embeddings,
        vector_store=vector_store,
        namespace="test"
    )
    
    # Create ensemble retriever
    ensemble_retriever = RetrieverFactory.create_ensemble_retriever(
        retriever_configs=[
            {"type": "simple", "weight": 0.6},
            {"type": "hybrid", "weight": 0.4}
        ],
        embeddings=embeddings,
        vector_store=vector_store,
        ensemble_strategy="weighted"
    )
    
    # Test retrieval
    query = "What is computer vision?"
    results = ensemble_retriever.retrieve(query, k=3)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print()


def main():
    """
    Main function to run all retrieval examples.
    """
    print("Retrieval Strategies Example")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_ENVIRONMENT"):
        print("Please set PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    try:
        # Set up vector store and embeddings
        print("Setting up vector store and embeddings...")
        embeddings, vector_store = setup_vector_store()
        
        # Create and add sample documents
        print("Adding sample documents...")
        documents = create_sample_documents()
        vector_store.add_documents(documents, embeddings, namespace="test")
        
        # Test different retrieval strategies
        test_simple_retriever(embeddings, vector_store)
        # test_hybrid_retriever(embeddings, vector_store)
        # test_multi_query_retriever(embeddings, vector_store)
        # test_contextual_retriever(embeddings, vector_store)
        # test_rerank_retriever(embeddings, vector_store)
        # test_time_aware_retriever(embeddings, vector_store)
        # test_ensemble_retriever(embeddings, vector_store)
        
        print("\nAll retrieval tests completed successfully!")
        
    except Exception as e:
        print(f"Error running retrieval examples: {str(e)}")
        logger.error(f"Error running retrieval examples: {str(e)}")


if __name__ == "__main__":
    main()
