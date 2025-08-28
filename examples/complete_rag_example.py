"""
Complete RAG Pipeline Example

This example demonstrates the full RAG pipeline including:
1. Document processing and embedding
2. Vector storage
3. Retrieval
4. Generation

It shows how all modules work together in a complete RAG system.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Import all RAG components
from document_prepration.processors.document_processor import DocumentProcessor
from embeddings.embedder_factory import EmbedderFactory
from vector_stores.pinecone_store import PineconeStore
from vector_stores.vector_store_processor import VectorStoreProcessor
from retrieval.retriever_factory import RetrieverFactory
from generation import GeneratorFactory
from rag_processor import RAGProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def create_sample_documents():
    """Create sample documents for demonstration."""
    return [
        Document(
            page_content="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem solving.",
            metadata={"source": "AI_Introduction.txt", "category": "AI", "author": "Tech Expert"}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            metadata={"source": "ML_Overview.txt", "category": "ML", "author": "Data Scientist"}
        ),
        Document(
            page_content="Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.",
            metadata={"source": "Deep_Learning.txt", "category": "DL", "author": "ML Researcher"}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language.",
            metadata={"source": "NLP_Guide.txt", "category": "NLP", "author": "NLP Specialist"}
        ),
        Document(
            page_content="Computer Vision is a field of AI that trains computers to interpret and understand visual information from the world. It enables machines to identify and process images, videos, and other visual data.",
            metadata={"source": "Computer_Vision.txt", "category": "CV", "author": "Vision Expert"}
        ),
        Document(
            page_content="Robotics combines AI, machine learning, and computer vision to create autonomous systems that can perform tasks in the physical world. Modern robots can learn from their environment and adapt to new situations.",
            metadata={"source": "Robotics.txt", "category": "Robotics", "author": "Robotics Engineer"}
        )
    ]


def demonstrate_complete_rag_pipeline():
    """Demonstrate the complete RAG pipeline."""
    print("COMPLETE RAG PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not found in environment variables.")
        return
    
    try:
        # Step 1: Create language model
        print("\n1. Creating Language Model...")
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        print("✓ Language model created")
        
        # Step 2: Create RAG processor with all components
        print("\n2. Creating RAG Processor...")
        rag_processor = RAGProcessor.create_with_factory(
            embedder_type="openai",
            index_name="complete-rag-demo",
            generator_type="contextual",
            llm=llm,
            namespace="demo",
            max_tokens=800,
            temperature=0.5
        )
        print("✓ RAG processor created with all components")
        
        # Step 3: Set up vector store index
        print("\n3. Setting up Vector Store Index...")
        success = rag_processor.setup_index(metric="cosine")
        if success:
            print("✓ Vector store index set up successfully")
        else:
            print("⚠ Index already exists or setup failed")
        
        # Step 4: Process documents
        print("\n4. Processing Documents...")
        documents = create_sample_documents()
        success = rag_processor.process_documents(documents)
        if success:
            print(f"✓ Processed {len(documents)} documents")
        else:
            print("✗ Failed to process documents")
            return
        
        # Step 5: Demonstrate different query types
        print("\n5. Demonstrating RAG Queries...")
        
        queries = [
            "What is artificial intelligence?",
            "How does machine learning relate to AI?",
            "Explain the differences between AI, ML, and deep learning",
            "What are the applications of computer vision?",
            "How do robots use AI and machine learning?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # Simple query
            response = rag_processor.query(query, k=3)
            print(f"Response: {response[:200]}...")
            
            # Query with metadata
            result = rag_processor.query_with_metadata(query, k=3)
            print(f"Sources: {result['generation_metadata']['sources']}")
            print(f"Documents retrieved: {result['retrieval_stats']['num_documents_retrieved']}")
            print(f"Average similarity: {result['retrieval_stats']['average_similarity_score']:.3f}")
        
        # Step 6: Demonstrate different generation strategies
        print("\n6. Demonstrating Different Generation Strategies...")
        
        test_query = "What is the relationship between AI and robotics?"
        
        generation_strategies = [
            ("simple", "Simple Generation"),
            ("contextual", "Contextual Generation"),
            ("chain_of_thought", "Chain of Thought"),
            ("multi_agent", "Multi-Agent Generation")
        ]
        
        for strategy, name in generation_strategies:
            print(f"\n--- {name} ---")
            
            # Create new generator
            generator = GeneratorFactory.create_generator(
                generator_type=strategy,
                llm=llm,
                max_tokens=400,
                show_reasoning=False,
                show_agent_contributions=False
            )
            
            # Set generator and query
            rag_processor.set_generator(generator)
            response = rag_processor.query(test_query, k=3)
            print(f"Response: {response[:150]}...")
        
        # Step 7: Show system statistics
        print("\n7. System Statistics...")
        stats = rag_processor.get_index_stats()
        print(f"Index statistics: {stats}")
        
        generator_config = rag_processor.get_generator_config()
        print(f"Generator configuration: {generator_config}")
        
        print("\n" + "=" * 60)
        print("COMPLETE RAG PIPELINE DEMONSTRATION FINISHED")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in complete RAG pipeline: {e}")
        print(f"Error: {e}")


def demonstrate_individual_components():
    """Demonstrate individual components separately."""
    print("\nINDIVIDUAL COMPONENTS DEMONSTRATION")
    print("=" * 60)
    
    try:
        # 1. Document Processing
        print("\n1. Document Processing Component...")
        embedder = EmbedderFactory.create_embedder("openai")
        doc_processor = DocumentProcessor(embedder)
        
        documents = create_sample_documents()
        embedded_docs = doc_processor.embed_documents(documents)
        print(f"✓ Embedded {len(embedded_docs)} documents")
        
        # 2. Vector Store
        print("\n2. Vector Store Component...")
        vector_store = PineconeStore(index_name="component-demo")
        vector_store_processor = VectorStoreProcessor(vector_store)
        
        # Set up index
        dimension = doc_processor.get_embedding_dimensions()
        vector_store_processor.setup_index(dimension, "cosine")
        
        # Store documents
        success = vector_store_processor.store_documents(
            documents=documents,
            embeddings=embedder.embedder
        )
        print(f"✓ Stored documents in vector store: {success}")
        
        # 3. Retrieval
        print("\n3. Retrieval Component...")
        retriever = RetrieverFactory.create_retriever(
            "simple",
            embedder.embedder,
            vector_store
        )
        
        query = "What is machine learning?"
        retrieved_docs = retriever.retrieve(query, k=3)
        print(f"✓ Retrieved {len(retrieved_docs)} documents")
        
        # 4. Generation
        print("\n4. Generation Component...")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        generator = GeneratorFactory.create_generator(
            "contextual",
            llm=llm,
            max_tokens=300
        )
        
        response = generator.generate(query, retrieved_docs)
        print(f"✓ Generated response: {response[:100]}...")
        
        print("\n✓ All individual components working correctly")
        
    except Exception as e:
        logger.error(f"Error in individual components: {e}")
        print(f"Error: {e}")


def demonstrate_advanced_features():
    """Demonstrate advanced features of the RAG system."""
    print("\nADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Create RAG processor
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        rag_processor = RAGProcessor.create_with_factory(
            embedder_type="openai",
            index_name="advanced-demo",
            generator_type="multi_agent",
            llm=llm,
            namespace="advanced",
            show_agent_contributions=True
        )
        
        # Set up and populate
        rag_processor.setup_index()
        documents = create_sample_documents()
        rag_processor.process_documents(documents)
        
        # Advanced query with metadata
        query = "Explain the evolution of AI technologies from basic AI to modern deep learning applications"
        
        print(f"Query: {query}")
        print("\nGenerating comprehensive response...")
        
        result = rag_processor.query_with_metadata(query, k=4)
        
        print(f"\nResponse: {result['response']}")
        print(f"\nRetrieved {result['retrieval_stats']['num_documents_retrieved']} documents")
        print(f"Average similarity score: {result['retrieval_stats']['average_similarity_score']:.3f}")
        print(f"Sources: {result['generation_metadata']['sources']}")
        print(f"Generator type: {result['generation_metadata']['generator_type']}")
        
        # Show retrieved documents with scores
        print("\nRetrieved Documents:")
        for i, (doc, score) in enumerate(result['retrieved_documents'], 1):
            print(f"{i}. Score: {score:.3f} | Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:100]}...")
        
        print("\n✓ Advanced features demonstrated successfully")
        
    except Exception as e:
        logger.error(f"Error in advanced features: {e}")
        print(f"Error: {e}")


def main():
    """Main function to run all demonstrations."""
    print("COMPLETE RAG SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_complete_rag_pipeline()
    demonstrate_individual_components()
    demonstrate_advanced_features()
    
    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
