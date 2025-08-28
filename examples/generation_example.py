"""
Example demonstrating the usage of the generation module.

This example shows how to use different generation strategies
with the RAG system.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Import the generation module
from generation import GeneratorFactory

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
            metadata={"source": "AI_Introduction.txt", "score": 0.95}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            metadata={"source": "ML_Overview.txt", "score": 0.88}
        ),
        Document(
            page_content="Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.",
            metadata={"source": "Deep_Learning.txt", "score": 0.92}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language.",
            metadata={"source": "NLP_Guide.txt", "score": 0.85}
        )
    ]


def demonstrate_simple_generator(llm):
    """Demonstrate the simple generator."""
    print("\n" + "="*60)
    print("SIMPLE GENERATOR DEMONSTRATION")
    print("="*60)
    
    # Create simple generator
    generator = GeneratorFactory.create_generator(
        generator_type="simple",
        llm=llm,
        max_tokens=500,
        temperature=0.7,
        include_sources=True
    )
    
    # Sample query and documents
    query = "What is artificial intelligence and how does it relate to machine learning?"
    documents = create_sample_documents()
    
    # Generate response
    print(f"Query: {query}")
    print("\nGenerating response...")
    
    response = generator.generate(query, documents)
    print(f"\nResponse:\n{response}")
    
    # Generate with metadata
    print("\n" + "-"*40)
    print("WITH METADATA:")
    result = generator.generate_with_metadata(query, documents)
    print(f"Sources: {result['sources']}")
    print(f"Documents used: {result['num_documents_used']}")
    print(f"Generator type: {result['generator_type']}")


def demonstrate_contextual_generator(llm):
    """Demonstrate the contextual generator."""
    print("\n" + "="*60)
    print("CONTEXTUAL GENERATOR DEMONSTRATION")
    print("="*60)
    
    # Create contextual generator
    generator = GeneratorFactory.create_generator(
        generator_type="contextual",
        llm=llm,
        max_tokens=800,
        temperature=0.5,
        context_window=3000,
        enable_reasoning=True
    )
    
    # Sample query and documents
    query = "Explain the relationship between AI, machine learning, and deep learning with examples."
    documents = create_sample_documents()
    
    # Generate response
    print(f"Query: {query}")
    print("\nGenerating response...")
    
    response = generator.generate(query, documents)
    print(f"\nResponse:\n{response}")
    
    # Generate with metadata
    print("\n" + "-"*40)
    print("WITH METADATA:")
    result = generator.generate_with_metadata(query, documents)
    print(f"Sources: {result['sources']}")
    print(f"Source scores: {result['source_scores']}")
    print(f"Generator type: {result['generator_type']}")


def demonstrate_chain_of_thought_generator(llm):
    """Demonstrate the chain of thought generator."""
    print("\n" + "="*60)
    print("CHAIN OF THOUGHT GENERATOR DEMONSTRATION")
    print("="*60)
    
    # Create chain of thought generator
    generator = GeneratorFactory.create_generator(
        generator_type="chain_of_thought",
        llm=llm,
        max_tokens=1000,
        temperature=0.3,
        reasoning_steps=4,
        show_reasoning=True
    )
    
    # Sample query and documents
    query = "How would you explain the evolution from AI to deep learning to someone new to the field?"
    documents = create_sample_documents()
    
    # Generate response
    print(f"Query: {query}")
    print("\nGenerating response...")
    
    response = generator.generate(query, documents)
    print(f"\nResponse:\n{response}")
    
    # Generate with metadata
    print("\n" + "-"*40)
    print("WITH METADATA:")
    result = generator.generate_with_metadata(query, documents)
    print(f"Reasoning steps: {result['reasoning_steps']}")
    print(f"Reasoning quality: {result['reasoning_quality']}")
    print(f"Generator type: {result['generator_type']}")


def demonstrate_multi_agent_generator(llm):
    """Demonstrate the multi-agent generator."""
    print("\n" + "="*60)
    print("MULTI-AGENT GENERATOR DEMONSTRATION")
    print("="*60)
    
    # Create multi-agent generator
    generator = GeneratorFactory.create_generator(
        generator_type="multi_agent",
        llm=llm,
        max_tokens=1200,
        temperature=0.6,
        num_agents=3,
        show_agent_contributions=True
    )
    
    # Sample query and documents
    query = "What are the key differences between traditional AI, machine learning, and deep learning approaches?"
    documents = create_sample_documents()
    
    # Generate response
    print(f"Query: {query}")
    print("\nGenerating response...")
    
    response = generator.generate(query, documents)
    print(f"\nResponse:\n{response}")
    
    # Generate with metadata
    print("\n" + "-"*40)
    print("WITH METADATA:")
    result = generator.generate_with_metadata(query, documents)
    print(f"Number of agents: {result['num_agents']}")
    print(f"Agent contributions: {list(result['agent_contributions'].keys())}")
    print(f"Generator type: {result['generator_type']}")


def demonstrate_generator_comparison(llm):
    """Compare different generators with the same query."""
    print("\n" + "="*60)
    print("GENERATOR COMPARISON")
    print("="*60)
    
    query = "What is the difference between AI and machine learning?"
    documents = create_sample_documents()
    
    generators = {
        "Simple": GeneratorFactory.create_generator("simple", llm, max_tokens=300),
        "Contextual": GeneratorFactory.create_generator("contextual", llm, max_tokens=300),
        "Chain of Thought": GeneratorFactory.create_generator("chain_of_thought", llm, max_tokens=300, show_reasoning=False),
        "Multi-Agent": GeneratorFactory.create_generator("multi_agent", llm, max_tokens=300, show_agent_contributions=False)
    }
    
    for name, generator in generators.items():
        print(f"\n{name} Generator:")
        print("-" * 30)
        response = generator.generate(query, documents)
        print(response[:200] + "..." if len(response) > 200 else response)


def demonstrate_factory_features():
    """Demonstrate factory features."""
    print("\n" + "="*60)
    print("FACTORY FEATURES")
    print("="*60)
    
    # Get available generators
    available = GeneratorFactory.get_available_generators()
    print("Available generators:")
    for gen_type, description in available.items():
        print(f"  - {gen_type}: {description}")
    
    # Get default configurations
    print("\nDefault configurations:")
    for gen_type in available.keys():
        config = GeneratorFactory.get_generator_config(gen_type)
        print(f"  - {gen_type}: {config}")


def main():
    """Main function to run all demonstrations."""
    print("GENERATION MODULE DEMONSTRATION")
    print("="*60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        return
    
    try:
        # Create language model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Demonstrate factory features
        demonstrate_factory_features()
        
        # Demonstrate each generator type
        demonstrate_simple_generator(llm)
        demonstrate_contextual_generator(llm)
        demonstrate_chain_of_thought_generator(llm)
        demonstrate_multi_agent_generator(llm)
        
        # Compare generators
        demonstrate_generator_comparison(llm)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
