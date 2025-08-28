# Naive RAG Project

A comprehensive RAG (Retrieval-Augmented Generation) system with modular components for document processing, embedding, vector storage, retrieval, and generation.

## Project Overview

This project provides a complete RAG pipeline with the following components:

- **Document Preparation**: Loaders and chunkers for various document formats
- **Embeddings**: Multiple embedding strategies with factory pattern
- **Vector Stores**: Vector database integration (Pinecone)
- **Retrieval**: Advanced retrieval strategies for improved document search
- **Generation**: Multiple generation strategies for response creation
- **RAG Processor**: High-level orchestrator for the complete pipeline

## Project Structure

```
naive_rag/
├── document_prepration/     # Document loading and chunking
│   ├── loaders/            # Document loaders (PDF, DOCX, TXT, etc.)
│   ├── chunkers/           # Text chunking strategies
│   └── processors/         # Document processing utilities
├── embeddings/             # Embedding generation
│   ├── base_embedder.py    # Base embedding interface
│   ├── openai_embedder.py  # OpenAI embeddings
│   └── embedder_factory.py # Factory for creating embedders
├── vector_stores/          # Vector database integration
│   ├── pinecone_store.py   # Pinecone vector store
│   └── vector_store_processor.py
├── retrieval/              # Advanced retrieval strategies
│   ├── base_retriever.py   # Base retriever interface
│   ├── retriever_factory.py # Factory for creating retrievers
│   └── strategies/         # Different retrieval strategies
│       ├── simple_retriever.py
│       ├── hybrid_retriever.py
│       ├── multi_query_retriever.py
│       ├── contextual_retriever.py
│       ├── rerank_retriever.py
│       ├── time_aware_retriever.py
│       └── ensemble_retriever.py
├── generation/             # Response generation strategies
│   ├── base_generator.py   # Base generator interface
│   ├── generator_factory.py # Factory for creating generators
│   └── strategies/         # Different generation strategies
│       ├── simple_generator.py
│       ├── contextual_generator.py
│       ├── chain_of_thought_generator.py
│       └── multi_agent_generator.py
├── examples/               # Usage examples
├── notebooks/              # Jupyter notebooks
├── data/                   # Data files
├── main.py                 # Main application entry point
├── rag_processor.py        # High-level RAG orchestrator
└── requirements.txt        # Python dependencies
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd naive_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_ENVIRONMENT="your-pinecone-environment"
```

### Basic Usage

```python
from rag_processor import RAGProcessor

# Create a RAG processor
processor = RAGProcessor.create_with_factory(
    embedder_type="openai",
    index_name="my-documents",
    namespace="my-namespace"
)

# Process documents
documents = [...]  # Your documents
processor.process_documents(documents)

# Search for similar documents
results = processor.search_similar("What is machine learning?", k=5)

# Complete RAG query with generation
response = processor.query("What is machine learning?", k=5)
```

### Using Advanced Retrieval

```python
from retrieval import RetrieverFactory
from embeddings.embedder_factory import EmbedderFactory
from vector_stores.pinecone_store import PineconeStore

# Set up components
embeddings = EmbedderFactory.create_embedder("openai")
vector_store = PineconeStore(index_name="my-index")
```

### Using Generation Strategies

```python
from generation import GeneratorFactory
from langchain_openai import ChatOpenAI

# Create language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create different generators
simple_gen = GeneratorFactory.create_generator("simple", llm=llm)
contextual_gen = GeneratorFactory.create_generator("contextual", llm=llm)
cot_gen = GeneratorFactory.create_generator("chain_of_thought", llm=llm)
multi_agent_gen = GeneratorFactory.create_generator("multi_agent", llm=llm)

# Generate responses
response = generator.generate(query, retrieved_documents)
```

# Create different retrieval strategies
simple_retriever = RetrieverFactory.create_retriever(
    retriever_type="simple",
    embeddings=embeddings,
    vector_store=vector_store
)

hybrid_retriever = RetrieverFactory.create_retriever(
    retriever_type="hybrid",
    embeddings=embeddings,
    vector_store=vector_store,
    vector_weight=0.7,
    keyword_weight=0.3
)

# Use retrievers
results = simple_retriever.retrieve("What is AI?", k=5)
hybrid_results = hybrid_retriever.retrieve("What is AI?", k=5)
```

## Available Retrieval Strategies

The retrieval module provides 7 different strategies:

1. **Simple Retriever**: Basic vector similarity search
2. **Hybrid Retriever**: Combines vector and keyword search
3. **Multi-Query Retriever**: Generates multiple queries for better coverage
4. **Contextual Retriever**: Uses conversation history for context-aware retrieval
5. **Rerank Retriever**: Two-stage retrieval with sophisticated reranking
6. **Time-Aware Retriever**: Considers temporal aspects in retrieval
7. **Ensemble Retriever**: Combines multiple strategies for better performance

See [retrieval/README.md](retrieval/README.md) for detailed documentation.

## Examples

- `examples/retrieval_example.py`: Comprehensive examples of all retrieval strategies
- `examples/embedder_integration.py`: Embedding system examples
- `examples/loader_chunker_integration.py`: Document processing examples
- `notebooks/`: Jupyter notebooks for exploration

## Features

### Document Preparation
- Support for multiple document formats (PDF, DOCX, TXT, websites)
- Configurable chunking strategies (character, sentence, token-based)
- Metadata preservation and processing

### Embeddings
- Multiple embedding providers (OpenAI, with extensible architecture)
- Factory pattern for easy embedding creation
- Configurable embedding dimensions and parameters

### Vector Storage
- Pinecone integration with serverless support
- Namespace management for document organization
- Index statistics and management

### Retrieval
- 7 different retrieval strategies
- Configurable parameters for each strategy
- Ensemble methods for combining strategies
- Comprehensive error handling and logging

### RAG Processing
- High-level orchestrator for complete RAG pipeline
- Factory pattern for easy setup
- Integrated document processing and retrieval

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment

# Optional
PINECONE_INDEX_NAME=default-index
OPENAI_MODEL_NAME=text-embedding-ada-002
```

### Configuration Files

You can configure various aspects of the system through the factory methods and constructor parameters. See individual module documentation for specific configuration options.

## Performance Considerations

1. **Batch Processing**: Process documents in batches for better performance
2. **Caching**: Cache embeddings and retrieval results when possible
3. **Index Optimization**: Use appropriate Pinecone index configurations
4. **Retrieval Strategy Selection**: Choose the right retrieval strategy for your use case

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
1. Check the documentation in each module
2. Review the examples
3. Open an issue on GitHub

## Roadmap

- [ ] Additional embedding providers (Hugging Face, Cohere)
- [ ] More vector store integrations (Weaviate, Qdrant)
- [ ] Advanced reranking models
- [ ] Performance benchmarking tools
- [ ] Web interface for RAG pipeline management
