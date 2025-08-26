# Embeddings Module

This module provides a comprehensive system for generating embeddings from text and documents.

## Overview

The embeddings module consists of the following components:

- **BaseEmbedder**: Abstract base class for all embedders
- **OpenAIEmbedder**: Implementation using OpenAI's embedding models
- **EmbedderFactory**: Factory pattern for creating embedders

For vector storage and high-level processing, see the `vector_stores` module which contains:
- **PineconeStore**: Vector store for Pinecone integration
- **VectorStoreProcessor**: Processor for vector store operations

For complete RAG pipeline orchestration, see the `rag_processor.py` file which contains:
- **RAGProcessor**: High-level processor that orchestrates both embedding and vector store operations

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Environment Variables

Set up the following environment variables:

```bash
# OpenAI API key for embeddings
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone credentials
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
```

## Quick Start

### Basic Usage

```python
from rag_processor import RAGProcessor

# Create a RAG processor
processor = RAGProcessor.create_with_factory(
    embedder_type="openai",
    index_name="my-index",
    namespace="my-namespace"
)

# Set up the Pinecone index
processor.setup_index()

# Process documents
documents = [...]  # List of LangChain Document objects
success = processor.process_documents(documents)

# Search for similar documents
query = "What is machine learning?"
similar_docs = processor.search_similar(query, k=5)
```

### Complete Pipeline Example

```python
from document_prepration.loaders.txt_loader import TXTLoader
from document_prepration.chunkers.recursive_character_chunker import RecursiveCharacterChunker
from rag_processor import RAGProcessor

# 1. Load documents
loader = TXTLoader("path/to/document.txt")
documents = loader.load_with_metadata({'source': 'example'})

# 2. Chunk documents
chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
chunked_docs = chunker.chunk_documents(documents)

# 3. Create RAG processor
processor = RAGProcessor.create_with_factory(
    embedder_type="openai",
    index_name="example-index"
)

# 4. Set up index and process
processor.setup_index()
processor.process_documents(chunked_docs)

# 5. Search
results = processor.search_similar("Your query here", k=3)
```

## Components

### BaseEmbedder

Abstract base class that defines the interface for all embedders.

```python
from embeddings import BaseEmbedder

class CustomEmbedder(BaseEmbedder):
    def _create_embedder(self, **kwargs):
        # Implement your custom embedder
        pass
```

### OpenAIEmbedder

Uses OpenAI's embedding models through LangChain.

```python
from embeddings import OpenAIEmbedder

embedder = OpenAIEmbedder(
    model_name="text-embedding-ada-002",
    api_key="your_api_key"  # Optional if set in environment
)
```

### PineconeStore

Manages Pinecone vector store operations.

```python
from vector_stores import PineconeStore

store = PineconeStore(
    index_name="my-index",
    api_key="your_api_key",  # Optional if set in environment
    environment="your_environment"  # Optional if set in environment
)

# Create index
store.create_index(dimension=1536, metric="cosine")

# Add documents
store.add_documents(documents, embeddings)

# Search
results = store.similarity_search(query, embeddings, k=5)
```

### EmbedderFactory

Factory pattern for creating embedders.

```python
from embeddings import EmbedderFactory

# List available embedders
available = EmbedderFactory.list_available_embedders()

# Create embedder
embedder = EmbedderFactory.get_embedder("openai", model_name="text-embedding-ada-002")

# Register custom embedder
EmbedderFactory.register_embedder("custom", CustomEmbedder)
```

### EmbeddingProcessor

High-level processor that combines embedding and storage functionality.

```python
from vector_stores import EmbeddingProcessor

# Create processor
processor = EmbeddingProcessor.create_with_factory(
    embedder_type="openai",
    index_name="my-index",
    namespace="my-namespace"
)

# Set up index
processor.setup_index()

# Process documents
processor.process_documents(documents)

# Search
results = processor.search_similar("query", k=5)
results_with_scores = processor.search_similar_with_scores("query", k=5)

# Get statistics
stats = processor.get_index_stats()

# Clean up
processor.delete_index()
```

## Advanced Usage

### Custom Embedders

You can create custom embedders by inheriting from `BaseEmbedder`:

```python
from embeddings import BaseEmbedder
from langchain_core.embeddings import Embeddings

class CustomEmbedder(BaseEmbedder):
    def _create_embedder(self, **kwargs) -> Embeddings:
        # Return your custom embeddings implementation
        return YourCustomEmbeddings(**kwargs)
```

### Batch Processing

For large datasets, you can process texts in batches:

```python
texts = ["text1", "text2", "text3", ...]
metadatas = [{"source": "doc1"}, {"source": "doc2"}, ...]

processor.process_texts(texts, metadatas)
```

### Namespace Management

Use namespaces to organize your embeddings:

```python
# Store in different namespaces
processor.namespace = "technical_docs"
processor.process_documents(technical_docs)

processor.namespace = "user_queries"
processor.process_documents(user_queries)

# Search in specific namespace
results = processor.search_similar("query", k=5)  # Uses current namespace
```

## Error Handling

The module includes comprehensive error handling and logging:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# All operations include proper error handling
try:
    processor.process_documents(documents)
except Exception as e:
    logger.error(f"Error processing documents: {e}")
```

## Examples

See the `examples/` directory for complete working examples:

- `embedding_example.py`: Basic embedding and search example
- `embedder_integration.py`: Complete pipeline integration

## API Reference

### EmbeddingProcessor Methods

- `setup_index(dimension=None, metric="cosine")`: Set up Pinecone index
- `process_documents(documents)`: Process and store document embeddings
- `process_texts(texts, metadatas=None)`: Process and store text embeddings
- `search_similar(query, k=4)`: Search for similar documents
- `search_similar_with_scores(query, k=4)`: Search with similarity scores
- `get_index_stats()`: Get index statistics
- `delete_index()`: Delete the index

### PineconeStore Methods

- `create_index(dimension, metric="cosine")`: Create new index
- `add_documents(documents, embeddings, namespace=None)`: Add documents
- `add_texts(texts, embeddings, metadatas=None, namespace=None)`: Add texts
- `similarity_search(query, embeddings, k=4, namespace=None)`: Search documents
- `similarity_search_with_score(query, embeddings, k=4, namespace=None)`: Search with scores
- `get_index_stats()`: Get statistics
- `delete_index()`: Delete index

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required environment variables are set
2. **Index Already Exists**: Use `setup_index()` which handles existing indexes
3. **Dimension Mismatch**: Let the system auto-detect dimensions with `setup_index()`
4. **Rate Limits**: The system includes built-in rate limiting for API calls

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
