# Retrieval Module

This module provides a comprehensive set of retrieval strategies for RAG (Retrieval-Augmented Generation) systems. It includes various approaches to improve document retrieval quality and relevance.

## Overview

The retrieval module is designed with a modular architecture that allows you to easily switch between different retrieval strategies or combine them for better performance. All retrievers inherit from a common `BaseRetriever` interface, ensuring consistency and ease of use.

## Available Retrieval Strategies

### 1. Simple Retriever
- **Purpose**: Basic vector similarity search
- **Use Case**: Standard retrieval when you need straightforward semantic search
- **Features**: 
  - Direct vector similarity search
  - Query preprocessing
  - Duplicate removal

### 2. Hybrid Retriever
- **Purpose**: Combines vector similarity with keyword-based search
- **Use Case**: When you want to leverage both semantic and lexical matching
- **Features**:
  - Configurable weights for vector vs keyword scoring
  - TF-IDF like keyword scoring
  - Stop word removal

### 3. Multi-Query Retriever
- **Purpose**: Generates multiple queries from the original query
- **Use Case**: Improving retrieval diversity and coverage
- **Features**:
  - Synonym-based query generation
  - Paraphrasing techniques
  - Keyword extraction
  - Result aggregation and deduplication

### 4. Contextual Retriever
- **Purpose**: Uses conversation history to improve retrieval
- **Use Case**: Multi-turn conversations where context matters
- **Features**:
  - Conversation history management
  - Context-aware query expansion
  - Configurable context window

### 5. Rerank Retriever
- **Purpose**: Two-stage retrieval with sophisticated reranking
- **Use Case**: When you need high-quality results and can afford additional computation
- **Features**:
  - BM25 scoring
  - Cross-encoder scoring (simplified)
  - Hybrid scoring combining multiple methods

### 6. Time-Aware Retriever
- **Purpose**: Considers temporal aspects in retrieval
- **Use Case**: When document recency or timestamps matter
- **Features**:
  - Time decay scoring
  - Date range filtering
  - Recent document retrieval

### 7. Ensemble Retriever
- **Purpose**: Combines multiple retrieval strategies
- **Use Case**: Leveraging strengths of different approaches
- **Features**:
  - Weighted combination
  - Voting mechanism
  - Rank fusion (RRF)

## Quick Start

### Basic Usage

```python
from retrieval import RetrieverFactory
from embeddings.embedder_factory import EmbedderFactory
from vector_stores.pinecone_store import PineconeStore

# Set up components
embeddings = EmbedderFactory.create_embedder("openai")
vector_store = PineconeStore(index_name="my-index")

# Create a simple retriever
retriever = RetrieverFactory.create_retriever(
    retriever_type="simple",
    embeddings=embeddings,
    vector_store=vector_store,
    namespace="my-namespace"
)

# Retrieve documents
results = retriever.retrieve("What is machine learning?", k=5)
```

### Using Different Strategies

```python
# Hybrid retriever
hybrid_retriever = RetrieverFactory.create_retriever(
    retriever_type="hybrid",
    embeddings=embeddings,
    vector_store=vector_store,
    vector_weight=0.7,
    keyword_weight=0.3
)

# Multi-query retriever
multi_query_retriever = RetrieverFactory.create_retriever(
    retriever_type="multi_query",
    embeddings=embeddings,
    vector_store=vector_store,
    num_queries=3,
    query_generation_strategy="synonym"
)

# Contextual retriever
contextual_retriever = RetrieverFactory.create_retriever(
    retriever_type="contextual",
    embeddings=embeddings,
    vector_store=vector_store,
    context_window=3,
    context_weight=0.3
)

# Add conversation history
contextual_retriever.add_to_history("Tell me about AI", is_user=True)
contextual_retriever.add_to_history("AI is a broad field", is_user=False)
results = contextual_retriever.retrieve("How does it work?")
```

### Ensemble Retrieval

```python
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

results = ensemble_retriever.retrieve("What is deep learning?", k=5)
```

## Configuration Options

### Common Parameters

All retrievers support these common parameters:
- `embeddings`: Embeddings instance
- `vector_store`: Vector store instance
- `namespace`: Optional namespace for vector store operations

### Strategy-Specific Parameters

#### Hybrid Retriever
- `vector_weight`: Weight for vector similarity (0.0-1.0)
- `keyword_weight`: Weight for keyword matching (0.0-1.0)

#### Multi-Query Retriever
- `num_queries`: Number of queries to generate
- `query_generation_strategy`: "synonym", "paraphrase", or "keyword"

#### Contextual Retriever
- `context_window`: Number of previous messages to consider
- `context_weight`: Weight for context in scoring

#### Rerank Retriever
- `initial_k`: Number of documents to retrieve in first stage
- `final_k`: Number of documents to return after reranking
- `rerank_strategy`: "cross_encoder", "bm25", or "hybrid"

#### Time-Aware Retriever
- `time_decay_factor`: Factor for time decay (0.0-1.0)
- `recency_weight`: Weight for recency in scoring

#### Ensemble Retriever
- `retriever_configs`: List of retriever configurations with weights
- `ensemble_strategy`: "weighted", "voting", or "rank_fusion"

## Advanced Usage

### Custom Retriever

You can create custom retrievers by inheriting from `BaseRetriever`:

```python
from retrieval.base_retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def retrieve(self, query: str, k: int = 4, **kwargs):
        # Your custom retrieval logic here
        pass
    
    def retrieve_with_scores(self, query: str, k: int = 4, **kwargs):
        # Your custom retrieval with scores logic here
        pass

# Register your custom retriever
RetrieverFactory.register_retriever("custom", CustomRetriever)
```

### Performance Monitoring

```python
# Get retriever configuration and statistics
config = retriever.get_config()
stats = retriever.get_retrieval_stats()

print(f"Retriever type: {config['retriever_type']}")
print(f"Configuration: {config['config']}")
```

### Error Handling

All retrievers include comprehensive error handling:

```python
try:
    results = retriever.retrieve("invalid query", k=5)
except Exception as e:
    print(f"Retrieval failed: {str(e)}")
    # Handle error appropriately
```

## Best Practices

### Choosing the Right Strategy

1. **Simple Retriever**: Use for basic semantic search
2. **Hybrid Retriever**: Use when you need both semantic and keyword matching
3. **Multi-Query Retriever**: Use to improve retrieval diversity
4. **Contextual Retriever**: Use for conversational applications
5. **Rerank Retriever**: Use when you need high-quality results and can afford additional computation
6. **Time-Aware Retriever**: Use when document recency matters
7. **Ensemble Retriever**: Use to combine multiple strategies for better performance

### Performance Optimization

1. **Batch Processing**: Process multiple queries together when possible
2. **Caching**: Cache frequently requested results
3. **Indexing**: Ensure your vector store is properly indexed
4. **Monitoring**: Track retrieval performance and adjust parameters accordingly

### Configuration Tuning

1. **Start Simple**: Begin with the simple retriever and add complexity as needed
2. **A/B Testing**: Compare different strategies on your specific use case
3. **Parameter Tuning**: Adjust weights and parameters based on your data and requirements
4. **Evaluation**: Use metrics like precision, recall, and user feedback to evaluate performance

## Examples

See `examples/retrieval_example.py` for comprehensive examples of all retrieval strategies.

## Dependencies

- `langchain`: For document handling and embeddings
- `pinecone-client`: For vector store operations
- `numpy`: For numerical operations
- `re`: For regular expressions (keyword extraction)

## Contributing

To add new retrieval strategies:

1. Create a new class inheriting from `BaseRetriever`
2. Implement the required methods (`retrieve` and `retrieve_with_scores`)
3. Add the strategy to the `RetrieverFactory.RETRIEVER_REGISTRY`
4. Update this README with documentation
5. Add tests and examples

## License

This module is part of the naive_rag project and follows the same license terms.
