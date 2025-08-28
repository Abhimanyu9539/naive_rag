# Generation Module

The Generation module provides various text generation strategies for creating responses based on retrieved documents and user queries in the RAG system.

## Overview

This module implements different generation approaches, from simple prompt-based generation to sophisticated multi-agent collaborative systems. Each strategy is designed to handle different use cases and requirements.

## Features

- **Multiple Generation Strategies**: Simple, Contextual, Chain of Thought, and Multi-Agent
- **Configurable Parameters**: Temperature, max tokens, context windows, etc.
- **Source Attribution**: Automatic citation and source tracking
- **Metadata Support**: Rich metadata for responses including sources and generation stats
- **Extensible Architecture**: Easy to add new generation strategies

## Architecture

```
generation/
├── __init__.py                 # Module exports
├── base_generator.py          # Abstract base class
├── generator_factory.py       # Factory for creating generators
├── README.md                  # This file
└── strategies/               # Generation strategies
    ├── __init__.py
    ├── simple_generator.py
    ├── contextual_generator.py
    ├── chain_of_thought_generator.py
    └── multi_agent_generator.py
```

## Generation Strategies

### 1. Simple Generator
- **Purpose**: Basic generation using simple prompt templates
- **Best for**: Quick responses, simple queries
- **Features**: 
  - Basic prompt template
  - Source citation
  - Configurable parameters

### 2. Contextual Generator
- **Purpose**: Enhanced context-aware generation
- **Best for**: Complex queries requiring deep understanding
- **Features**:
  - Enhanced context organization
  - Relevance scoring
  - Reasoning instructions
  - Source grouping

### 3. Chain of Thought Generator
- **Purpose**: Step-by-step reasoning generation
- **Best for**: Complex problem-solving, educational content
- **Features**:
  - Structured reasoning steps
  - Quality assessment
  - Configurable reasoning depth
  - Option to show/hide reasoning

### 4. Multi-Agent Generator
- **Purpose**: Collaborative generation using specialized agents
- **Best for**: Research tasks, comprehensive analysis
- **Features**:
  - Research agent for information extraction
  - Analysis agent for evaluation
  - Synthesis agent for final response
  - Collaborative workflow

## Usage

### Basic Usage

```python
from generation import GeneratorFactory
from langchain_openai import ChatOpenAI

# Create a language model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create a generator
generator = GeneratorFactory.create_generator(
    generator_type="contextual",
    llm=llm,
    max_tokens=1500,
    temperature=0.5
)

# Generate response
response = generator.generate(query, retrieved_documents)
```

### Advanced Usage with Metadata

```python
# Generate response with metadata
result = generator.generate_with_metadata(query, retrieved_documents)

print(f"Response: {result['response']}")
print(f"Sources: {result['sources']}")
print(f"Documents used: {result['num_documents_used']}")
print(f"Generator type: {result['generator_type']}")
```

### Configuration Examples

```python
# Simple generator
simple_gen = GeneratorFactory.create_generator(
    "simple",
    llm=llm,
    max_tokens=1000,
    temperature=0.7,
    include_sources=True
)

# Contextual generator
contextual_gen = GeneratorFactory.create_generator(
    "contextual",
    llm=llm,
    max_tokens=1500,
    temperature=0.5,
    context_window=4000,
    enable_reasoning=True
)

# Chain of thought generator
cot_gen = GeneratorFactory.create_generator(
    "chain_of_thought",
    llm=llm,
    max_tokens=2000,
    temperature=0.3,
    reasoning_steps=4,
    show_reasoning=True
)

# Multi-agent generator
multi_gen = GeneratorFactory.create_generator(
    "multi_agent",
    llm=llm,
    max_tokens=2000,
    temperature=0.6,
    num_agents=3,
    show_agent_contributions=True
)
```

## Configuration Parameters

### Common Parameters
- `max_tokens`: Maximum number of tokens in response
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = very random)
- `include_sources`: Whether to include source citations

### Strategy-Specific Parameters

#### Contextual Generator
- `context_window`: Maximum context length in characters
- `enable_reasoning`: Enable enhanced reasoning instructions

#### Chain of Thought Generator
- `reasoning_steps`: Number of reasoning steps to follow
- `show_reasoning`: Whether to show reasoning process in output

#### Multi-Agent Generator
- `num_agents`: Number of agents in the collaboration
- `show_agent_contributions`: Whether to show individual agent outputs

## Available Generators

```python
# Get list of available generators
available = GeneratorFactory.get_available_generators()
for gen_type, description in available.items():
    print(f"{gen_type}: {description}")

# Get default configuration for a generator
config = GeneratorFactory.get_generator_config("contextual")
print(config)
```

## Extending the Module

### Adding a New Generator Strategy

1. Create a new class inheriting from `BaseGenerator`
2. Implement the required abstract methods
3. Register the new generator with the factory

```python
from generation.base_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    def generate(self, query, retrieved_docs, **kwargs):
        # Implementation here
        pass
    
    def generate_with_metadata(self, query, retrieved_docs, **kwargs):
        # Implementation here
        pass

# Register the new generator
GeneratorFactory.register_generator("custom", CustomGenerator)
```

## Best Practices

1. **Choose the Right Strategy**: Match the generator type to your use case
2. **Configure Parameters**: Adjust temperature and max_tokens based on requirements
3. **Handle Errors**: Always wrap generation calls in try-catch blocks
4. **Validate Inputs**: Use the built-in validation methods
5. **Monitor Performance**: Use metadata to track generation quality

## Error Handling

The module includes comprehensive error handling:

```python
try:
    response = generator.generate(query, documents)
except Exception as e:
    logger.error(f"Generation failed: {e}")
    # Handle error appropriately
```

## Performance Considerations

- **Context Length**: Longer contexts require more processing time
- **Model Selection**: Different LLMs have varying performance characteristics
- **Caching**: Consider implementing response caching for repeated queries
- **Batch Processing**: For multiple queries, consider batch processing

## Integration with RAG System

The generation module integrates seamlessly with the retrieval module:

```python
from retrieval import RetrieverFactory
from generation import GeneratorFactory

# Create retriever and generator
retriever = RetrieverFactory.create_retriever("simple", embeddings, vector_store)
generator = GeneratorFactory.create_generator("contextual", llm)

# Complete RAG pipeline
documents = retriever.retrieve(query)
response = generator.generate(query, documents)
```
