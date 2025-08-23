# LangChain Document Pipeline for RAG Systems

A complete document processing pipeline using **pure LangChain components**, replacing custom implementations with battle-tested, well-maintained functionality.

## 🚀 Key Features

- **Native LangChain Components**: Uses only LangChain's official document loaders, text splitters, and Document class
- **Token-Based Chunking**: Advanced chunking using tiktoken with GPT tokenizer (cl100k_base)
- **Comprehensive Metadata Preservation**: All document metadata flows through to chunks
- **Multiple File Types**: PDF, Text, DOCX, Web URLs, Markdown, HTML, CSV, JSON
- **Structure-Aware Splitting**: Markdown and HTML header-based splitting
- **Progress Tracking**: Built-in progress bars for large directory processing
- **Error Resilience**: Graceful handling of corrupt/unsupported files
- **Statistics Generation**: Comprehensive analysis of processed documents

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
from document_preparation import DocumentPreparationPipeline

# Initialize pipeline with token-based chunking
pipeline = DocumentPreparationPipeline(
    max_tokens=500,
    overlap_tokens=50,
    use_tokens=True  # Use tiktoken for accurate token counting
)

# Process a single document
chunks = pipeline.process_single_document("document.pdf")

# Process a directory
chunks = pipeline.process_documents("./documents/", file_types=['pdf', 'txt'])

# Get comprehensive statistics
stats = pipeline.get_statistics(chunks)
print(f"Created {stats['total_chunks']} chunks with {stats['total_tokens']} total tokens")
```

## 🔄 Migration from Custom Implementation

### What Changed

| Component | Before (Custom) | After (LangChain) |
|-----------|----------------|-------------------|
| Document Class | Custom `Document` dataclass | `langchain_core.documents.Document` |
| Chunk Class | Custom `Chunk` dataclass | `langchain_core.documents.Document` |
| Loaders | Custom wrapper classes | Direct LangChain loaders |
| Text Splitters | Custom token chunker | `TokenTextSplitter` with tiktoken |
| Preprocessors | Custom text cleaner | Enhanced LangChain-compatible preprocessor |

### Benefits of Migration

✅ **Reduced Code Complexity**: From ~1000 lines to ~400 lines  
✅ **Better Maintenance**: Leverages LangChain's active development  
✅ **Enhanced Features**: Advanced text splitters, better metadata handling  
✅ **Improved Reliability**: Battle-tested components with extensive testing  
✅ **Future-Proof**: Automatic updates with LangChain releases  
✅ **Better Integration**: Native compatibility with LangChain ecosystem  

## 📚 Usage Examples

### 1. Basic Document Processing

```python
from document_preparation import DocumentPreparationPipeline

pipeline = DocumentPreparationPipeline(
    max_tokens=500,
    overlap_tokens=50,
    use_tokens=True
)

# Process single file
chunks = pipeline.process_single_document("document.txt")

# Process directory with filters
chunks = pipeline.process_documents(
    "./documents/", 
    file_types=['pdf', 'txt', 'docx'],
    recursive=True
)
```

### 2. Token vs Character-Based Chunking

```python
# Token-based chunking (recommended for LLMs)
token_pipeline = DocumentPreparationPipeline(
    max_tokens=500,
    overlap_tokens=50,
    use_tokens=True  # Uses tiktoken
)

# Character-based chunking
char_pipeline = DocumentPreparationPipeline(
    max_tokens=2000,  # Character count
    overlap_tokens=200,  # Character overlap
    use_tokens=False
)
```

### 3. Individual Component Usage

```python
from document_preparation.loaders import TextDocumentLoader, PDFDocumentLoader
from document_preparation.chunkers import TokenChunker
from document_preparation.preprocessors import TextPreprocessor

# Load documents
text_loader = TextDocumentLoader()
documents = text_loader.load("document.txt")

# Preprocess
preprocessor = TextPreprocessor(remove_html_tags=True, remove_urls=True)
processed_docs = preprocessor.preprocess_documents(documents)

# Chunk
chunker = TokenChunker(max_tokens=500, overlap_tokens=50)
chunks = chunker.chunk_documents(processed_docs)
```

### 4. Web Document Processing

```python
from document_preparation.loaders import WebDocumentLoader

# Load from URL
web_loader = WebDocumentLoader()
documents = web_loader.load("https://example.com/article")

# Process with pipeline
chunks = pipeline.process_documents("https://example.com/article")
```

### 5. Comprehensive Statistics

```python
# Get detailed statistics
stats = pipeline.get_statistics(chunks)

print(f"Total chunks: {stats['total_chunks']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Average tokens per chunk: {stats['average_tokens_per_chunk']}")
print(f"File type distribution: {stats['file_type_distribution']}")
print(f"Source distribution: {stats['source_distribution']}")
```

## 🏗️ Architecture

```
document_preparation/
├── __init__.py                 # Main exports
├── pipeline.py                 # Main pipeline orchestrator
├── loaders/                    # LangChain document loaders
│   ├── base_loader.py         # Base loader interface
│   ├── text_loader.py         # Text file loader
│   ├── pdf_loader.py          # PDF loader
│   ├── docx_loader.py         # DOCX loader
│   ├── web_loader.py          # Web URL loader
│   └── directory_loader.py    # Directory loader
├── preprocessors/              # Text preprocessing
│   ├── base_preprocessor.py   # Base preprocessor interface
│   └── text_preprocessor.py   # Text cleaning and normalization
├── chunkers/                   # Text chunking
│   ├── base_chunker.py        # Base chunker interface
│   └── token_chunker.py       # Token-based chunking
└── utils.py                   # Utility functions
```

## 🔧 Configuration

### Pipeline Configuration

```python
pipeline = DocumentPreparationPipeline(
    max_tokens=500,              # Maximum tokens per chunk
    overlap_tokens=50,           # Overlap between chunks
    use_tokens=True,             # Use token-based splitting
    preprocessor_config={        # Text preprocessing options
        'remove_extra_whitespace': True,
        'normalize_unicode': True,
        'remove_html_tags': True,
        'remove_urls': True,
        'remove_emails': True,
        'remove_special_chars': False,
        'lowercase': False,
        'remove_numbers': False
    }
)
```

### Supported File Types

- **Text**: `.txt`, `.md`, `.rst`, `.log`
- **PDF**: `.pdf`
- **Word**: `.docx`, `.doc`
- **Web**: URLs (http/https)
- **Markdown**: `.md`
- **HTML**: `.html`, `.htm`
- **Data**: `.csv`, `.json`

## 📊 Statistics and Analysis

The pipeline provides comprehensive statistics:

```python
stats = pipeline.get_statistics(chunks)

# Basic counts
stats['total_chunks']
stats['total_words']
stats['total_characters']
stats['total_tokens']

# Averages
stats['average_words_per_chunk']
stats['average_characters_per_chunk']
stats['average_tokens_per_chunk']

# Distribution analysis
stats['file_type_distribution']
stats['source_distribution']
stats['unique_sources']

# Chunk size analysis
stats['chunk_size_analysis']['min']
stats['chunk_size_analysis']['max']
stats['chunk_size_analysis']['mean']
stats['chunk_size_analysis']['median']
```

## 🚨 Error Handling

The pipeline gracefully handles various error scenarios:

- **Non-existent files**: Returns empty list with warning
- **Unsupported file types**: Skips with warning
- **Corrupt files**: Continues processing other files
- **Network errors**: Handles web loading failures
- **Permission errors**: Logs and continues

## 🔄 Migration Guide

### From Custom Implementation

1. **Update imports**:
   ```python
   # Before
   from document_preparation import Document, Chunk
   
   # After
   from langchain_core.documents import Document
   ```

2. **Update document access**:
   ```python
   # Before
   document.content
   document.title
   
   # After
   document.page_content
   document.metadata.get('title')
   ```

3. **Update chunk access**:
   ```python
   # Before
   chunk.content
   chunk.document_id
   
   # After
   chunk.page_content
   chunk.metadata.get('source')
   ```

4. **Update statistics**:
   ```python
   # Before
   stats = pipeline.get_statistics(chunks)
   
   # After (same interface, enhanced data)
   stats = pipeline.get_statistics(chunks)
   ```

## 🤝 Contributing

This project uses pure LangChain components. To contribute:

1. Ensure all changes use LangChain's native components
2. Maintain backward compatibility where possible
3. Add comprehensive tests for new features
4. Update documentation for any API changes

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain Team**: For providing excellent document processing components
- **OpenAI**: For tiktoken tokenizer
- **Community**: For feedback and contributions

---

**Why LangChain?** By using pure LangChain components, we eliminate custom code maintenance, leverage battle-tested functionality, and ensure compatibility with the broader LangChain ecosystem. The result is a simpler, more reliable, and more powerful document processing pipeline.
