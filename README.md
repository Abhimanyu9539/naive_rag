# Naive RAG - Document Preparation Module

This module handles the document preparation phase of a naive RAG (Retrieval-Augmented Generation) system using LangChain's document loaders.

## Features

- **Document Loading**: Support for PDF, TXT, DOCX, and web pages using LangChain loaders
- **Text Preprocessing**: Cleaning, normalization, and special character handling
- **Document Chunking**: Intelligent splitting into manageable pieces (200-1000 tokens)
- **Modular Design**: Each component can be used independently
- **LangChain Integration**: Leverages LangChain's robust document loading capabilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from document_preparation import DocumentPreparationPipeline

# Initialize the pipeline
pipeline = DocumentPreparationPipeline()

# Process documents
chunks = pipeline.process_documents("path/to/documents/")
```

### Using Directory Loader

```python
from document_preparation import DirectoryDocumentLoader

# Load all supported documents from a directory
loader = DirectoryDocumentLoader()
documents = loader.load_with_filters("path/to/documents/", file_types=['txt', 'pdf'])
```

### Individual Components

```python
from document_preparation.loaders import PDFLoader, TextLoader
from document_preparation.preprocessors import TextPreprocessor
from document_preparation.chunkers import TokenChunker

# Load documents using LangChain loaders
loader = PDFLoader()
document = loader.load("document.pdf")

# Preprocess text
preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess(document.content)

# Chunk documents
chunker = TokenChunker(max_tokens=500)
chunks = chunker.chunk_document(document)
```

## Supported File Types

- **Text Files** (.txt) - Using LangChain's TextLoader
- **PDF Files** (.pdf) - Using LangChain's PyPDFLoader
- **Word Documents** (.docx, .doc) - Using LangChain's Docx2txtLoader
- **Web Pages** (URLs) - Using LangChain's WebBaseLoader

## Project Structure

```
document_preparation/
├── __init__.py
├── loaders/
│   ├── __init__.py
│   ├── base_loader.py
│   ├── pdf_loader.py
│   ├── text_loader.py
│   ├── docx_loader.py
│   ├── web_loader.py
│   └── directory_loader.py
├── preprocessors/
│   ├── __init__.py
│   ├── base_preprocessor.py
│   └── text_preprocessor.py
├── chunkers/
│   ├── __init__.py
│   ├── base_chunker.py
│   └── token_chunker.py
├── pipeline.py
└── utils.py
```

## Configuration

The system can be configured through environment variables or direct parameter passing. See individual component documentation for details.

## LangChain Integration Benefits

- **Robust Document Loading**: LangChain's loaders handle edge cases and encoding issues
- **Better Metadata Extraction**: Automatic extraction of document metadata
- **Consistent Interface**: Standardized document format across all loaders
- **Extensibility**: Easy to add new document types using LangChain's ecosystem
- **Error Handling**: Improved error handling and recovery mechanisms
