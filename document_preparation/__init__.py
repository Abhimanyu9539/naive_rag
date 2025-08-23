"""
Document preparation pipeline using LangChain components.

This module provides a complete document processing pipeline using only LangChain components,
replacing custom implementations with battle-tested, well-maintained functionality.

Key Features:
- Native LangChain document loaders (PDF, Text, DOCX, Web, Directory)
- Advanced text splitters with token-based chunking using tiktoken
- Comprehensive metadata preservation
- Structure-aware splitting for Markdown and HTML
- Progress tracking and error resilience
- Statistics generation and analysis
"""

from .pipeline import DocumentPreparationPipeline
from .loaders import (
    BaseDocumentLoader,
    TextDocumentLoader,
    PDFDocumentLoader,
    DocxDocumentLoader,
    WebDocumentLoader,
    DirectoryDocumentLoader
)
from .preprocessors import BasePreprocessor, TextPreprocessor
from .chunkers import BaseChunker, TokenChunker, CharacterChunker
from .utils import (
    get_file_extension,
    is_supported_file,
    sanitize_filename,
    create_chunk_id,
    merge_chunks,
    add_metadata_to_document,
    create_document_from_text,
    get_document_statistics
)

__all__ = [
    # Main pipeline
    'DocumentPreparationPipeline',
    
    # Loaders
    'BaseDocumentLoader',
    'TextDocumentLoader',
    'PDFDocumentLoader',
    'DocxDocumentLoader',
    'WebDocumentLoader',
    'DirectoryDocumentLoader',
    
    # Preprocessors
    'BasePreprocessor',
    'TextPreprocessor',
    
    # Chunkers
    'BaseChunker',
    'TokenChunker',
    'CharacterChunker',
    
    # Utilities
    'get_file_extension',
    'is_supported_file',
    'sanitize_filename',
    'create_chunk_id',
    'merge_chunks',
    'add_metadata_to_document',
    'create_document_from_text',
    'get_document_statistics'
]

__version__ = "2.0.0"
