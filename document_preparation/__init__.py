"""
Document Preparation Module for Naive RAG

This module provides tools for loading, preprocessing, and chunking documents
for use in a Retrieval-Augmented Generation (RAG) system.
"""

from .pipeline import DocumentPreparationPipeline
from .loaders import PDFLoader, TextLoader, DocxLoader, WebLoader, DirectoryDocumentLoader
from .preprocessors import TextPreprocessor
from .chunkers import TokenChunker
from .utils import Document, Chunk

__version__ = "1.0.0"
__all__ = [
    "DocumentPreparationPipeline",
    "PDFLoader",
    "TextLoader", 
    "DocxLoader",
    "WebLoader",
    "DirectoryDocumentLoader",
    "TextPreprocessor",
    "TokenChunker",
    "Document",
    "Chunk"
]
