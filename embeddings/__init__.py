"""
Embeddings module for the naive RAG system.

This module provides classes for generating embeddings from text and documents.
"""

from .base_embedder import BaseEmbedder
from .openai_embedder import OpenAIEmbedder
from .embedder_factory import EmbedderFactory
from .document_processor import DocumentProcessor

__all__ = [
    'BaseEmbedder',
    'OpenAIEmbedder',
    'EmbedderFactory',
    'DocumentProcessor'
]
