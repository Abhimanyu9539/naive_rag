"""
Vector stores module for the naive RAG system.

This module provides classes for storing and retrieving embeddings from vector databases.
"""

from .pinecone_store import PineconeStore
from .vector_store_processor import VectorStoreProcessor

__all__ = [
    'PineconeStore',
    'VectorStoreProcessor'
]