"""
Utility functions for document preparation using LangChain components.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib
from datetime import datetime

# Use LangChain's native Document class instead of custom ones
from langchain_core.documents import Document


def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    return Path(file_path).suffix.lower()


def is_supported_file(file_path: str) -> bool:
    """Check if the file type is supported by LangChain loaders."""
    supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.html', '.htm', '.csv', '.json'}
    return get_file_extension(file_path) in supported_extensions


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing invalid characters."""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    return sanitized


def create_chunk_id(document_id: str, chunk_index: int) -> str:
    """Create a unique chunk ID for LangChain documents."""
    return f"{document_id}_chunk_{chunk_index:04d}"


def merge_chunks(chunks: List[Document]) -> str:
    """Merge multiple LangChain document chunks back into a single text."""
    return " ".join(chunk.page_content for chunk in chunks)


def add_metadata_to_document(doc: Document, **kwargs) -> Document:
    """Add metadata to a LangChain document."""
    doc.metadata.update(kwargs)
    return doc


def create_document_from_text(content: str, source: str, **metadata) -> Document:
    """Create a LangChain Document from text content."""
    doc_metadata = {
        'source': source,
        'created_date': datetime.now(),
        **metadata
    }
    return Document(page_content=content, metadata=doc_metadata)


def get_document_statistics(documents: List[Document]) -> Dict[str, Any]:
    """Get statistics about a list of LangChain documents."""
    if not documents:
        return {}
    
    total_docs = len(documents)
    total_words = sum(len(doc.page_content.split()) for doc in documents)
    total_chars = sum(len(doc.page_content) for doc in documents)
    
    # Source analysis
    sources = {}
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    return {
        'total_documents': total_docs,
        'total_words': total_words,
        'total_characters': total_chars,
        'average_words_per_document': round(total_words / total_docs, 2) if total_docs > 0 else 0,
        'average_characters_per_document': round(total_chars / total_docs, 2) if total_docs > 0 else 0,
        'unique_sources': len(sources),
        'source_distribution': sources
    }
