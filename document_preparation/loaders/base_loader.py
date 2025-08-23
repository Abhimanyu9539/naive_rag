"""
Base loader class that defines the interface for all document loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import os

from langchain_community.document_loaders.base import BaseLoader as LangChainBaseLoader
from langchain.schema import Document as LangChainDocument

from ..utils import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the loader.
        
        Args:
            encoding: Text encoding to use when reading files
        """
        self.encoding = encoding
    
    @abstractmethod
    def load(self, file_path: str) -> Document:
        """
        Load a single document from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object containing the loaded content and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        pass
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of Document objects
        """
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if recursive:
            file_paths = directory.rglob('*')
        else:
            file_paths = directory.glob('*')
        
        for file_path in file_paths:
            if file_path.is_file() and self.can_load(str(file_path)):
                try:
                    document = self.load(str(file_path))
                    documents.append(document)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        return documents
    
    @abstractmethod
    def can_load(self, file_path: str) -> bool:
        """
        Check if this loader can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the loader can handle this file type
        """
        pass
    
    def _get_file_metadata(self, file_path: str) -> dict:
        """
        Get basic file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        path = Path(file_path)
        stat = path.stat()
        
        return {
            'file_size': stat.st_size,
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'file_extension': path.suffix.lower(),
            'file_name': path.name,
            'file_path': str(path.absolute())
        }
    
    def _convert_langchain_document(self, langchain_doc: LangChainDocument, source: str) -> Document:
        """
        Convert a LangChain document to our Document format.
        
        Args:
            langchain_doc: LangChain document object
            source: Source file path or URL
            
        Returns:
            Our Document object
        """
        # Extract metadata
        metadata = langchain_doc.metadata.copy() if langchain_doc.metadata else {}
        
        # Get title from metadata or filename
        title = metadata.get('title') or metadata.get('source') or Path(source).stem
        
        # Get author from metadata
        author = metadata.get('author') or metadata.get('creator')
        
        return Document(
            content=langchain_doc.page_content,
            source=source,
            title=title,
            author=author,
            metadata=metadata
        )
