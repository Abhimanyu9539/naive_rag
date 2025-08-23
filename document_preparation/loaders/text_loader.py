"""
Text file loader for plain text documents using LangChain.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader as LangChainTextLoader

from .base_loader import BaseLoader
from ..utils import Document


class TextLoader(BaseLoader):
    """Loader for plain text files using LangChain."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the text loader.
        
        Args:
            encoding: Text encoding to use when reading files
        """
        super().__init__(encoding)
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the given file."""
        return Path(file_path).suffix.lower() == '.txt'
    
    def load(self, file_path: str) -> Document:
        """
        Load a text file using LangChain's TextLoader.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Document object containing the text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.can_load(file_path):
            raise ValueError(f"File {file_path} is not a supported text file")
        
        try:
            # Use LangChain's TextLoader
            langchain_loader = LangChainTextLoader(file_path, encoding=self.encoding)
            langchain_docs = langchain_loader.load()
            
            if not langchain_docs:
                raise ValueError(f"No content loaded from {file_path}")
            
            # Convert LangChain document to our format
            document = self._convert_langchain_document(langchain_docs[0], file_path)
            
            # Add file metadata
            file_metadata = self._get_file_metadata(file_path)
            document.metadata.update(file_metadata)
            
            return document
            
        except Exception as e:
            raise ValueError(f"Error reading text file {file_path}: {e}")
