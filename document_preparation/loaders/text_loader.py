"""
Text document loader using LangChain's TextLoader.
"""

from typing import List
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from .base_loader import BaseDocumentLoader


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for text files using LangChain's TextLoader."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the text loader.
        
        Args:
            encoding: Text encoding to use when reading files
        """
        super().__init__()
        self.supported_extensions = {'.txt', '.md', '.rst', '.log'}
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load text documents from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of LangChain Document objects
        """
        try:
            # Use LangChain's TextLoader
            loader = TextLoader(file_path, encoding=self.encoding)
            documents = loader.load()
            
            # Add metadata
            documents = self.add_metadata(documents, file_path, file_type='text')
            
            return documents
            
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return []
