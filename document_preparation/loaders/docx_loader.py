"""
DOCX file loader for Microsoft Word documents using LangChain.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader

from .base_loader import BaseLoader
from ..utils import Document


class DocxLoader(BaseLoader):
    """Loader for DOCX files using LangChain."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the DOCX loader.
        
        Args:
            encoding: Text encoding (not used for DOCX but kept for consistency)
        """
        super().__init__(encoding)
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the given file."""
        return Path(file_path).suffix.lower() in ['.docx', '.doc']
    
    def load(self, file_path: str) -> Document:
        """
        Load a DOCX file using LangChain's Docx2txtLoader.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Document object containing the extracted text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.can_load(file_path):
            raise ValueError(f"File {file_path} is not a supported DOCX file")
        
        try:
            # Use LangChain's Docx2txtLoader
            langchain_loader = Docx2txtLoader(file_path)
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
            raise ValueError(f"Error reading DOCX file {file_path}: {e}")
