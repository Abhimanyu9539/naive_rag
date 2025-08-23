"""
PDF file loader for PDF documents using LangChain.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader

from .base_loader import BaseLoader
from ..utils import Document


class PDFLoader(BaseLoader):
    """Loader for PDF files using LangChain."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the PDF loader.
        
        Args:
            encoding: Text encoding (not used for PDFs but kept for consistency)
        """
        super().__init__(encoding)
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the given file."""
        return Path(file_path).suffix.lower() == '.pdf'
    
    def load(self, file_path: str) -> Document:
        """
        Load a PDF file using LangChain's PyPDFLoader.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document object containing the extracted text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.can_load(file_path):
            raise ValueError(f"File {file_path} is not a supported PDF file")
        
        try:
            # Use LangChain's PyPDFLoader
            langchain_loader = PyPDFLoader(file_path)
            langchain_docs = langchain_loader.load()
            
            if not langchain_docs:
                raise ValueError(f"No content loaded from {file_path}")
            
            # Combine all pages into one document
            combined_content = "\n\n".join([doc.page_content for doc in langchain_docs])
            
            # Use the first document's metadata as base
            base_metadata = langchain_docs[0].metadata.copy() if langchain_docs[0].metadata else {}
            
            # Add page count
            base_metadata['num_pages'] = len(langchain_docs)
            
            # Create a combined LangChain document
            combined_doc = type(langchain_docs[0])(
                page_content=combined_content,
                metadata=base_metadata
            )
            
            # Convert to our format
            document = self._convert_langchain_document(combined_doc, file_path)
            
            # Add file metadata
            file_metadata = self._get_file_metadata(file_path)
            document.metadata.update(file_metadata)
            
            return document
            
        except Exception as e:
            raise ValueError(f"Error reading PDF file {file_path}: {e}")
