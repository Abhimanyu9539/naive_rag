"""
DOCX document loader using LangChain's Docx2txtLoader.
"""

from typing import List
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from .base_loader import BaseDocumentLoader


class DocxDocumentLoader(BaseDocumentLoader):
    """Loader for DOCX files using LangChain's Docx2txtLoader."""
    
    def __init__(self):
        """Initialize the DOCX loader."""
        super().__init__()
        self.supported_extensions = {'.docx', '.doc'}
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load DOCX documents from a file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List of LangChain Document objects
        """
        try:
            # Use LangChain's Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            documents = self.add_metadata(documents, file_path, file_type='docx')
            
            return documents
            
        except Exception as e:
            print(f"Error loading DOCX file {file_path}: {e}")
            return []
