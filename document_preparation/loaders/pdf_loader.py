"""
PDF document loader using LangChain's PyPDFLoader.
"""

from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .base_loader import BaseDocumentLoader


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF files using LangChain's PyPDFLoader."""
    
    def __init__(self):
        """Initialize the PDF loader."""
        super().__init__()
        self.supported_extensions = {'.pdf'}
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load PDF documents from a file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of LangChain Document objects (one per page)
        """
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            documents = self.add_metadata(documents, file_path, file_type='pdf')
            
            return documents
            
        except Exception as e:
            print(f"Error loading PDF file {file_path}: {e}")
            return []
