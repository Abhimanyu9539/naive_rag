"""
This module contains the PDFLoader class, which is used to load PDF files into a list of documents.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import os
from typing import List

class PDFLoader:
    """
    This class is used to load PDF files into a list of documents.
    """
    def __init__(self, file_path: str):
        """
        Initialize the PDFLoader with a file path.
        
        Args:
            file_path (str): Path to the PDF file to load
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {file_path}")
        
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path)
    
    def load(self) -> List[Document]:
        """
        Load the PDF file and return a list of Document objects.
        
        Returns:
            List[Document]: List of Document objects, one for each page in the PDF
        """
        try:
            documents = self.loader.load()
            return documents
        except Exception as e:
            raise Exception(f"Error loading PDF file {self.file_path}: {str(e)}")
    

