"""
This module contains the PDFLoader class, which is used to load PDF files into a list of documents.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import os
import logging
from typing import List

# Configure logger for this module
logger = logging.getLogger(__name__)

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
        logger.info(f"Initializing PDFLoader for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            logger.error(f"File must be a PDF: {file_path}")
            raise ValueError(f"File must be a PDF: {file_path}")
        
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path)
        logger.info(f"PDFLoader initialized successfully for: {file_path}")
    
    def load(self) -> List[Document]:
        """
        Load the PDF file and return a list of Document objects.
        
        Returns:
            List[Document]: List of Document objects, one for each page in the PDF
        """
        logger.info(f"Starting to load PDF file: {self.file_path}")
        try:
            documents = self.loader.load()
            logger.info(f"Successfully loaded PDF file: {self.file_path}. Found {len(documents)} pages/documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF file {self.file_path}: {str(e)}")
            raise Exception(f"Error loading PDF file {self.file_path}: {str(e)}")
    

