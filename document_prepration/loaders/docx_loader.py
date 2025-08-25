"""
This module contains the DOCXLoader class, which is used to load DOCX files into a list of documents.
"""

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
import os
import logging
from typing import List

# Configure logger for this module
logger = logging.getLogger(__name__)

class DOCXLoader:
    """
    This class is used to load DOCX files into a list of documents.
    """
    def __init__(self, file_path: str):
        """
        Initialize the DOCXLoader with a file path.
        
        Args:
            file_path (str): Path to the DOCX file to load
        """
        logger.info(f"Initializing DOCXLoader for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"DOCX file not found: {file_path}")
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        
        if not file_path.lower().endswith(('.docx', '.doc')):
            logger.error(f"File must be a DOCX or DOC file: {file_path}")
            raise ValueError(f"File must be a DOCX or DOC file: {file_path}")
        
        self.file_path = file_path
        self.loader = Docx2txtLoader(file_path)
        logger.info(f"DOCXLoader initialized successfully for: {file_path}")
    
    def load(self) -> List[Document]:
        """
        Load the DOCX file and return a list of Document objects.
        
        Returns:
            List[Document]: List of Document objects from the DOCX file
        """
        logger.info(f"Starting to load DOCX file: {self.file_path}")
        try:
            documents = self.loader.load()
            logger.info(f"Successfully loaded DOCX file: {self.file_path}. Found {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading DOCX file {self.file_path}: {str(e)}")
            raise Exception(f"Error loading DOCX file {self.file_path}: {str(e)}")
