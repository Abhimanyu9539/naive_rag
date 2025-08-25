"""
This module contains the TXTLoader class, which is used to load text files into a list of documents.
"""

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
import os
import logging
from typing import List, Optional

# Configure logger for this module
logger = logging.getLogger(__name__)


class TXTLoader:
    """
    This class is used to load text files into a list of documents.
    """
    def __init__(self, file_path: str, encoding: Optional[str] = None):
        """
        Initialize the TXTLoader with a file path.
        
        Args:
            file_path (str): Path to the text file to load
            encoding (str, optional): Encoding to use when reading the file (default: auto-detect)
        """
        logger.info(f"Initializing TXTLoader for file: {file_path} with encoding: {encoding}")
        
        if not os.path.exists(file_path):
            logger.error(f"Text file not found: {file_path}")
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        if not file_path.lower().endswith(('.txt', '.text', '.md', '.rst')):
            logger.error(f"File must be a text file (.txt, .text, .md, .rst): {file_path}")
            raise ValueError(f"File must be a text file (.txt, .text, .md, .rst): {file_path}")
        
        self.file_path = file_path
        self.encoding = encoding
        self.loader = TextLoader(file_path, encoding=encoding)
        logger.info(f"TXTLoader initialized successfully for: {file_path}")
    
    def load(self) -> List[Document]:
        """
        Load the text file and return a list of Document objects.
        
        Returns:
            List[Document]: List of Document objects from the text file
        """
        logger.info(f"Starting to load text file: {self.file_path}")
        try:
            documents = self.loader.load()
            logger.info(f"Successfully loaded text file: {self.file_path}. Found {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading text file {self.file_path}: {str(e)}")
            raise Exception(f"Error loading text file {self.file_path}: {str(e)}")
    
    def load_with_metadata(self, custom_metadata: Optional[dict] = None) -> List[Document]:
        """
        Load the text file with custom metadata.
        
        Args:
            custom_metadata (dict, optional): Additional metadata to add to documents
            
        Returns:
            List[Document]: List of Document objects with metadata
        """
        logger.info(f"Starting to load text file with metadata: {self.file_path}")
        try:
            documents = self.loader.load()
            
            # Add custom metadata if provided
            if custom_metadata:
                logger.info(f"Adding custom metadata to {len(documents)} documents")
                for doc in documents:
                    doc.metadata.update(custom_metadata)
                    doc.metadata['source_file'] = self.file_path
                    doc.metadata['file_type'] = 'text'
            
            logger.info(f"Successfully loaded text file with metadata: {self.file_path}. Found {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading text file {self.file_path}: {str(e)}")
            raise Exception(f"Error loading text file {self.file_path}: {str(e)}")



if __name__ == "__main__":
    loader = TXTLoader(r"E:\Gen AI\RAG_Projects\naive_rag\data\raw\attention.txt")
    loader.load()