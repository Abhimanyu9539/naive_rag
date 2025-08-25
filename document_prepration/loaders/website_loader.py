"""
This module contains the WebsiteLoader class, which is used to load content from websites into a list of documents.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from urllib.parse import urlparse
import logging
from typing import List, Optional

# Configure logger for this module
logger = logging.getLogger(__name__)

class WebsiteLoader:
    """
    This class is used to load content from websites into a list of documents.
    """
    def __init__(self, url: str, requests_kwargs: Optional[dict] = None):
        """
        Initialize the WebsiteLoader with a URL.
        
        Args:
            url (str): URL of the website to load
            requests_kwargs (dict, optional): Additional arguments to pass to requests
        """
        logger.info(f"Initializing WebsiteLoader for URL: {url}")
        
        if not self._is_valid_url(url):
            logger.error(f"Invalid URL format: {url}")
            raise ValueError(f"Invalid URL format: {url}")
        
        self.url = url
        self.requests_kwargs = requests_kwargs or {}
        self.loader = WebBaseLoader(url, requests_kwargs=self.requests_kwargs)
        logger.info(f"WebsiteLoader initialized successfully for: {url}")
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate if the provided string is a valid URL.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if valid URL, False otherwise
        """
        try:
            result = urlparse(url)
            is_valid = all([result.scheme, result.netloc])
            if is_valid:
                logger.debug(f"URL validation successful: {url}")
            else:
                logger.debug(f"URL validation failed: {url} - missing scheme or netloc")
            return is_valid
        except Exception as e:
            logger.debug(f"URL validation failed: {url} - {str(e)}")
            return False
    
    def load(self) -> List[Document]:
        """
        Load the website content and return a list of Document objects.
        
        Returns:
            List[Document]: List of Document objects from the website
        """
        logger.info(f"Starting to load website: {self.url}")
        try:
            documents = self.loader.load()
            logger.info(f"Successfully loaded website: {self.url}. Found {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading website {self.url}: {str(e)}")
            raise Exception(f"Error loading website {self.url}: {str(e)}")
    
    def load_with_metadata(self, custom_metadata: Optional[dict] = None) -> List[Document]:
        """
        Load the website content with custom metadata.
        
        Args:
            custom_metadata (dict, optional): Additional metadata to add to documents
            
        Returns:
            List[Document]: List of Document objects with metadata
        """
        logger.info(f"Starting to load website with metadata: {self.url}")
        try:
            documents = self.loader.load()
            
            # Add custom metadata if provided
            if custom_metadata:
                logger.info(f"Adding custom metadata to {len(documents)} documents")
                for doc in documents:
                    doc.metadata.update(custom_metadata)
                    doc.metadata['source_url'] = self.url
            
            logger.info(f"Successfully loaded website with metadata: {self.url}. Found {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading website {self.url}: {str(e)}")
            raise Exception(f"Error loading website {self.url}: {str(e)}")
