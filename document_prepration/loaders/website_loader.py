"""
This module contains the WebsiteLoader class, which is used to load content from websites into a list of documents.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from urllib.parse import urlparse
from typing import List, Optional

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
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        
        self.url = url
        self.requests_kwargs = requests_kwargs or {}
        self.loader = WebBaseLoader(url, requests_kwargs=self.requests_kwargs)
    
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
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def load(self) -> List[Document]:
        """
        Load the website content and return a list of Document objects.
        
        Returns:
            List[Document]: List of Document objects from the website
        """
        try:
            documents = self.loader.load()
            return documents
        except Exception as e:
            raise Exception(f"Error loading website {self.url}: {str(e)}")
    
    def load_with_metadata(self, custom_metadata: Optional[dict] = None) -> List[Document]:
        """
        Load the website content with custom metadata.
        
        Args:
            custom_metadata (dict, optional): Additional metadata to add to documents
            
        Returns:
            List[Document]: List of Document objects with metadata
        """
        try:
            documents = self.loader.load()
            
            # Add custom metadata if provided
            if custom_metadata:
                for doc in documents:
                    doc.metadata.update(custom_metadata)
                    doc.metadata['source_url'] = self.url
            
            return documents
        except Exception as e:
            raise Exception(f"Error loading website {self.url}: {str(e)}")
