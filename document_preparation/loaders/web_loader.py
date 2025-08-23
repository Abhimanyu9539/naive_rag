"""
Web page loader for HTML content using LangChain.
"""

import os
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader

from .base_loader import BaseLoader
from ..utils import Document


class WebLoader(BaseLoader):
    """Loader for web pages using LangChain."""
    
    def __init__(self, encoding: str = 'utf-8', timeout: int = 10):
        """
        Initialize the web loader.
        
        Args:
            encoding: Text encoding to use
            timeout: Request timeout in seconds
        """
        super().__init__(encoding)
        self.timeout = timeout
    
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the given URL."""
        # Check if it's a URL
        parsed = urlparse(file_path)
        return bool(parsed.scheme and parsed.netloc)
    
    def load(self, url: str) -> Document:
        """
        Load a web page using LangChain's WebBaseLoader.
        
        Args:
            url: URL of the web page to load
            
        Returns:
            Document object containing the extracted text
        """
        if not self.can_load(url):
            raise ValueError(f"Invalid URL: {url}")
        
        try:
            # Use LangChain's WebBaseLoader
            langchain_loader = WebBaseLoader(
                web_paths=[url],
                requests_kwargs={'timeout': self.timeout}
            )
            langchain_docs = langchain_loader.load()
            
            if not langchain_docs:
                raise ValueError(f"No content loaded from {url}")
            
            # Convert LangChain document to our format
            document = self._convert_langchain_document(langchain_docs[0], url)
            
            # Add URL-specific metadata
            url_metadata = {
                'url': url,
                'domain': urlparse(url).netloc,
                'content_type': 'web_page'
            }
            document.metadata.update(url_metadata)
            
            return document
            
        except Exception as e:
            raise ValueError(f"Error loading web page {url}: {e}")
