"""
Web document loader using LangChain's WebBaseLoader.
"""

from typing import List
from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from .base_loader import BaseDocumentLoader


class WebDocumentLoader(BaseDocumentLoader):
    """Loader for web documents using LangChain's WebBaseLoader."""
    
    def __init__(self):
        """Initialize the web loader."""
        super().__init__()
        # Web loader doesn't have file extensions, but we can check URLs
        self.supported_extensions = set()
    
    def can_load(self, file_path: str) -> bool:
        """
        Check if this loader can handle the given URL.
        
        Args:
            file_path: URL to check
            
        Returns:
            True if the path is a valid URL
        """
        try:
            result = urlparse(file_path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a web URL.
        
        Args:
            file_path: URL to load from
            
        Returns:
            List of LangChain Document objects
        """
        try:
            # Use LangChain's WebBaseLoader
            loader = WebBaseLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            documents = self.add_metadata(documents, file_path, file_type='web')
            
            return documents
            
        except Exception as e:
            print(f"Error loading web document {file_path}: {e}")
            return []
