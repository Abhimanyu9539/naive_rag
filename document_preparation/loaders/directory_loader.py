"""
Directory document loader using LangChain's DirectoryLoader.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

from .base_loader import BaseDocumentLoader
from .text_loader import TextDocumentLoader
from .pdf_loader import PDFDocumentLoader
from .docx_loader import DocxDocumentLoader
from .web_loader import WebDocumentLoader


class DirectoryDocumentLoader(BaseDocumentLoader):
    """Loader for directories using LangChain's DirectoryLoader."""
    
    def __init__(self):
        """Initialize the directory loader."""
        super().__init__()
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.md', '.html', '.htm'}
        
        # Initialize individual loaders
        self.loaders = {
            'text': TextDocumentLoader(),
            'pdf': PDFDocumentLoader(),
            'docx': DocxDocumentLoader(),
            'web': WebDocumentLoader()
        }
    
    def load_with_filters(self, 
                         directory_path: str, 
                         file_types: Optional[List[str]] = None,
                         recursive: bool = True) -> List[Document]:
        """
        Load documents from a directory with file type filters.
        
        Args:
            directory_path: Path to the directory
            file_types: List of file extensions to include (e.g., ['pdf', 'txt'])
            recursive: Whether to search subdirectories
            
        Returns:
            List of LangChain Document objects
        """
        try:
            directory_path = Path(directory_path)
            if not directory_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            # Create glob pattern based on file types
            if file_types:
                extensions = [f"*.{ext.lstrip('.')}" for ext in file_types]
                glob_pattern = f"**/*.{{{','.join(extensions)}}}" if recursive else f"*.{{{','.join(extensions)}}}"
            else:
                glob_pattern = "**/*" if recursive else "*"
            
            # Use LangChain's DirectoryLoader with automatic loader selection
            loader = DirectoryLoader(
                str(directory_path),
                glob=glob_pattern,
                loader_cls=self._get_loader_for_extension,
                show_progress=True,
                use_multithreading=True
            )
            
            documents = loader.load()
            
            # Add directory metadata
            for doc in documents:
                if 'directory' not in doc.metadata:
                    doc.metadata['directory'] = str(directory_path)
            
            return documents
            
        except Exception as e:
            print(f"Error loading directory {directory_path}: {e}")
            return []
    
    def _get_loader_for_extension(self, file_path: str):
        """Get the appropriate loader class for a file extension."""
        extension = Path(file_path).suffix.lower()
        
        if extension in {'.txt', '.md', '.rst', '.log'}:
            return TextDocumentLoader
        elif extension == '.pdf':
            return PDFDocumentLoader
        elif extension in {'.docx', '.doc'}:
            return DocxDocumentLoader
        else:
            return TextDocumentLoader  # Default fallback
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a directory (alias for load_with_filters).
        
        Args:
            file_path: Path to the directory
            
        Returns:
            List of LangChain Document objects
        """
        return self.load_with_filters(file_path)
